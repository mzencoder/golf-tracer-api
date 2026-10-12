"""ONNX golf-ball detections, temporal association, and honest tracer rendering."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from branding import VideoBranding


MODEL_PATH = Path(__file__).parent / "models" / "best.onnx"


@dataclass
class Point:
    frame: int
    x: float
    y: float
    confidence: float
    source: str = "detected"


class BallDetector:
    def __init__(self, path: Path = MODEL_PATH, confidence: float = 0.25,
                 timings: dict[str, float] | None = None):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        output = self.session.get_outputs()[0].shape
        if output[1] != 5:
            raise ValueError(f"Expected single-class YOLO output [1, 5, N], got {output}")
        self.confidence = confidence
        self.timings = timings

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Point]:
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        new_w, new_h = round(width * scale), round(height * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        left, top = (640 - new_w) // 2, (640 - new_h) // 2
        canvas = cv2.copyMakeBorder(resized, top, 640 - new_h - top,
                                     left, 640 - new_w - left, cv2.BORDER_CONSTANT,
                                     value=(114, 114, 114))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        inference_started = time.perf_counter()
        try:
            raw = self.session.run(None, {self.input_name: tensor})[0][0].T
        finally:
            if self.timings is not None:
                self.timings["inference_s"] = (self.timings.get("inference_s", 0.0)
                                                 + time.perf_counter() - inference_started)
        raw = raw[raw[:, 4] >= self.confidence]
        if len(raw) == 0:
            return []
        boxes, scores = [], []
        # This export uses normalized xywh coordinates, unlike common YOLO pixel outputs.
        for cx, cy, bw, bh, confidence in raw:
            boxes.append([int((cx - bw / 2) * 640), int((cy - bh / 2) * 640),
                          max(1, int(bw * 640)), max(1, int(bh * 640))])
            scores.append(float(confidence))
        selected = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, 0.45)
        points = []
        for index in np.array(selected).reshape(-1):
            cx, cy, _, _, confidence = raw[index]
            x, y = (float(cx) * 640 - left) / scale, (float(cy) * 640 - top) / scale
            if 0 <= x < width and 0 <= y < height:
                points.append(Point(frame_id, x, y, float(confidence)))
        return points


def select_track(detections: list[list[Point]], width: int, height: int) -> list[Point]:
    """Prefer a coherent moving track; reject static false positives."""
    tracks: list[list[Point]] = []
    max_jump = max(width, height) * 0.20
    for candidates in detections:
        used = set()
        for track in sorted(tracks, key=lambda t: t[-1].frame, reverse=True):
            last = track[-1]
            gap = candidates[0].frame - last.frame if candidates else 1
            if gap > 4:
                continue
            choices = [(np.hypot(point.x - last.x, point.y - last.y), index, point)
                       for index, point in enumerate(candidates) if index not in used]
            choices = [choice for choice in choices if choice[0] <= max_jump * max(gap, 1)]
            if choices:
                _, index, point = min(choices, key=lambda c: c[0])
                track.append(point)
                used.add(index)
        tracks.extend([[point] for index, point in enumerate(candidates) if index not in used])
    def score(track: list[Point]) -> float:
        if len(track) < 4:
            return -1
        movement = sum(np.hypot(b.x - a.x, b.y - a.y) for a, b in zip(track, track[1:]))
        span = np.hypot(track[-1].x - track[0].x, track[-1].y - track[0].y)
        duration = track[-1].frame - track[0].frame
        # A shot must begin near the tee/ground, not on a moving head or cloud.
        if (track[0].y < height * 0.5 or
                span < max(width, height) * 0.05 or
                span / max(movement, 1) < 0.65 or
                span / max(duration, 1) < max(width, height) * 0.0015):
            return -1
        return len(track) + min(span / max(width, height), 1) * 8
    best = max(tracks, key=score, default=[])
    return best if score(best) > 0 else []


def select_launch_seed(detections: list[list[Point]], width: int, height: int,
                       fps: float) -> Point | None:
    """Find a stable tee ball that disappears around the shot, avoiding edge markers."""
    bins: dict[tuple[int, int], list[Point]] = defaultdict(list)
    for candidates in detections:
        for point in candidates:
            if (0.2 * width <= point.x <= 0.8 * width and
                    0.5 * height <= point.y <= 0.96 * height and
                    point.confidence >= 0.35):
                bins[(round(point.x / 16), round(point.y / 16))].append(point)
    options = []
    min_seen = max(8, round(fps * 0.5))
    min_absent = max(6, round(fps * 0.25))
    for points in bins.values():
        if len(points) < min_seen:
            continue
        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])
        center = (float(np.median(xs)), float(np.median(ys)))
        if np.median(np.hypot(xs - center[0], ys - center[1])) > 10:
            continue
        last = max(p.frame for p in points)
        if last < len(detections) * 0.2 or last > len(detections) - min_absent:
            continue
        # A genuine launch disappears after a long stable observation.
        recent = sum(p.frame >= last - round(fps) for p in points)
        if recent < max(5, round(fps * 0.2)):
            continue
        center_bonus = 1 - abs(center[0] / width - 0.5)
        options.append((recent * center_bonus + np.mean([p.confidence for p in points]) * 10 +
                        20 * last / len(detections),
                        Point(last, center[0], center[1], float(np.mean([p.confidence for p in points])))))
    if not options:
        return None
    seed = max(options, key=lambda item: item[0])[1]
    # Quantized bins can split one tee ball as the club approaches it. Continue
    # through nearby detections so the estimate starts after the last sighting.
    nearby = [point for candidates in detections for point in candidates
              if point.frame >= seed.frame and point.confidence >= 0.35 and
              np.hypot(point.x - seed.x, point.y - seed.y) <= 20]
    return Point(max(point.frame for point in nearby), seed.x, seed.y,
                 seed.confidence) if nearby else seed


def detect_tee_crop(source: Path, detector: BallDetector,
                    width: int, height: int) -> list[list[Point]]:
    """Retry a high-resolution shot around the tee at a useful model scale."""
    if height > width * 1.25:
        x0, x1 = round(width * 0.35), round(width * 0.75)
        y0, y1 = round(height * 0.48), round(height * 0.69)
    else:
        x0, x1 = round(width * 0.35), round(width * 0.65)
        y0, y1 = round(height * 0.65), height
    cap = cv2.VideoCapture(str(source))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            points = detector.detect(frame[y0:y1, x0:x1], len(frames))
            for point in points:
                point.x += x0
                point.y += y0
            frames.append(points)
    finally:
        cap.release()
    return frames


def track_motion_from_seed(source: Path, seed: Point, fps: float,
                           width: int, height: int, total: int) -> list[Point]:
    """Follow a small moving image feature just after a model-confirmed tee ball disappears."""
    cap = cv2.VideoCapture(str(source))
    left = max(0, round(seed.x - max(24, width * 0.035)))
    right = min(width, round(seed.x + max(24, width * 0.035)))
    top = max(0, round(seed.y - height * 0.58))
    bottom = min(height, round(seed.y - height * 0.035))
    start = seed.frame + 1
    end = min(total, start + round(fps * 2.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, seed.frame))
    previous = None
    observations: list[Point] = []
    misses = 0
    try:
        for frame_id in range(max(0, seed.frame), end):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            if previous is None:
                previous = gray
                continue
            difference = cv2.absdiff(gray, previous)
            previous = gray
            _, mask = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
            candidates = []
            for index in range(1, count):
                _, _, blob_w, blob_h, area = stats[index]
                if not (2 <= area <= 45 and blob_w <= 12 and blob_h <= 12):
                    continue
                x = float(centroids[index][0] + left)
                y = float(centroids[index][1] + top)
                if abs(x - seed.x) <= max(24, width * 0.035):
                    candidates.append(Point(frame_id, x, y, 0.5, "motion"))
            if not observations:
                if frame_id - seed.frame > max(8, round(fps * 0.18)):
                    break
                choices = [p for p in candidates if
                           seed.y - height * 0.25 <= p.y <= seed.y - height * 0.07]
                if choices:
                    observations.append(max(choices, key=lambda p: p.y))
                continue
            last = observations[-1]
            gap = frame_id - last.frame
            plausible = [p for p in candidates if
                         abs(p.x - last.x) <= max(12, width * 0.015) and
                         p.y <= last.y - 1.5 and
                         last.y - p.y <= max(height * 0.10, 20) * gap]
            if plausible:
                # The old ball position also appears in a frame difference. Choose the new,
                # higher position while the launch is still rising.
                next_point = min(plausible, key=lambda p: p.y + 2 * abs(p.x - last.x))
                observations.append(next_point)
                misses = 0
                continue
            misses += 1
            if misses > max(6, round(fps * 0.13)):
                break
    finally:
        cap.release()
    if len(observations) < 6:
        return []
    rise = observations[0].y - observations[-1].y
    upward = sum(b.y < a.y for a, b in zip(observations, observations[1:]))
    if rise < height * 0.12 or upward / max(len(observations) - 1, 1) < 0.7:
        return []
    if len(observations) < 10:
        steps = [a.y - b.y for a, b in zip(observations, observations[1:])]
        # A short apparent flight that abruptly settles on a fixed background
        # feature is not enough evidence for a measured tracer.
        if (max(steps[:3]) > height * 0.05 and
                min(steps[-2:]) < max(5, height * 0.002)):
            return []
    return [seed] + observations


def build_prediction(track: list[Point], seed: Point | None, total: int,
                     width: int, height: int, fps: float) -> list[Point]:
    """Extrapolate measured flight, or make a labeled visual estimate from the tee."""
    if track:
        first, last = track[0], track[-1]
        if first.y - last.y < 0.04 * height or last.frame >= total - 2:
            return []
        tail = track[-min(8, len(track)):]
        times = np.array([(p.frame - last.frame) / fps for p in tail])
        if len(np.unique(times)) < 2:
            return []
        vx = float(np.polyfit(times, [p.x for p in tail], 1)[0])
        vy = float(np.polyfit(times, [p.y for p in tail], 1)[0])
        if vy >= 0:
            return []
        vx = float(np.clip(vx, -width * 0.25, width * 0.25))
        duration = 1.65
        if (total - last.frame - 1) / fps < 0.25:
            return []
        # Far ground is higher in the image than the tee foreground. This is a
        # screen-space estimate; one camera cannot measure the landing distance.
        landing_y = max(last.y + height * 0.05,
                        min(first.y - height * 0.10, height * 0.55))
        gravity = max(0, 2 * (landing_y - last.y - vy * duration) / duration**2)
        points = []
        for frame in range(last.frame + 1, min(total, last.frame + round(duration * fps) + 1)):
            t = (frame - last.frame) / fps
            x = last.x + vx * t
            y = last.y + vy * t + 0.5 * gravity * t * t
            if not (0 <= x < width and 0 <= y < height):
                break
            points.append(Point(frame, x, y, 0, "predicted"))
        return points
    if seed is None:
        return []
    start = seed.frame + 1
    duration = 2.2
    if (total - start - 1) / fps < 0.4:
        return []
    apex = min(height * 0.30, seed.y - height * 0.20)
    direction = 1 if seed.x <= width * 0.55 else -1
    drift = direction * width * 0.03
    landing_y = min(seed.y - height * 0.10, height * 0.55)
    points = []
    for frame in range(start, min(total, start + round(duration * fps) + 1)):
        u = (frame - start) / max(duration * fps, 1)
        x = seed.x + drift * u
        y = seed.y + (landing_y - seed.y) * u - 4 * apex * u * (1 - u)
        if 0 <= x < width and 0 <= y < height:
            points.append(Point(frame, x, y, 0, "predicted"))
    return points


def draw_tracer(frame: np.ndarray, points: list[Point], color: tuple[int, int, int],
                thickness: int, glow: bool = True, max_gap_frames: int = 6) -> None:
    """Draw a solid tracer with a soft glow only at its growing tip."""
    segments = [(first, second) for first, second in zip(points, points[1:])
                if second.frame - first.frame <= max_gap_frames]
    if not segments:
        return
    if glow:
        first, tip = segments[-1]
        direction = np.array([tip.x - first.x, tip.y - first.y], dtype=float)
        length = float(np.hypot(*direction))
        if length > 0:
            tail = min(length, thickness * 5)
            start = np.array([tip.x, tip.y]) - direction / length * tail
            pad = thickness * 5
            x0 = max(0, int(min(start[0], tip.x) - pad))
            y0 = max(0, int(min(start[1], tip.y) - pad))
            x1 = min(frame.shape[1], int(max(start[0], tip.x) + pad + 1))
            y1 = min(frame.shape[0], int(max(start[1], tip.y) + pad + 1))
            if x1 > x0 and y1 > y0:
                overlay = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
                cv2.line(overlay, (round(start[0] - x0), round(start[1] - y0)),
                         (round(tip.x - x0), round(tip.y - y0)), color,
                         thickness * 3, cv2.LINE_AA)
                cv2.circle(overlay, (round(tip.x - x0), round(tip.y - y0)),
                           thickness * 2, color, -1, cv2.LINE_AA)
                blurred = cv2.GaussianBlur(overlay, (0, 0), thickness * 1.4)
                region = frame[y0:y1, x0:x1]
                cv2.addWeighted(region, 1, blurred, 0.6, 0, dst=region)
    for first, second in segments:
        cv2.line(frame, (round(first.x), round(first.y)),
                 (round(second.x), round(second.y)), color, thickness, cv2.LINE_AA)


def render_manual_video(source: Path, destination: Path, points: list[dict], progress=None,
                        timings: dict[str, float] | None = None) -> dict:
    """Render a user-adjusted path on the original video, without rerunning detection."""
    timings = timings if timings is not None else {}
    stage_started = time.perf_counter()
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timings["probe_s"] = time.perf_counter() - stage_started
    if not cap.isOpened() or not (1 <= fps <= 240) or width < 160 or height < 120 or total < 2:
        cap.release()
        raise ValueError("Could not read the uploaded video")
    if points[-1]["t"] > total / fps + 0.1:
        cap.release()
        raise ValueError("Manual path extends beyond the video")
    path = [Point(round(item["t"] * fps), item["x"] * width,
                  item["y"] * height, 1, "predicted") for item in points]
    temp = destination.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the rendered video")
    line_thickness = max(5, round(min(width, height) / 720 * 6))
    render_started = time.perf_counter()
    try:
        frame_id = 0
        while True:
            stage_started = time.perf_counter()
            ok, frame = cap.read()
            timings["decode_render_s"] = (timings.get("decode_render_s", 0.0)
                                           + time.perf_counter() - stage_started)
            if not ok:
                break
            stage_started = time.perf_counter()
            visible = [point for point in path if point.frame <= frame_id]
            if visible and len(visible) < len(path):
                next_point = path[len(visible)]
                previous = visible[-1]
                if next_point.frame > previous.frame:
                    fraction = (frame_id - previous.frame) / (next_point.frame - previous.frame)
                    visible.append(Point(frame_id,
                                         previous.x + (next_point.x - previous.x) * fraction,
                                         previous.y + (next_point.y - previous.y) * fraction,
                                         1, "predicted"))
            draw_tracer(frame, visible, (48, 59, 255), line_thickness,
                        max_gap_frames=total)
            writer.write(frame)
            timings["draw_write_s"] = (timings.get("draw_write_s", 0.0)
                                        + time.perf_counter() - stage_started)
            frame_id += 1
            if progress and frame_id % 10 == 0:
                progress(min(90, int(frame_id / total * 90)))
    finally:
        cap.release()
        writer.release()
        timings["render_s"] = time.perf_counter() - render_started
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temp), "-i", str(source),
               "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
               "-shortest", str(destination)]
    stage_started = time.perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    finally:
        timings["ffmpeg_s"] = time.perf_counter() - stage_started
        temp.unlink(missing_ok=True)
    if progress:
        progress(100)
    return {"trace_found": True, "source": "manual", "detected_frames": 0,
            "motion_frames": 0, "predicted_frames": len(path), "total_frames": total,
            "fps": round(fps, 2), "width": width, "height": height,
            "duration": round(frame_id / fps, 3),
            "trace_points": [{"x": point["x"], "y": point["y"], "t": point["t"],
                              "kind": "predicted"} for point in points]}


def render_branding_video(source: Path, destination: Path, club_name: str | None,
                          include_watermark: bool, progress=None,
                          timings: dict[str, float] | None = None) -> dict:
    """Decorate an already traced MP4 without loading the ball detector."""
    timings = timings if timings is not None else {}
    stage_started = time.perf_counter()
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timings["probe_s"] = time.perf_counter() - stage_started
    if not cap.isOpened() or not (1 <= fps <= 240) or width < 160 or height < 120 or total < 2:
        cap.release()
        raise ValueError("Could not read the traced video")
    branding = VideoBranding(width, height, club_name, include_watermark)
    temp = destination.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the decorated video")
    render_started = time.perf_counter()
    try:
        frame_id = 0
        while True:
            stage_started = time.perf_counter()
            ok, frame = cap.read()
            timings["decode_render_s"] = (timings.get("decode_render_s", 0.0)
                                           + time.perf_counter() - stage_started)
            if not ok:
                break
            stage_started = time.perf_counter()
            branding.draw(frame)
            writer.write(frame)
            timings["draw_write_s"] = (timings.get("draw_write_s", 0.0)
                                        + time.perf_counter() - stage_started)
            frame_id += 1
            if progress and frame_id % 10 == 0:
                progress(min(90, int(frame_id / total * 90)))
    finally:
        cap.release()
        writer.release()
        timings["render_s"] = time.perf_counter() - render_started
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temp), "-i", str(source),
               "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
               "-shortest", str(destination)]
    stage_started = time.perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    finally:
        timings["ffmpeg_s"] = time.perf_counter() - stage_started
        temp.unlink(missing_ok=True)
    if progress:
        progress(100)
    return {"trace_found": True, "source": "decorate", "total_frames": frame_id,
            "fps": round(fps, 2), "width": width, "height": height,
            "duration": round(frame_id / fps, 3)}


def process_video(source: Path, destination: Path, progress=None,
                  timings: dict[str, float] | None = None) -> dict:
    timings = timings if timings is not None else {}
    stage_started = time.perf_counter()
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timings["probe_s"] = time.perf_counter() - stage_started
    if not cap.isOpened() or not (1 <= fps <= 240) or width < 160 or height < 120 or total < 2:
        cap.release()
        raise ValueError("Не удалось прочитать видео. Загрузите MP4 или MOV с обычной частотой кадров.")
    stage_started = time.perf_counter()
    detector = BallDetector(timings=timings)
    timings["model_init_s"] = time.perf_counter() - stage_started
    frames: list[list[Point]] = []
    try:
        while True:
            stage_started = time.perf_counter()
            ok, frame = cap.read()
            timings["decode_first_s"] = (timings.get("decode_first_s", 0.0)
                                          + time.perf_counter() - stage_started)
            if not ok:
                break
            stage_started = time.perf_counter()
            frames.append(detector.detect(frame, len(frames)))
            timings["detect_first_s"] = (timings.get("detect_first_s", 0.0)
                                          + time.perf_counter() - stage_started)
            if progress and len(frames) % 10 == 0:
                progress(min(85, int(len(frames) / max(total, 1) * 85)))
    finally:
        cap.release()
    if not frames:
        raise ValueError("Видео не содержит читаемых кадров.")
    stage_started = time.perf_counter()
    track = select_track(frames, width, height)
    seed = select_launch_seed(frames, width, height, fps) if not track else None
    timings["select_s"] = time.perf_counter() - stage_started
    if not track and (height > width * 1.25 or width >= 2560):
        stage_started = time.perf_counter()
        cropped_frames = detect_tee_crop(source, detector, width, height)
        cropped_track = select_track(cropped_frames, width, height)
        cropped_seed = (select_launch_seed(cropped_frames, width, height, fps)
                        if not cropped_track else None)
        if (cropped_seed and
                (seed is None or cropped_seed.confidence > seed.confidence + 0.12)):
            frames, track, seed = cropped_frames, [], cropped_seed
        elif cropped_track and seed is None:
            frames, track, seed = cropped_frames, cropped_track, None
        timings["crop_retry_s"] = time.perf_counter() - stage_started
    motion_used = False
    if seed is not None:
        stage_started = time.perf_counter()
        motion_track = track_motion_from_seed(source, seed, fps, width, height, len(frames))
        timings["motion_s"] = time.perf_counter() - stage_started
        if motion_track:
            track = motion_track
            motion_used = True
    launch_seed = seed if not track else None
    stage_started = time.perf_counter()
    predicted = build_prediction(track, launch_seed, len(frames), width, height, fps)
    timings["prediction_s"] = time.perf_counter() - stage_started
    by_frame = {point.frame: point for point in track}
    prediction_by_frame = {point.frame: point for point in predicted}
    temp = destination.with_suffix(".raw.mp4")
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Не удалось создать выходной видеофайл.")
    history: list[Point] = []
    forecast_history: list[Point] = [launch_seed] if launch_seed and predicted else []
    line_thickness = max(5, round(min(width, height) / 720 * 6))
    measured_color = (0, 122, 255)  # orange, BGR
    predicted_color = (48, 59, 255)  # red, BGR
    render_started = time.perf_counter()
    try:
        frame_id = 0
        while True:
            stage_started = time.perf_counter()
            ok, frame = cap.read()
            timings["decode_render_s"] = (timings.get("decode_render_s", 0.0)
                                           + time.perf_counter() - stage_started)
            if not ok:
                break
            stage_started = time.perf_counter()
            current = by_frame.get(frame_id)
            if current:
                history.append(current)
            forecast = prediction_by_frame.get(frame_id)
            if forecast:
                forecast_history.append(forecast)
            # Orange is observed motion; red is a visually estimated segment.
            draw_tracer(frame, history, measured_color, line_thickness,
                        glow=not forecast_history)
            forecast_points = forecast_history if launch_seed else ([track[-1]] + forecast_history if track and predicted else [])
            draw_tracer(frame, forecast_points, predicted_color, line_thickness)
            if forecast_history and frame_id >= forecast_history[0].frame:
                cv2.rectangle(frame, (16, 16), (233, 52), (25, 33, 48), -1)
                cv2.putText(frame, "PREDICTED FLIGHT", (28, 41),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            timings["draw_write_s"] = (timings.get("draw_write_s", 0.0)
                                        + time.perf_counter() - stage_started)
            frame_id += 1
    finally:
        cap.release()
        writer.release()
        timings["render_s"] = time.perf_counter() - render_started
    if progress:
        progress(92)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temp), "-i", str(source),
               "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
               "-shortest", str(destination)]
    stage_started = time.perf_counter()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    finally:
        timings["ffmpeg_s"] = time.perf_counter() - stage_started
        temp.unlink(missing_ok=True)
    if progress:
        progress(100)
    source_kind = ("motion+predicted" if motion_used and predicted else
                   "motion" if motion_used else
                   "detected+predicted" if track and predicted else
                   "predicted" if predicted else "detected" if track else "none")
    trace = ([launch_seed] if launch_seed and predicted else []) + track + predicted
    if len(trace) > 240:
        indices = np.linspace(0, len(trace) - 1, 240, dtype=int)
        trace = [trace[index] for index in indices]
    trace_points = [{"x": round(point.x / width, 6),
                     "y": round(point.y / height, 6),
                     "t": round(point.frame / fps, 4),
                     "kind": "seed" if point is launch_seed else point.source}
                    for point in trace]
    return {"detected_frames": len(track), "motion_frames": len(track) - 1 if motion_used else 0,
            "predicted_frames": len(predicted),
            "total_frames": len(frames), "trace_found": bool(track or predicted),
            "source": source_kind,
            "seed_frame": seed.frame if seed and predicted else None, "fps": round(fps, 2),
            "width": width, "height": height, "duration": round(len(frames) / fps, 3),
            "trace_points": trace_points}
