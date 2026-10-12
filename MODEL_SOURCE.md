# Model provenance and license notices

`models/best.onnx` is a copy of [`notjulietxd/golf-ball-tracker/best.onnx`](https://huggingface.co/notjulietxd/golf-ball-tracker/blob/main/best.onnx).

SHA-256: `9e948135c9898c2b0e4194fa992308a0bbb478e21f6407dda12cb49e3e91ac22`

The [model card](https://huggingface.co/notjulietxd/golf-ball-tracker) labels the project Apache-2.0. The ONNX file's embedded metadata says `AGPL-3.0 License (https://ultralytics.com/license)`. These notices conflict. The MIT license in this repository covers only the API's own source code; it does not relicense the model. Confirm the applicable terms with the model owner before redistribution or commercial use.

The API expects a compatible YOLOv8 ONNX detector at `models/best.onnx`. You can replace this file with a model you have rights to use, provided its input and output format matches `BallDetector` in `tracker.py`.
