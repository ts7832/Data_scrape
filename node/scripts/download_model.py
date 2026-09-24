"""Download the YAMNet TFLite model and its class map into data/models/ (gitignored)."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    # MediaPipe's float32 YAMNet audio classifier (Apache-2.0).
    "yamnet.tflite": "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite",  # noqa: E501
    # AudioSet class names in model output order (Apache-2.0, tensorflow/models).
    "yamnet_class_map.csv": "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",  # noqa: E501
}


def main() -> int:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = out_dir / name
        if target.exists():
            print(f"exists  {target}")
            continue
        print(f"fetch   {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()[:16]
        print(f"wrote   {target}  {len(data):,} bytes  sha256:{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
