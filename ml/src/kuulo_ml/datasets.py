"""Licensed datasets for Step B, downloaded into data/datasets/ (gitignored). See ml/DATASETS.md.

- DroneNoise Database (University of Salford, CC BY 4.0): outdoor overflights of four drones,
  each flight recorded by up to nine microphones at once. Drone positives.
- ESC-50 (Piczak, CC BY-NC 3.0): 2000 five-second environmental clips in 50 classes and five
  folds built from separate source recordings. Negatives, including hard negatives.

Splits are by source, never by window or file: every microphone of one flight lands in the same
split (they hear the same event), and ESC-50's folds are used as they ship.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

DRONENOISE_ARTICLE = "https://api.figshare.com/v2/articles/22133411"
ESC50_ZIP = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_DIR = "ESC-50-master"

# Drone-like sounds reported separately in evaluation (spec §10 "hard negatives").
HARD_NEGATIVES = frozenset({
    "chainsaw", "engine", "airplane", "helicopter", "hand_saw", "vacuum_cleaner", "insects",
})
_MIC = re.compile(r"_M\d+$")


@dataclass(frozen=True)
class Clip:
    path: Path
    label: int  # 1 = drone
    group: str  # recordings that must share a split
    kind: str  # "drone:<type>" or the ESC-50 category
    split: str  # "train" | "val" | "test"


def flight_of(stem: str) -> str:
    return _MIC.sub("", stem)


def _download(client: httpx.Client, url: str, target: Path, size: int | None = None) -> None:
    if target.exists() and (size is None or target.stat().st_size == size):
        return
    part = target.with_suffix(target.suffix + ".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
    if size is not None and part.stat().st_size != size:
        raise OSError(f"{target.name}: expected {size} bytes, got {part.stat().st_size}")
    part.replace(target)


def download_dronenoise(root: Path, log=print) -> None:
    folder = root / "dronenoise"
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        files = client.get(DRONENOISE_ARTICLE).raise_for_status().json()["files"]
        wavs = [f for f in files
                if f["name"].endswith(".wav") and not f["name"].startswith("Calib")]
        for i, f in enumerate(wavs, 1):
            target = folder / f["name"]
            if not (target.exists() and target.stat().st_size == f["size"]):
                log(f"[{i}/{len(wavs)}] {f['name']}")
            _download(client, f["download_url"], target, f["size"])


def download_esc50(root: Path, log=print) -> None:
    folder = root / "esc50"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "master.zip"
    if not (folder / ESC50_DIR / "meta" / "esc50.csv").exists():
        log("ESC-50 (~600 MB)")
        with httpx.Client(follow_redirects=True, timeout=600) as client:
            _download(client, ESC50_ZIP, archive)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist()
                       if m.startswith((f"{ESC50_DIR}/audio/", f"{ESC50_DIR}/meta/"))
                       or m == f"{ESC50_DIR}/LICENSE"]
            zf.extractall(folder, members)


def dronenoise_manifest(root: Path) -> list[Clip]:
    paths = sorted(p for p in (root / "dronenoise").glob("*.wav") if not p.name.startswith("Calib"))
    flights_by_drone: dict[str, list[str]] = {}
    for p in paths:
        drone, flight = p.stem.split("_")[1], flight_of(p.stem)
        flights_by_drone.setdefault(drone, [])
        if flight not in flights_by_drone[drone]:
            flights_by_drone[drone].append(flight)
    split_of: dict[str, str] = {}
    for flights in flights_by_drone.values():
        for i, flight in enumerate(sorted(flights)):
            split_of[flight] = "test" if i % 4 == 0 else "val" if i % 4 == 1 else "train"
    return [
        Clip(p, 1, flight_of(p.stem), f"drone:{p.stem.split('_')[1]}", split_of[flight_of(p.stem)])
        for p in paths
    ]


def esc50_manifest(root: Path) -> list[Clip]:
    base = root / "esc50" / ESC50_DIR
    with open(base / "meta" / "esc50.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    split = {"5": "test", "4": "val"}
    return [
        Clip(base / "audio" / r["filename"], 0, f"esc50:{r['src_file']}", r["category"],
             split.get(r["fold"], "train"))
        for r in rows
    ]


def manifest(root: Path) -> list[Clip]:
    return dronenoise_manifest(root) + esc50_manifest(root)
