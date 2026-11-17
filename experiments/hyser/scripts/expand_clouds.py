#!/usr/bin/env python3
"""Decode quantized real 250×3 Hyser points; generated from actual normalized data.
Each sample: 2 base36 digits per coordinate, 1 KMeans label, 1 GMM label.
Coordinates are rounded to ~0.0067 units solely for visual plots, NOT fitting.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "figures" / "clouds_base36.txt"
for line in source.read_text().splitlines():
    if not line.strip():
        continue
    name, packed = line.split(":", 1)
    assert name in ("EMG", "FORCE") and len(packed) == 8 * 250
    pts = []
    for i in range(0, len(packed), 8):
        entry = packed[i:i + 8]
        coords = [int(entry[j:j + 2], 36) / 150.0 - 4 for j in (0, 2, 4)]
        pts.append([*coords, int(entry[6]), int(entry[7])])
    for method, col in (("kmeans", 3), ("gmm", 4)):
        dest = ROOT / "figures" / f"hiwa_covariance_whitening_{name.lower()}_{method}.csv"
        with dest.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", "y", "z", "cluster"])
            writer.writerows([[f"{v:.6f}" for v in p[:3]] + [p[col]] for p in pts])
        print("generated", dest)
