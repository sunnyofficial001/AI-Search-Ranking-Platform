#!/usr/bin/env python3
"""Generate a tiny synthetic MSLR-like dataset for quick demos.

This writes three files under `data/Fold1/`: train.txt, vali.txt, test.txt
in a simplified SVMLight-like format.
"""

import os
import random

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Fold1")


def ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def write_split(filename, qid_start, n_queries=5, docs_per_query=4):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for qi in range(qid_start, qid_start + n_queries):
            for di in range(1, docs_per_query + 1):
                label = random.choices([0, 1, 2, 3, 4], weights=[40, 25, 15, 12, 8])[0]
                features = []
                for feat_idx in range(1, 6):
                    val = round(random.random(), 4)
                    features.append(f"{feat_idx}:{val}")
                # simple SVMLight-like: <label> qid:<qid> 1:<v> 2:<v> # docid=...
                line = f"{label} qid:{qi} " + " ".join(features) + f" # docid=Q{qi}D{di}\n"
                f.write(line)


def generate_all():
    ensure_dir()
    write_split("train.txt", qid_start=1, n_queries=6, docs_per_query=5)
    write_split("vali.txt", qid_start=1001, n_queries=3, docs_per_query=4)
    write_split("test.txt", qid_start=2001, n_queries=3, docs_per_query=4)
    print(f"Demo dataset generated in: {BASE_DIR}")


if __name__ == "__main__":
    generate_all()
