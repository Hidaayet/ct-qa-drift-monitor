"""
04_generate_dataset.py
Generate a labeled, reproducible dataset of simulated phantom QA scans.

Design decisions (write these into docs/):
  * Each simulated SCANNER has its own dose level, calibration offset and
    baseline resolution. Real scanners differ, and drift must be judged
    against each scanner's own baseline.
  * The first N_REF scans of every scanner are fault-free REFERENCE scans
    (the commissioning baseline).
  * Train/val/test are split BY SCANNER, never by scan. Splitting by scan
    would put scans from the same scanner in both train and test, and the
    model would just memorise scanner fingerprints (data leakage).
  * Every scan has its own seed, so results are reproducible and
    independent of how many CPU cores are used.
  * A manifest records the configuration, library versions and a SHA-256
    of the CSV (traceability).

Usage:  python 04_generate_dataset.py [output_dir]      (default: ../data)
"""
import csv
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import skimage

import ctsim
import ctmetrics

MASTER_SEED = 2026
N_SCANNERS = 10
N_REF = 5                 # fault-free reference scans per scanner
N_RANDOM = 30             # random scans per scanner (some faulty, some not)
SPLIT = {**{s: "train" for s in range(0, 6)},
         **{s: "val" for s in (6, 7)},
         **{s: "test" for s in (8, 9)}}

FAULT_PROBS = {"none": 0.25, "ring": 0.15, "hu_drift": 0.15,
               "blur": 0.15, "cupping": 0.15, "tube_decl": 0.15}

# fault: (low, high, log-uniform?, unit)
FAULT_RANGES = {
    "ring":      (0.002, 0.06, True,  "gain error (fraction)"),
    "hu_drift":  (1.0,   15.0, False, "HU shift (magnitude)"),
    "blur":      (0.2,   2.0,  False, "extra blur sigma (channels)"),
    "cupping":   (0.002, 0.06, True,  "beam-hardening k"),
    "tube_decl": (0.15,  0.90, False, "remaining tube output (fraction)"),
}

# ---- per-process simulator state (built once per worker) --------------
_SINO = None


def _sino():
    global _SINO
    if _SINO is None:
        _SINO = ctsim.forward(ctsim.make_phantom(with_inserts=False))
    return _SINO


def scanner_properties(scanner_id):
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 1000 + scanner_id]))
    return {
        "I0": float(np.clip(1e5 * np.exp(rng.normal(0, 0.25)), 5e4, 2e5)),
        "hu_offset": float(rng.normal(0, 1.0)),
        "blur_base": float(rng.uniform(0.05, 0.40)),
    }


def draw_fault(rng, is_reference):
    if is_reference:
        return "none", 0.0, 0.0
    names = list(FAULT_PROBS)
    fault = names[rng.choice(len(names), p=list(FAULT_PROBS.values()))]
    if fault == "none":
        return "none", 0.0, 0.0
    lo, hi, log, _ = FAULT_RANGES[fault]
    u = rng.uniform()
    param = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if log
                  else lo + u * (hi - lo))
    if fault == "tube_decl":
        severity = (hi - param) / (hi - lo)       # lower output = more severe
    elif log:
        severity = (np.log(param) - np.log(lo)) / (np.log(hi) - np.log(lo))
    else:
        severity = (param - lo) / (hi - lo)
    return fault, param, float(severity)


def run_task(task):
    scan_id, scanner_id, is_reference = task
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, scan_id]))
    props = scanner_properties(scanner_id)
    sino = _sino()
    n_det = sino.shape[0]
    fault, param, severity = draw_fault(rng, is_reference)

    I0 = props["I0"]
    kw = dict(hu_shift=props["hu_offset"], blur_sigma=props["blur_base"])
    detail = ""
    if fault == "ring":
        offset = int(rng.integers(10, 90)) * int(rng.choice([-1, 1]))
        g = np.ones(n_det)
        g[n_det // 2 + offset] *= 1 + param
        kw["channel_gain"] = g
        detail = f"channel_offset={offset}"
    elif fault == "hu_drift":
        kw["hu_shift"] += param * int(rng.choice([-1, 1]))
    elif fault == "blur":
        kw["blur_sigma"] = props["blur_base"] + param
    elif fault == "cupping":
        kw["bh_k"] = param
    elif fault == "tube_decl":
        I0 = props["I0"] * param

    m = ctmetrics.measure(ctsim.simulate_scan(sino, I0, rng, **kw))
    return {
        "scan_id": scan_id, "scanner_id": scanner_id,
        "split": SPLIT[scanner_id], "is_reference": int(is_reference),
        "fault": fault, "param": round(param, 6),
        "severity_norm": round(severity, 4), "detail": detail,
        "I0": round(I0, 1), **{k: round(v, 5) for k, v in m.items()},
    }


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, sid = [], 0
    for s in range(N_SCANNERS):
        for k in range(N_REF + N_RANDOM):
            tasks.append((sid, s, k < N_REF))
            sid += 1

    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run_task, tasks, chunksize=5))
    rows.sort(key=lambda r: r["scan_id"])
    print(f"Generated {len(rows)} scans in {time.time() - t0:.0f} s")

    csv_path = out_dir / "qa_dataset.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    manifest = {
        "master_seed": MASTER_SEED, "n_scanners": N_SCANNERS,
        "n_reference_per_scanner": N_REF, "n_random_per_scanner": N_RANDOM,
        "split_by_scanner": {str(k): v for k, v in SPLIT.items()},
        "fault_probabilities": FAULT_PROBS,
        "fault_ranges": {k: {"low": v[0], "high": v[1], "log_uniform": v[2],
                             "unit": v[3]} for k, v in FAULT_RANGES.items()},
        "scanner_properties": {str(s): scanner_properties(s)
                               for s in range(N_SCANNERS)},
        "metrics": ctmetrics.METRIC_NAMES,
        "simulator_limitations": "parallel-beam, monoenergetic, no scatter, "
                                 "no electronic noise, empirical beam hardening",
        "software": {"python": platform.python_version(),
                     "numpy": np.__version__, "scikit-image": skimage.__version__},
        "csv_file": csv_path.name, "csv_sha256": sha,
        "generated_unix_time": int(time.time()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- summary
    print(f"Wrote {csv_path}\n      sha256 {sha[:16]}...")
    print("\nScans per split and fault (reference scans excluded):")
    faults = list(FAULT_PROBS)
    print(f"{'split':8s}" + "".join(f"{f:>11s}" for f in faults))
    for sp in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == sp and not r["is_reference"]]
        print(f"{sp:8s}" + "".join(
            f"{sum(r['fault'] == f for r in sub):>11d}" for f in faults))


if __name__ == "__main__":
    main()