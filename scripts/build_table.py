#!/usr/bin/env python3
"""Turn the raw per-run JSON into the one table the article publishes.

Every row is measured on one rented A40, so every number in the article is
reproducible by anyone for about fifty cents. Nothing here depends on the
author's laptop, which is the point: a latency that only one machine can
produce is an anecdote.

Two corrections happen here rather than in the benchmark, because both were
only visible once every row existed.

The first is a label. The runner never offloads the NF4 backbone -- bitsandbytes
places it with a device map -- but the CLI flag defaulted to on and the record
copied the flag rather than what happened. The flag is not the fact.

The second is the fidelity metric. AbsRel is the standard in depth papers and
it is the wrong choice here: it divides by the target, and Marigold's
affine-invariant output is centred near zero, so the 0.46% of pixels closest to
zero dominate a number that then reads 0.14 and looks like catastrophe. RMSE as
a fraction of the output range, and Pearson correlation, say what is actually
true -- that every quantized backbone tracks the bf16 reference to within about
two percent. Both are reported so the discrepancy is visible rather than tidied
away.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from marigoldedge.core import metrics

A40 = Path("output/a40")
A40_2 = Path("output/a40_session2")
REFERENCE = "15_kitten_bf16_768.npy"
REFERENCE_2 = "15_kitten_bf16_768_resident.npy"

# label -> (bench json, depth npy, host, what actually happened to placement)
ROWS = [
    ("bf16", "bench_15_kitten_bf16_768.json", REFERENCE, "A40 48 GB", "resident"),
    ("bnb-NF4", "bench_15_kitten_bnb-NF4_768.json", "15_kitten_bnb-NF4_768.npy",
     "A40 48 GB", "resident"),
    ("GGUF Q4_K_M", "bench_15_kitten_Q4_K_M_768_resident.json",
     "15_kitten_Q4_K_M_768_resident.npy", "A40 48 GB", "resident"),
    ("GGUF Q4_K_M", "bench_15_kitten_Q4_K_M_768_offloaded.json",
     "15_kitten_Q4_K_M_768.npy", "A40 48 GB", "offloaded"),
]

# Second session, same GPU model, different host and driver.
ROWS_2 = [
    ("torchao-INT8", "bench_15_kitten_torchao-INT8_768.json",
     "15_kitten_torchao-INT8_768.npy"),
    ("torchao-INT4", "bench_15_kitten_torchao-INT4_768.json",
     "15_kitten_torchao-INT4_768.npy"),
    ("bnb-INT8", "bench_15_kitten_bnb-INT8_768.json", "15_kitten_bnb-INT8_768.npy"),
]


# The second session's rows are scored against the second session's own bf16
# run, not the first's. Both were measured on an A40 and agree to 1.3% (0.79 s
# against 0.80 s) on different hosts with different drivers, which is a
# reproducibility result in its own right -- but a fidelity score compares
# arrays, and those must come from the same session to mean anything.
SESSIONS = {"1": (A40, REFERENCE), "2": (A40_2, REFERENCE_2)}


def fidelity(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    aligned = metrics.align_affine(pred, ref)
    diff = aligned - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    span = float(ref.max() - ref.min())
    absdiff = np.abs(diff)
    # The distribution is heavily tailed, so a single average hides the shape.
    # The percentiles are what show that quantization agrees almost everywhere
    # and disagrees at depth edges.
    return {
        "p50_abs_diff_over_range": float(np.percentile(absdiff, 50)) / span,
        "p99_abs_diff_over_range": float(np.percentile(absdiff, 99)) / span,
        "max_abs_diff_over_range": float(absdiff.max()) / span,
        "fraction_over_5pct_of_range": float(np.mean(absdiff > 0.05 * span)),
        "abs_rel_vs_ref": metrics.abs_rel(aligned, ref),
        "rmse_vs_ref": rmse,
        "rmse_over_range": rmse / span,
        "pearson_r_vs_ref": float(np.corrcoef(aligned.ravel(), ref.ravel())[0, 1]),
    }


def main() -> None:
    ref = np.load(A40 / REFERENCE)
    table = []

    for label, bench, npy, host, placement in ROWS:
        record = json.loads((A40 / bench).read_text())
        row = {
            "backend": label,
            "host": host,
            "placement": placement,  # the fact, not the flag
            "seconds_median": record["seconds_median"],
            "peak_vram_gb": record["peak_vram_gb"],
        }
        if npy != REFERENCE:
            row.update(fidelity(np.load(A40 / npy), ref))
        table.append(row)

    ref2 = np.load(A40_2 / REFERENCE_2)
    for label, bench, npy in ROWS_2:
        path = A40_2 / bench
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        row = {
            "backend": label, "host": "A40 48 GB", "placement": "resident", "session": 2,
            "seconds_median": record["seconds_median"],
            "peak_vram_gb": record["peak_vram_gb"],
        }
        row.update(fidelity(np.load(A40_2 / npy), ref2))
        table.append(row)

    resident = {r["backend"]: r for r in table if r["placement"] == "resident"}
    off = next(r for r in table if r["placement"] == "offloaded")
    derived = {
        "nf4_vs_bf16_time": resident["bnb-NF4"]["seconds_median"]
        / resident["bf16"]["seconds_median"],
        "nf4_vs_bf16_memory": resident["bnb-NF4"]["peak_vram_gb"]
        / resident["bf16"]["peak_vram_gb"],
        "gguf_vs_nf4_time": resident["GGUF Q4_K_M"]["seconds_median"]
        / resident["bnb-NF4"]["seconds_median"],
        "offload_penalty_same_card": off["seconds_median"]
        / resident["GGUF Q4_K_M"]["seconds_median"],
    }

    out = {"table": table, "derived_ratios": derived,
           "reference": "bf16 on the A40", "image": "15_kitten.jpg", "resolution": 768}
    Path("output/a40_table.json").write_text(json.dumps(out, indent=2))

    print(f"{'backend':<13} {'host':<22} {'placement':<11} {'s':>6} {'VRAM GB':>8} "
          f"{'rmse/rng':>9} {'r':>8}")
    for r in table:
        rr = f"{r['rmse_over_range']:.2%}" if "rmse_over_range" in r else "ref"
        pr = f"{r['pearson_r_vs_ref']:.4f}" if "pearson_r_vs_ref" in r else "-"
        print(f"{r['backend']:<13} {r['host']:<22} {r['placement'][:11]:<11} "
              f"{r['seconds_median']:>6.2f} {r['peak_vram_gb']:>8.2f} {rr:>9} {pr:>8}")
    print()
    print("tail shape (fraction of range):")
    for r in table:
        if "p50_abs_diff_over_range" in r:
            print(f"  {r['backend']:<13} p50={r['p50_abs_diff_over_range']:.2%} "
                  f"p99={r['p99_abs_diff_over_range']:.2%} "
                  f"max={r['max_abs_diff_over_range']:.1%} "
                  f">5%: {r['fraction_over_5pct_of_range']:.2%}")
    print()
    for k, v in derived.items():
        print(f"  {k:<36} {v:.2f}x")


if __name__ == "__main__":
    main()
