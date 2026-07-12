from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
ip_op = [

# ([
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_28\results_bokehlicious.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_36\results_bokehlicious.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\results_bokehlicious.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_60\results_bokehlicious.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\results_bokehlicious.csv",
#   ], r"I:\My Drive\DOF_benchmarking\Scene8\results_bokehlicious_all.csv"), 


# ([
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_28\results_bokehme_28mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_36\results_bokehme_36mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\results_bokehme_45mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_60\results_bokehme_60mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\results_bokehme_70mm.csv",
#   ], r"I:\My Drive\DOF_benchmarking\Scene8\results_bokehme_all.csv"), 


# ([
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_28\results_bokehdiff_28mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_36\results_bokehdiff_36mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\results_bokehdiff_45mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_60\results_bokehdiff_60mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\results_bokehdiff_70mm.csv",
#   ], r"I:\My Drive\DOF_benchmarking\Scene8\results_bokehdiff_all.csv"), 

# ([
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_28\results_drbokeh_28mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_36\results_drbokeh_36mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_45\results_drbokeh_45mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_60\results_drbokeh_60mm.csv",
#     r"I:\My Drive\DOF_benchmarking\Scene8\fl_70\results_drbokeh_70mm.csv",
#   ], r"I:\My Drive\DOF_benchmarking\Scene8\results_drbokeh_all.csv"), 

# ([r"I:\My Drive\DOF_benchmarking\inference\fl_28\drbokeh_all.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_drbokeh_fl36.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_drbokeh_fl45.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_drbokeh_fl60.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_drbokeh_fl70.csv"
# ], r"I:\My Drive\DOF_benchmarking\inference\results_drbokeh_scene1.csv")

# ([r"I:\My Drive\DOF_benchmarking\inference\fl_28\results_bokehdiff.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_bokehdiff.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_bokehdiff.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_bokehdiff.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_bokehdiff.csv"
# ], r"I:\My Drive\DOF_benchmarking\inference\results_bokehdiff_scene1.csv")

# ([r"I:\My Drive\DOF_benchmarking\inference\fl_28\results_bokehme.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_bokehme.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_bokehme.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_bokehme.csv",
# r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_bokehme.csv"
# ], r"I:\My Drive\DOF_benchmarking\inference\results_bokehme_scene1.csv")


([r"I:\My Drive\DOF_benchmarking\inference\fl_28\results_bokehlicious_p8.csv",
r"I:\My Drive\DOF_benchmarking\inference\fl_36\results_bokehlicious_p8.csv",
r"I:\My Drive\DOF_benchmarking\inference\fl_45\results_bokehlicious_p8.csv",
r"I:\My Drive\DOF_benchmarking\inference\fl_60\results_bokehlicious_p8.csv",
r"I:\My Drive\DOF_benchmarking\inference\fl_70\results_bokehlicious_p8.csv"
], r"I:\My Drive\DOF_benchmarking\inference\results_bokehlicious_scene1.csv")
]
SEPARATOR = "\n"
# ──────────────────────────────────────────────────────────────

for file_paths, output_path in ip_op:
    print(f"\n── {output_path}")
    chunks = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            print(f"  [WARN] File not found, skipping: {fp}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        chunks.append(text)
        print(f"  [OK] Read {len(text)} chars from {fp}")

    combined = SEPARATOR.join(chunks)
    Path(output_path).write_text(combined, encoding="utf-8")
    print(f"  Wrote {len(combined)} chars to {output_path}")
