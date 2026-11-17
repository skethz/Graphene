"""Job lists and value selection for the paper's cost tables.

Tables and settings (all x-scale entries use the protocol in benchmarks/README.md):

  bf16-64     Figure 1, Tables 3, 13, 14   BF16 autocast + torch.compile, TF32 off, LR 64x64, x2/x3/x4,
                                           50 warm-up + 200 timed passes, 7 fresh processes
  bf16-large  Table 4                      same precision and compilation, x4, LR 180x320 and 720x1280 (H x W),
                                           5 warm-up + 20 timed passes (the first timed pass is dropped), 3 processes
  fp32-64     Table 15, Figure 8           FP32 eager, TF32 off, LR 64x64, x2/x3/x4, 50 + 200 passes, 7 processes
  fp32-large  Table 16                     FP32 eager, TF32 off, x4, LR 180x320 and 720x1280, 5 + 20 passes, 3 processes

BF16 compiled variants. Released code runs unchanged with torch.compile(mode="max-autotune"). Only where the released
code cannot run this setting, the single failing part is changed (methods.apply_fallback), and every applicable
variant is measured; the fastest is reported (the variant most favourable to the baseline):
  CAT-A       cat_reshape, cat_eager
  IPG         ipg_cache (max-autotune); released code in max-autotune-no-cudagraphs
  PFT         smm_fp32;            PFT-light  none, smm_fp32
  IET         smm_fp16, smm_fp32 x NATTEN fp32, gemm_tf32_off
  IET-light   none, smm_fp16, smm_fp32 x NATTEN fp32, gemm_tf32_off

Procedure (bf16 tables):
  1. `jobs --stage probe`   one run (r1) of every variant.
  2. `jobs --stage full`    if no variant of a method completes at a size, its variants are probed again in
                            max-autotune-no-cudagraphs and default; then runs r2..rN for every variant whose r1
                            latency is within 1.25x of the method's fastest r1 (fresh processes differ by about 5%).
  3. `jobs --stage memory`  the reported peak memory of a pick compiled with CUDA graphs (max-autotune) is the median of
                            three max-autotune-no-cudagraphs runs, because CUDA-graph memory pools are not visible
                            to the allocator statistics. Other picks report the memory of their timing runs.
  4. `select`               median over the runs of each variant; the reported value is the variant with the lowest
                            median latency among those with all N runs.
FP32 tables have one variant per method (released code, NATTEN FP32 kernels for IET / IET-light): run `jobs --stage
full` and `select`.

Run every job list with run_queue.py (one fresh process per job on an idle node). Out-of-memory runs are results and
are reported as OOM.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

CL_PLAIN = ["SwinIR", "ART", "HAT", "SRFormer", "ATD", "MambaIR", "MambaIRv2", "Graphene"]
LT_PLAIN = ["SwinIR-light", "SRFormer-light", "ATD-light", "MambaIR-light", "MambaIRv2-light", "UCAN", "Graphene-light"]
CLASSICAL = ["SwinIR", "CAT-A", "ART", "HAT", "SRFormer", "IPG", "ATD", "MambaIR", "MambaIRv2", "PFT", "IET", "Graphene"]
LIGHT = ["SwinIR-light", "SRFormer-light", "ATD-light", "MambaIR-light", "MambaIRv2-light", "PFT-light", "UCAN",
         "IET-light", "Graphene-light"]
MA, NOCG = "max-autotune", "max-autotune-no-cudagraphs"
CUDAGRAPH_MODES = ("max-autotune", "reduce-overhead")

BF16_VARIANTS = (
    [((m, "none", None), MA) for m in CL_PLAIN]
    + [(("CAT-A", "cat_reshape", None), MA), (("CAT-A", "cat_eager", None), MA),
       (("IPG", "ipg_cache", None), MA), (("IPG", "none", None), NOCG), (("PFT", "smm_fp32", None), MA)]
    + [(("IET", fb, nat), MA) for nat in ("fp32", "gemm_tf32_off") for fb in ("smm_fp16", "smm_fp32")]
    + [((m, "none", None), MA) for m in LT_PLAIN]
    + [(("PFT-light", "none", None), MA), (("PFT-light", "smm_fp32", None), MA)]
    + [(("IET-light", fb, nat), MA) for nat in ("fp32", "gemm_tf32_off") for fb in ("none", "smm_fp16", "smm_fp32")]
)

TABLES = {
    "bf16-64": dict(execution="bf16_compiled", scales=(2, 3, 4), sizes=((64, 64),), warmup=50, iterations=200, repeats=7),
    "bf16-large": dict(execution="bf16_compiled", scales=(4,), sizes=((180, 320), (720, 1280)), warmup=5, iterations=20,
                       repeats=3),
    "fp32-64": dict(execution="fp32_eager", scales=(2, 3, 4), sizes=((64, 64),), warmup=50, iterations=200, repeats=7),
    "fp32-large": dict(execution="fp32_eager", scales=(4,), sizes=((180, 320), (720, 1280)), warmup=5, iterations=20,
                       repeats=3),
}
FINAL = ("complete", "failed", "oom", "timeout")
FALLBACKS = ("smm_fp32", "smm_fp16", "cat_reshape", "cat_eager", "ipg_const", "ipg_eager", "ipg_cache")
RUN_FILE = re.compile(r"_r\d+\.json$")


def variants(table: str):
    if TABLES[table]["execution"] == "bf16_compiled":
        return list(BF16_VARIANTS)
    return [((m, "none", "fp32" if m.startswith("IET") else None), None) for m in CLASSICAL + LIGHT]


def job(table: str, v, mode, scale: int, hw, r: int, out: Path, kind: str = "runs") -> dict:
    t = TABLES[table]
    (m, fb, nat), (h, w) = v, hw
    j = dict(method=m, scale=scale, execution=t["execution"], height=h, width=w, warmup=t["warmup"],
             iterations=t["iterations"], timeout=7200 if h >= 720 else 3600, tf32="off")
    tag = f"{m}_x{scale}_{t['execution']}"
    if t["execution"] == "bf16_compiled":
        j.update(compile_mode=mode, fallback=fb)
        tag += f"_{mode}_{fb}"
    if nat:
        j["natten_mode"] = nat
        tag += f"_natten-{nat}"
    j["output"] = str(out / kind / f"{tag}_{h}x{w}_r{r}.json")
    return j


def load(p) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def ok(j) -> bool:
    x = load(j["output"])
    return x.get("status") == "complete" and x.get("output_finite", True) is not False


def grid(table):
    t = TABLES[table]
    return [(v, mode, s, hw) for s in t["scales"] for hw in t["sizes"] for v, mode in variants(table)]


def with_mode_fallback(table, out):
    """Primary grid plus, for a method with no completing primary variant at a (scale, size), its variants in
    max-autotune-no-cudagraphs and default (bf16 tables only)."""
    g = grid(table)
    if TABLES[table]["execution"] != "bf16_compiled":
        return g, g
    extra = []
    methods = list(dict.fromkeys(v[0] for v, _ in variants(table)))
    for s in TABLES[table]["scales"]:
        for hw in TABLES[table]["sizes"]:
            for m in methods:
                mine = [x for x in g if x[0][0] == m and x[2] == s and x[3] == hw]
                if not any(ok(job(table, v, mode, s, hw, 1, out)) for v, mode, _, _ in mine):
                    extra += [(v, m2, s, hw) for v, mode, _, _ in mine for m2 in (NOCG, "default") if m2 != mode]
    return g, g + list(dict.fromkeys(extra))


def jobs_command(a) -> int:
    t, out = TABLES[a.table], Path(a.results)
    if a.stage == "probe":
        jobs = [job(a.table, v, mode, s, hw, 1, out) for v, mode, s, hw in grid(a.table)]
    elif a.stage == "full":
        primary, allg = with_mode_fallback(a.table, out)
        if TABLES[a.table]["execution"] != "bf16_compiled":
            jobs = [job(a.table, v, mode, s, hw, r, out) for r in range(1, t["repeats"] + 1) for v, mode, s, hw in allg]
        else:
            new = [x for x in allg if x not in primary and load(job(a.table, *x[:2], x[2], x[3], 1, out)["output"]).get("status") not in FINAL]
            if new:  # mode-fallback probes first; rerun `jobs --stage full` afterwards
                jobs = [job(a.table, v, mode, s, hw, 1, out) for v, mode, s, hw in new]
            else:
                good = [x for x in allg if ok(job(a.table, x[0], x[1], x[2], x[3], 1, out))]
                r1 = {x: load(job(a.table, x[0], x[1], x[2], x[3], 1, out)["output"])["latency_ms"] for x in good}
                full = []
                for m in dict.fromkeys(x[0][0] for x in good):
                    for s in t["scales"]:
                        for hw in t["sizes"]:
                            gs = [x for x in good if x[0][0] == m and x[2] == s and x[3] == hw]
                            gs = [x for x in gs if x in primary] or gs
                            if gs:
                                best = min(r1[x] for x in gs)
                                full += [x for x in gs if r1[x] <= 1.25 * best]
                jobs = [job(a.table, v, mode, s, hw, r, out) for r in range(2, t["repeats"] + 1) for v, mode, s, hw in full]
    else:  # memory
        picks = select(a.table, out)
        jobs = [job(a.table, (p["method"], p["fallback"], p.get("natten_mode")), NOCG, p["scale"], (p["height"], p["width"]),
                    r, out, kind="memory")
                for p in picks.values() if p.get("compile_mode") in CUDAGRAPH_MODES for r in (1, 2, 3)]
    jobs = [j for j in jobs if load(j["output"]).get("status") not in FINAL] if a.skip_done else jobs
    text = json.dumps(jobs, indent=1) + "\n"
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(text)
    print(f"{len(jobs)} jobs -> {a.output}")
    return 0


def vkey(x: dict, path: Path) -> tuple:
    fb = x.get("fallback")
    fb = fb.get("name") if isinstance(fb, dict) else fb
    if not fb:  # records that failed before the fallback was applied: take it from the file name
        fb = next((n for n in FALLBACKS if f"_{n}_" in path.name), "none")
    return (x.get("method"), x.get("scale"), tuple(x.get("input_shape") or ()), x.get("execution"), fb,
            x.get("natten_mode") if str(x.get("method", "")).startswith("IET") else None, x.get("compile_mode"),
            x.get("tf32"))


def status(x: dict) -> str:
    """Final status; a compilation or run that fails with CUDA out-of-memory counts as OOM."""
    if x.get("status") == "failed" and re.search(r"OutOfMemoryError|out of memory", str(x.get("error", ""))):
        return "oom"
    return x.get("status")


def collect(dirs) -> dict:
    groups: dict = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if not RUN_FILE.search(f.name):  # one record per job; renamed or partial files are not results
                continue
            x = load(f)
            if x.get("method") and x.get("status") in FINAL:
                groups.setdefault(vkey(x, f), []).append(x)
    return groups


def select(table: str, out: Path, runs_dirs=None, memory_dirs=None) -> dict:
    """Median per variant over its runs; per method, scale and size the variant with the lowest median latency among
    the table's variants with all N runs. Compilation modes other than a variant's own are used only where none of the
    method's variants completes (bf16 tables). Records with TF32 enabled or outside the table's variants are ignored."""
    t = TABLES[table]
    groups = collect(runs_dirs or [out / "runs"])
    mem = collect(memory_dirs or [out / "memory"])
    own = {(v, mode) for v, mode in variants(table)}
    known = {v for v, _ in variants(table)}
    rows: dict = {}
    for k, rs in groups.items():
        m, scale, shape, execution, fb, nat, mode, tf32 = k
        if execution != t["execution"] or tf32 != "off" or (m, fb, nat) not in known:
            continue
        primary = ((m, fb, nat), mode) in own
        if not primary and mode not in (NOCG, "default"):
            continue
        name = f"{m}|x{scale}|{shape[-2]}x{shape[-1]}"
        done = [x for x in rs if status(x) == "complete" and x.get("output_finite", True) is not False]
        if len(done) < t["repeats"]:
            if any(status(x) == "oom" for x in rs):
                rows.setdefault(name, []).append((primary, None))
            continue
        row = dict(method=m, scale=scale, height=shape[-2], width=shape[-1], execution=execution, fallback=fb,
                   natten_mode=nat, compile_mode=mode, mode_fallback=not primary, n=len(done),
                   latency_ms_median=statistics.median(x["latency_ms"] for x in done),
                   latency_ms_runs=[x["latency_ms"] for x in done],
                   peak_memory_mib_median=statistics.median(x["peak_memory_mib"] for x in done))
        if mode in CUDAGRAPH_MODES:
            mr = [x for x in mem.get((m, scale, shape, execution, fb, nat, NOCG, tf32), []) if status(x) == "complete"]
            row.update(peak_memory_source=f"{NOCG}, {len(mr)} runs",
                       peak_memory_mib_reported=statistics.median(x["peak_memory_mib"] for x in mr) if mr else None)
        else:
            row.update(peak_memory_source="timing runs", peak_memory_mib_reported=row["peak_memory_mib_median"])
        rows.setdefault(name, []).append((primary, row))
    picks: dict = {}
    for name, cands in rows.items():
        done = [r for p, r in cands if r and p] or [r for p, r in cands if r]
        m, scale = name.split("|")[:2]
        picks[name] = min(done, key=lambda r: r["latency_ms_median"]) if done else dict(method=m, scale=int(scale[1:]), status="oom")
    return dict(sorted(picks.items()))


def select_command(a) -> int:
    out = Path(a.results)
    picks = select(a.table, out, a.runs or None, a.memory or None)
    text = json.dumps(dict(table=a.table, setting=TABLES[a.table], picks=picks), indent=1) + "\n"
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(text)
    for k, p in picks.items():
        if p.get("status") == "oom":
            print(f"{k:32s} OOM")
        else:
            print(f"{k:32s} {p['latency_ms_median']:10.2f} ms {p['peak_memory_mib_reported'] or float('nan'):10.1f} MiB  "
                  f"{p['compile_mode'] or 'eager'} {p['fallback']} {p['natten_mode'] or ''}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    j = sub.add_parser("jobs", help="write a job list for run_queue.py")
    j.add_argument("--table", choices=TABLES, required=True)
    j.add_argument("--stage", choices=["probe", "full", "memory"], required=True)
    j.add_argument("--results", default=None, help="result root (default results/<table>)")
    j.add_argument("--output", required=True)
    j.add_argument("--skip-done", action="store_true", help="leave out jobs that already have a final record")
    s = sub.add_parser("select", help="medians per variant and the reported pick per method, scale and size")
    s.add_argument("--table", choices=TABLES, required=True)
    s.add_argument("--results", default=None, help="result root (default results/<table>)")
    s.add_argument("--runs", nargs="*", help="timing-run directories (default <results>/runs)")
    s.add_argument("--memory", nargs="*", help="memory-run directories (default <results>/memory)")
    s.add_argument("--output")
    a = p.parse_args(argv)
    a.results = a.results or f"results/{a.table}"
    return jobs_command(a) if a.command == "jobs" else select_command(a)


if __name__ == "__main__":
    raise SystemExit(main())
