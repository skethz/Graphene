#!/usr/bin/env python3
"""Serial timing queue: one fresh `profile.py run` process per job, on an otherwise idle node.

    python3 benchmarks/run_queue.py JOBS.json [--status STATUS.json] [--cpus 0-71] [--gpu-index 0]

A job is {"method", "scale", "execution", "height", "width", "warmup", "iterations",
"timeout", "output"} and optionally "natten_mode", "tf32" and "compile_mode" (as written by `profile.py jobs`;
other keys such as "tree" are ignored). Relative output paths are relative to the package root. Before each job the
queue waits until no compute process runs on any GPU of the node (or only on the chosen
GPU with --exclusive gpu), utilisation is <= 10 % and (with the CPU monitor) both CPU sets are near idle, for three
checks 5 s apart. A
foreign compute process that appears during a job marks it contaminated: the result
is renamed *.contaminatedN.json and the job is retried (at most three attempts). An
OOM or a timeout skips the remaining repeats of the same method/scale/execution/size.
Completed outputs are not rerun.

CPU monitor (--cpu-monitor, default on): the queue process moves itself to --monitor-cpus (outside the worker's CPU
set) and samples /proc once per second while a worker runs. It records the busy CPU time on the worker's CPUs that
does not belong to the worker's process group ("residual"), the busy time on all other CPUs, and every other process
that used CPU time. A run is contaminated, renamed and retried like a GPU-contaminated run, if the mean residual
exceeds --max-residual-cpu or the mean busy time on all other CPUs exceeds --max-other-cpus-busy (other users'
work on the other CPU socket also slows small launch-bound inputs).
The per-run record (means, maxima, top processes) is stored in the status file and in the result under "cpu_monitor".
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "benchmarks" / "profile.py"
JOB_KEYS = ("method", "scale", "execution", "height", "width", "warmup", "iterations")


def worker_argv(args, job: dict, output: Path) -> list[str]:
    argv = ["taskset", "-c", args.cpus, args.python, str(PROFILE), "run", "--threads", str(args.threads)]
    for k in JOB_KEYS:
        argv += [f"--{k}", str(job[k])]
    if "natten_mode" in job:
        argv += ["--natten-mode", job["natten_mode"]]
    if "tf32" in job:
        argv += ["--tf32", job["tf32"]]
    if "compile_mode" in job:
        argv += ["--compile-mode", job["compile_mode"]]
    if "fallback" in job:
        argv += ["--fallback", job["fallback"]]
    return argv + ["--output", str(output)] + args.extra_args.split()


def parse_cpus(text: str) -> list[int]:
    cpus = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        cpus += list(range(int(a), int(b or a) + 1))
    return cpus


def cpu_busy() -> dict[int, int]:
    busy = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                v = line.split()
                n = list(map(int, v[1:]))
                busy[int(v[0][3:])] = sum(n) - n[3] - n[4]  # all but idle and iowait
    return busy


def proc_times() -> dict[int, tuple]:
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            r = s.rfind(")")
            v = s[r + 2:].split()
            out[int(d)] = (s[s.find("(") + 1:r], int(v[11]) + int(v[12]), int(v[2]), int(v[36]))
        except (OSError, ValueError, IndexError):
            pass
    return out


class CpuMonitor(threading.Thread):
    """1 s samples of worker-CPU residual busy time, other-CPU busy time and per-process CPU use."""

    def __init__(self, worker_pgid: int, worker_cpus: list[int]):
        super().__init__(daemon=True)
        self.pgid, self.cpus, self.stop_event = worker_pgid, set(worker_cpus), threading.Event()
        self.samples, self.per_proc, self.hz = [], {}, os.sysconf("SC_CLK_TCK")

    def run(self):
        c0, p0, t0 = cpu_busy(), proc_times(), time.time()
        while not self.stop_event.wait(1.0):
            c1, p1, t1 = cpu_busy(), proc_times(), time.time()
            dt = t1 - t0
            busy_w = sum(c1[c] - c0.get(c, c1[c]) for c in c1 if c in self.cpus) / self.hz / dt
            busy_o = sum(c1[c] - c0.get(c, c1[c]) for c in c1 if c not in self.cpus) / self.hz / dt
            worker = 0.0
            for pid, (comm, t, pgrp, cpu) in p1.items():
                d = (t - p0[pid][1]) / self.hz if pid in p0 else 0.0
                if pgrp == self.pgid:
                    worker += d
                elif d > 0 and pid != os.getpid():
                    rec = self.per_proc.setdefault(pid, {"comm": comm, "cpu_s": 0.0, "last_cpu": cpu})
                    rec["cpu_s"] += d
                    rec["last_cpu"] = cpu
            self.samples.append((round(t1, 2), round(busy_w - worker / dt, 3), round(busy_o, 3), round(worker / dt, 3)))
            c0, p0, t0 = c1, p1, t1

    def summary(self, seconds: float) -> dict:
        res = [s[1] for s in self.samples] or [0.0]
        oth = [s[2] for s in self.samples] or [0.0]
        procs = sorted(({"pid": pid, **v, "cpu_mean": v["cpu_s"] / max(seconds, 1e-9)} for pid, v in self.per_proc.items()),
                       key=lambda r: -r["cpu_s"])[:10]
        for r in procs:
            try:
                r["user"] = __import__("pwd").getpwuid(os.stat(f"/proc/{r['pid']}").st_uid).pw_name
            except (OSError, KeyError):
                r["user"] = None
        return {"samples": len(self.samples), "residual_cpu_mean": sum(res) / len(res), "residual_cpu_max": max(res),
                "other_cpus_busy_mean": sum(oth) / len(oth), "other_cpus_busy_max": max(oth), "top_processes": procs,
                "series": self.samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--status", type=Path, default=Path("results/queue_status.json"))
    parser.add_argument("--gpu-index", default="0", help="physical GPU index (CUDA_DEVICE_ORDER=PCI_BUS_ID)")
    parser.add_argument("--cpus", default="0-71", help="taskset CPU list for the worker")
    parser.add_argument("--threads", type=int, default=72)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--exclusive", choices=["node", "gpu"], default="node")
    parser.add_argument("--extra-args", default="", help="appended to every worker, e.g. '--natten-mode fp32'")
    parser.add_argument("--cpu-monitor", choices=["on", "off"], default="on")
    parser.add_argument("--monitor-cpus", default="140-143", help="CPUs for the queue process itself (outside --cpus)")
    parser.add_argument("--max-residual-cpu", type=float, default=0.30,
                        help="contaminated if the mean non-worker busy time on the worker CPUs exceeds this (CPUs)")
    parser.add_argument("--max-other-cpus-busy", type=float, default=1.0,
                        help="contaminated if the mean busy time on all CPUs outside --cpus exceeds this (CPUs)")
    parser.add_argument("--dry-run", action="store_true", help="print the worker command of every job and exit")
    args = parser.parse_args()
    if args.dry_run:
        for job in json.loads(args.jobs.read_text()):
            print(" ".join(worker_argv(args, job, Path(job["output"]))))
        return 0

    if args.cpu_monitor == "on":
        os.sched_setaffinity(0, parse_cpus(args.monitor_cpus))
    status_path = args.status if args.status.is_absolute() else ROOT / args.status
    status_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu_index, "--query-gpu=uuid", "--format=csv,noheader"],
                                       text=True).strip()
    state = {"status": "running", "gpu": gpu_uuid, "jobs": str(args.jobs), "completed": [], "failed": [],
             "skipped": [], "contaminated": [], "active": None}

    def save():
        state["updated"] = time.time()
        status_path.write_text(json.dumps(state, indent=2) + "\n")

    def foreign_pids() -> set[int]:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        pids = set()
        for row in out.splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) == 2 and (args.exclusive == "node" or parts[0] == gpu_uuid):
                pids.add(int(parts[1]))
        return pids

    worker_cpus = parse_cpus(args.cpus)

    def cpu_quiet() -> bool:
        """No worker runs here, so all busy time is someone else's: both CPU sets must be near idle for 1 s."""
        if args.cpu_monitor != "on":
            return True
        hz, c0 = os.sysconf("SC_CLK_TCK"), cpu_busy()
        time.sleep(1.0)
        c1 = cpu_busy()
        busy_w = sum(c1[c] - c0.get(c, c1[c]) for c in c1 if c in worker_cpus) / hz
        busy_o = sum(c1[c] - c0.get(c, c1[c]) for c in c1 if c not in worker_cpus) / hz
        state["last_idle_cpu"] = {"worker_cpus_busy": round(busy_w, 3), "other_cpus_busy": round(busy_o, 3)}
        return busy_w <= args.max_residual_cpu and busy_o <= args.max_other_cpus_busy

    def idle() -> bool:
        if foreign_pids() or not cpu_quiet():
            return False
        query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        if args.exclusive == "gpu":
            query[1:1] = ["-i", args.gpu_index]
        return max(int(x) for x in subprocess.check_output(query, text=True).split()) <= 10

    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu_index, PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="8",
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / "cache/inductor"), TRITON_CACHE_DIR=str(ROOT / "cache/triton"))
    logs = ROOT / "results" / "queue_logs"
    logs.mkdir(parents=True, exist_ok=True)
    dead: set[tuple] = set()
    for job in json.loads(args.jobs.read_text()):
        output = Path(job["output"])
        output = output if output.is_absolute() else ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        label = output.stem
        key = tuple(job.get(k) for k in ("method", "scale", "execution", "height", "width", "natten_mode", "tf32", "compile_mode", "fallback"))
        if key in dead:
            state["skipped"].append(label)
            save()
            continue
        if output.exists() and json.loads(output.read_text()).get("status") == "complete":
            state["completed"].append(label)
            save()
            continue
        for attempt in range(1, 4):
            clear = 0
            while clear < 3:
                clear = clear + 1 if idle() else 0
                state["active"] = {"label": label, "phase": "waiting_for_idle", "clear_checks": clear}
                save()
                if clear < 3:
                    time.sleep(5)
            argv = worker_argv(args, job, output)
            log_path = logs / f"{label}_attempt{attempt}.log"
            start, foreign, timed_out = time.time(), set(), False
            print("START", label, attempt, flush=True)
            with log_path.open("w") as log:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                monitor = CpuMonitor(proc.pid, parse_cpus(args.cpus)) if args.cpu_monitor == "on" else None
                if monitor:
                    monitor.start()
                state["active"] = {"label": label, "phase": "running", "pid": proc.pid, "attempt": attempt,
                                   "started": start, "argv": argv}
                save()
                last_gpu_check = 0.0
                while proc.poll() is None:
                    gpu_pids = set()
                    if time.time() - last_gpu_check >= 5:  # nvidia-smi is heavy: every 5 s, on the monitor CPUs
                        last_gpu_check, gpu_pids = time.time(), foreign_pids()
                    for pid in gpu_pids:
                        try:
                            if os.getpgid(pid) != proc.pid:
                                foreign.add(pid)
                        except ProcessLookupError:
                            pass
                    if time.time() - start > job.get("timeout", 1800):
                        os.killpg(proc.pid, signal.SIGTERM)
                        timed_out = True
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                        break
                    time.sleep(1)
                code = proc.wait()
            seconds = time.time() - start
            cpu = None
            if monitor:
                monitor.stop_event.set()
                monitor.join(timeout=5)
                cpu = monitor.summary(seconds)
            result = json.loads(output.read_text()) if output.exists() else {}
            cpu_bad = bool(cpu and (cpu["residual_cpu_mean"] > args.max_residual_cpu
                                    or cpu["other_cpus_busy_mean"] > args.max_other_cpus_busy))
            if cpu and output.exists() and result:
                result["cpu_monitor"] = cpu
                output.write_text(json.dumps(result, indent=2) + "\n")
            record = {"label": label, "returncode": code, "seconds": seconds,
                      "foreign_pids": sorted(foreign), "cpu_contaminated": cpu_bad, "timed_out": timed_out,
                      "result_status": result.get("status"), "log": log_path.name}
            if cpu:
                record["cpu"] = {k: v for k, v in cpu.items() if k != "series"}
            if (foreign or cpu_bad) and not timed_out and result.get("status") != "oom":
                state["contaminated"].append(record)
                if output.exists():
                    output.rename(output.with_suffix(f".contaminated{attempt}.json"))
                save()
                continue
            if timed_out or result.get("status") == "oom":
                dead.add(key)
                state["failed"].append(record)
            elif code or result.get("status") != "complete":
                state["failed"].append(record)
            else:
                state["completed"].append(label)
            state["active"] = None
            save()
            print("END", label, code, round(record["seconds"], 1), result.get("status"), flush=True)
            break
        else:
            state["failed"].append({"label": label, "error": "three contaminated attempts"})
            save()
    state.update(status="complete", active=None)
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
