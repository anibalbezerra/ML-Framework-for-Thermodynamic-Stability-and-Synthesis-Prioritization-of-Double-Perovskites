#!/usr/bin/env python3
"""
Serial, resumable batch phonon runner for MatterSim/CHGNet workflows.

- Reads materials from a CSV with a column containing mp-IDs
- Runs them sequentially, one by one
- Resumes automatically (skips DONE or FAILED jobs that already have results)
- Saves per-job status as JSON under <outdir>/status/
"""

import os
import csv
import json
import time
import argparse
import subprocess
from datetime import datetime


def now_iso():
    """Return ISO timestamp (UTC) for logging."""
    return datetime.now().replace(microsecond=0).isoformat() + "Z"


def write_status(outdir, mp_id, status):
    """Safely write per-job JSON status."""
    status_dir = os.path.join(outdir, "status")
    os.makedirs(status_dir, exist_ok=True)
    path = os.path.join(status_dir, f"{mp_id}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def read_status(outdir, mp_id):
    """Return saved status or None if not existing."""
    path = os.path.join(outdir, "status", f"{mp_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def run_job(mp_id, script, python_path, outdir, extra_args):
    """Run one phonon job serially."""
    job_outdir = os.path.join(outdir, mp_id)
    os.makedirs(job_outdir, exist_ok=True)

    cmd = [python_path, script,
           "--mp-id", mp_id,
           "--output-dir", job_outdir,
           "--cache-dir", os.path.join(job_outdir, "cache")
           ] + extra_args

    print(f"[{mp_id}] Launching: {' '.join(cmd)}", flush=True)
    start = time.time()

    log_out = os.path.join(job_outdir, "stdout.log")
    log_err = os.path.join(job_outdir, "stderr.log")

    # Clean PYTHONPATH to avoid Intel OneAPI conflicts
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    with open(log_out, "w") as fout, open(log_err, "w") as ferr:
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, env=env)
        rc = proc.wait()

    elapsed = time.time() - start
    status = "DONE" if rc == 0 else "FAILED"
    print(f"[{mp_id}] Finished {status} in {elapsed:.1f}s (exit {rc})", flush=True)

    write_status(outdir, mp_id, {
        "mp_id": mp_id,
        "status": status,
        "returncode": rc,
        "runtime_sec": elapsed,
        "timestamp": now_iso()
    })


def main():
    ap = argparse.ArgumentParser(description="Serial, resumable batch phonon runner")
    ap.add_argument("--csv", required=True, help="Input CSV containing mp-ids")
    ap.add_argument("--script", required=True, help="Path to run_phonon_MATERSIM.py")
    ap.add_argument("--python", required=True, help="Python executable (e.g. ~/envs/tblite312/bin/python)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Extra args passed to phonon script")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    status_dir = os.path.join(args.outdir, "status")
    os.makedirs(status_dir, exist_ok=True)

    # Read & sort rows by nsites (ascending = small → large)
    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nsites = int(row["nsites"])
            except:
                nsites = 10**9  # send malformed rows to the end
            rows.append((row, nsites))

    # Sort by size (smallest structure first)
    rows.sort(key=lambda x: x[1])

    mp_ids = []
    for row, _ in rows:
        mp_id = row["material_id"].strip()
        mp_ids.append(mp_id)

    print(f"Found {len(mp_ids)} materials in CSV (sorted by nsites)")
    print("Sorted order (first 10):")
    for r in rows[:10]:
        print(f"  {r[0]['material_id']}  (nsites={r[1]})")

    print(f"Found {len(mp_ids)} materials in CSV")
    print(f"Using env python: {args.python}")
    print(f"Output dir: {args.outdir}")
    

    # Serial execution with resuming
    for i, mp_id in enumerate(mp_ids, 1):
        st = read_status(args.outdir, mp_id)
        if st and st.get("status") == "DONE":
            print(f"[{mp_id}] Skipping (already DONE)", flush=True)
            continue

        print(f"\n=== [{i}/{len(mp_ids)}] Running {mp_id} ===", flush=True)
        write_status(args.outdir, mp_id, {"status": "RUNNING", "start_time": now_iso()})
        try:
            run_job(mp_id, args.script, args.python, args.outdir, args.extra_args or [])
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting cleanly.")
            break
        except Exception as e:
            print(f"[{mp_id}] Exception: {e}", flush=True)
            write_status(args.outdir, mp_id, {"status": "FAILED", "error": str(e)})

    print("All materials processed or skipped.")


if __name__ == "__main__":
    main()
