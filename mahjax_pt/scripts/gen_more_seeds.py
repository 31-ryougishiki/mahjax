#!/usr/bin/env python3
"""Generate additional golden data seeds with shanten-guided action selection.

Picks seeds not already present in the output directory, stratified across
a wide range to maximise path diversity.

Usage:
  # Generate 300 new seeds (default)
  python gen_more_seeds.py

  # Custom count and parallel workers
  python gen_more_seeds.py -n 500 -j 8

  # Custom output dir and exploration params
  python gen_more_seeds.py -o ./golden_data --pass-epsilon 0.2 --no-meld-prob 0.2

  # Dry-run: just print the seed list, don't record
  python gen_more_seeds.py --dry-run
"""
import os, sys, re, subprocess, argparse, signal, time
import numpy as np


# ── Ctrl+C handling ──
_interrupted = False


def _on_interrupt(signum, frame):
    global _interrupted
    _interrupted = True
    # Restore default handler so a second Ctrl+C hard-kills
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    print("\n[INTERRUPTED] Ctrl+C received — stopping after current batch...",
          flush=True)


def _terminate_process(proc):
    """Terminate *proc* and its child tree.  Tries graceful shutdown first,
    then escalates to force-kill."""
    if proc.poll() is not None:
        return  # already exited

    # 1. On Windows, send CTRL_BREAK_EVENT to the process group.
    #    This lets record_jax_golden.py's own signal handler clean up its
    #    multiprocessing pool.
    if sys.platform == 'win32':
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            pass

    # 2. SIGTERM (Unix) / terminate (Windows)
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    # 3. SIGKILL / kill
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def existing_seeds(output_dir: str) -> set:
    """Return set of seed ids already present in *output_dir*."""
    seeds = set()
    if not os.path.isdir(output_dir):
        return seeds
    for f in os.listdir(output_dir):
        m = re.match(r'golden_seed_(\d+)\.pkl', f)
        if m:
            seeds.add(int(m.group(1)))
    return seeds


def pick_new_seeds(n: int, existing: set, rng: np.random.RandomState) -> list:
    """Pick *n* new seeds via stratified random sampling.

    Stratification ensures we don't cluster in a single range (consecutive
    seeds produce correlated initial hands, wasting coverage).
    """
    # ── Define strata ──
    # Each stratum is (lo, hi, weight).  Weight determines how many seeds
    # are drawn from that stratum.  Ranges below 50 000 were informed by
    # scan_rare_paths.py: 3000–3153 and 10000–10499 had the highest
    # rare-action density under the old (legal[0]) policy.
    strata = [
        # (lo,        hi,      weight,  description)
        (0,          3000,    30,      "low range (original seeds area)"),
        (3000,       5000,    50,      "mid-low (high rare-action density)"),
        (5000,      10000,    40,      "mid range (sparse in old data)"),
        (10000,     20000,    80,      "mid-high (highest rare-action density)"),
        (20000,     40000,    60,      "high range (sparse in old data)"),
        (40000,    100000,    40,      "very high (completely unexplored)"),
    ]
    total_weight = sum(w for _, _, w, _ in strata)

    chosen = []
    seen = set(existing)  # start with existing to avoid duplicates

    for lo, hi, weight, desc in strata:
        # How many seeds to draw from this stratum (proportional to weight).
        # Distribute remaining quota evenly.
        quota = max(1, int(n * weight / total_weight))
        # Adjust for already-picked seeds
        remaining = n - len(chosen)
        if remaining <= 0:
            break
        quota = min(quota, remaining)

        # Collect candidates in this stratum that aren't taken yet
        candidates = [s for s in range(lo, hi) if s not in seen]
        if len(candidates) <= quota:
            pick = candidates
        else:
            # Random sample without replacement
            idx = rng.choice(len(candidates), size=quota, replace=False)
            pick = [candidates[i] for i in idx]

        for s in pick:
            seen.add(s)
        chosen.extend(pick)
        print(f"  {desc}: {len(pick)} seeds (range {lo}–{hi-1})", flush=True)

    # If we still need more (unlikely), fill from a wide range
    if len(chosen) < n:
        extra_needed = n - len(chosen)
        extra_candidates = [s for s in range(max(seen) + 1, max(seen) + 100000)
                           if s not in seen]
        if len(extra_candidates) > extra_needed:
            idx = rng.choice(len(extra_candidates), size=extra_needed, replace=False)
            extra_candidates = [extra_candidates[i] for i in idx]
        chosen.extend(extra_candidates[:extra_needed])
        print(f"  overflow: {min(extra_needed, len(extra_candidates))} seeds", flush=True)

    return sorted(chosen)


def main():
    global _interrupted
    p = argparse.ArgumentParser(
        description="Generate additional golden data seeds with shanten-guided policy")
    p.add_argument('-n', '--count', type=int, default=300,
                   help='Number of new seeds to generate (default: 300)')
    p.add_argument('-j', '--jobs', type=int, default=4,
                   help='Parallel workers (default: 4)')
    p.add_argument('-o', '--output', default='golden_data',
                   help='Output directory (default: golden_data)')
    p.add_argument('--pass-epsilon', type=float, default=0.15,
                   help='PASS exploration probability (default: 0.15)')
    p.add_argument('--no-meld-prob', type=float, default=0.15,
                   help='No-meld exploration probability (default: 0.15)')
    p.add_argument('--action-seed', type=int, default=None,
                   help='Fixed action-selection RNG seed')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed for the seed-picker RNG (default: 42)')
    p.add_argument('--dry-run', action='store_true',
                   help='Only print chosen seeds, do not record')
    p.add_argument('--batch-size', type=int, default=100,
                   help='Seeds per record_jax_golden invocation (default: 100)')
    args = p.parse_args()

    # ── Find the record script ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    record_script = os.path.join(script_dir, 'record_jax_golden.py')
    if not os.path.isfile(record_script):
        sys.exit(f"ERROR: cannot find {record_script}")

    # ── Pick seeds ──
    existing = existing_seeds(args.output)
    print(f"Output directory: {args.output}/")
    print(f"Existing seeds:    {len(existing)}")
    print(f"Target new seeds:  {args.count}")
    print()

    rng = np.random.RandomState(args.seed)
    chosen = pick_new_seeds(args.count, existing, rng)

    print(f"\nTotal chosen: {len(chosen)} seeds")
    print(f"Range: {min(chosen)} – {max(chosen)}")
    print()

    if args.dry_run:
        print("DRY RUN — seed list:")
        for i, s in enumerate(chosen):
            print(f"  {i+1:>3d}. {s}")
        print(f"\nTo record, run without --dry-run")
        return

    # ── Install Ctrl+C handler ──
    signal.signal(signal.SIGINT, _on_interrupt)

    # ── Run record_jax_golden in batches ──
    os.makedirs(args.output, exist_ok=True)

    # Build base command parts
    base_cmd = [
        sys.executable, record_script,
        '-o', args.output,
        '-j', str(args.jobs),
        '--pass-epsilon', str(args.pass_epsilon),
        '--no-meld-prob', str(args.no_meld_prob),
    ]
    if args.action_seed is not None:
        base_cmd += ['--action-seed', str(args.action_seed)]

    # Windows: CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT to the
    # whole child tree (including the multiprocessing pool workers).
    popen_kwargs = {}
    if sys.platform == 'win32':
        popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

    n_batches = (len(chosen) + args.batch_size - 1) // args.batch_size
    total_ok = 0
    total_fail = 0
    completed_seeds = set()

    for batch_i in range(n_batches):
        if _interrupted:
            print("\n[SKIPPED] remaining batches due to interrupt", flush=True)
            break

        start = batch_i * args.batch_size
        end = min(start + args.batch_size, len(chosen))
        batch_seeds = chosen[start:end]

        cmd = base_cmd + [str(s) for s in batch_seeds]
        print(f"\n{'='*60}")
        print(f"Batch {batch_i+1}/{n_batches}: {len(batch_seeds)} seeds "
              f"(#{start+1}–#{end} of {len(chosen)})")
        print(f"{'='*60}")
        print(flush=True)

        # Launch child process
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(script_dir),
                                **popen_kwargs)

        # Poll until completion or interrupt
        try:
            while proc.poll() is None:
                if _interrupted:
                    _terminate_process(proc)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            _interrupted = True
            _terminate_process(proc)

        rc = proc.poll()
        if rc == 0:
            total_ok += len(batch_seeds)
            for s in batch_seeds:
                completed_seeds.add(s)
        elif rc is None:
            # Process was killed by us — count as partial failure
            # but check which seeds actually made it to disk
            saved = 0
            for s in batch_seeds:
                if os.path.exists(os.path.join(args.output,
                                               f'golden_seed_{s:04d}.pkl')):
                    completed_seeds.add(s)
                    saved += 1
            total_ok += saved
            total_fail += len(batch_seeds) - saved
            print(f"Batch killed: {saved}/{len(batch_seeds)} seeds saved",
                  flush=True)
        else:
            total_fail += len(batch_seeds)
            print(f"WARNING: batch returned exit code {rc}", flush=True)

    n_completed = len(completed_seeds)
    print(f"\n{'='*60}")
    print(f"DONE: {n_completed} seeds recorded ({total_fail} failed)")
    print(f"Golden data in: {os.path.abspath(args.output)}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
