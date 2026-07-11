#!/usr/bin/env python3
"""Test runner for mahjax_pt comprehensive test suite.

Usage:
    PYTHONPATH=. python mahjax_pt/tests/run_tests.py          # all tests
    PYTHONPATH=. python mahjax_pt/tests/run_tests.py --filter tile
    PYTHONPATH=. python mahjax_pt/tests/run_tests.py --list
"""

import sys, os, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mahjax_pt.tests.test_cases import ALL_TESTS as ALL_UNIT_TESTS
from mahjax_pt.tests.test_env_branches import ALL_TESTS as ALL_BRANCH_TESTS

# Merge all test registries
ALL_TESTS = {}
ALL_TESTS.update(ALL_UNIT_TESTS)
ALL_TESTS.update(ALL_BRANCH_TESTS)


def run_one_test(name, info, fn):
    """Run a single test function and return (name, passed, failed, details, elapsed)."""
    description, func = info
    t0 = time.time()
    try:
        results = func()
        passed = 0
        failed = 0
        details = []
        for item in results:
            case_name, actual, expected = item
            ok = (actual == expected)
            if isinstance(expected, bool):
                # For boolean tests, use is-True check
                ok = (actual is expected) if isinstance(actual, bool) else (actual == expected)
            if ok:
                passed += 1
                details.append(f"    [PASS] {case_name}")
            else:
                failed += 1
                details.append(f"    [FAIL] {case_name}: got {actual}, expected {expected}")
        elapsed = time.time() - t0
        return name, passed, failed, details, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        details = [f"    [ERROR] {e}", f"    {traceback.format_exc()}"]
        return name, 0, 1, details, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run mahjax_pt tests")
    parser.add_argument("--filter", default=None, help="Filter test names (substring match)")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all PASS details")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name, (desc, _) in ALL_TESTS.items():
            print(f"  {name:30s} {desc}")
        return

    # Select tests
    selected = {}
    for name, info in ALL_TESTS.items():
        if args.filter is None or args.filter in name:
            selected[name] = info

    if not selected:
        print(f"No tests match filter '{args.filter}'")
        return

    # Run
    print(f"{'='*70}")
    print(f"mahjax_pt Test Suite  ({len(selected)} test groups)")
    print(f"{'='*70}\n")

    total_pass = 0
    total_fail = 0
    total_elapsed = 0

    for name, info in selected.items():
        desc = info[0]
        name, passed, failed, details, elapsed = run_one_test(name, info, info[1])
        total_pass += passed
        total_fail += failed
        total_elapsed += elapsed

        status = "PASS" if failed == 0 else f"FAIL ({failed})"
        print(f"[{status}] {desc}  ({passed} ok, {failed} fail, {elapsed:.2f}s)")

        if failed > 0 or args.verbose:
            for d in details:
                print(d)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_pass} passed, {total_fail} failed, {total_elapsed:.1f}s total")
    if total_fail > 0:
        print(f"FAILURE: {total_fail} assertions failed!")
        sys.exit(1)
    else:
        print(f"SUCCESS: All tests pass!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
