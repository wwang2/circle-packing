"""Circle packing evaluator.

Evaluates a packing of n circles in a unit square [0,1]^2.
A solution is a list of (x, y, r) tuples — center coordinates and radius.

Metric: sum of radii (maximize).
Validity: all containment and non-overlap constraints satisfied within tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import yaml
from pathlib import Path


CONSTRAINT_TOL = 1e-10


def validate_packing(circles: list[tuple[float, float, float]], tol: float = CONSTRAINT_TOL) -> dict:
    """Validate a circle packing and return detailed results.

    Args:
        circles: List of (x, y, r) tuples.
        tol: Constraint violation tolerance.

    Returns:
        Dict with keys: valid, metric, max_violation, violations (list of strings).
    """
    n = len(circles)
    if n == 0:
        return {"valid": False, "metric": 0.0, "raw_metric": 0.0,
                "max_violation": float("inf"),
                "violations": ["Empty packing"], "n": 0}

    violations = []
    max_violation = 0.0

    for i, (x, y, r) in enumerate(circles):
        # Check for NaN/Inf values
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            violations.append(f"Circle {i}: non-finite values x={x}, y={y}, r={r}")
            max_violation = float("inf")
            continue

        # Check positive radius
        if r <= 0:
            violations.append(f"Circle {i}: non-positive radius r={r}")
            max_violation = max(max_violation, abs(r))
            continue

        # Check containment: r <= x, x <= 1-r, r <= y, y <= 1-r
        viol_left = r - x
        viol_right = (x + r) - 1.0
        viol_bottom = r - y
        viol_top = (y + r) - 1.0

        for name, v in [("left", viol_left), ("right", viol_right),
                        ("bottom", viol_bottom), ("top", viol_top)]:
            if v > tol:
                violations.append(f"Circle {i}: {name} containment violation = {v:.2e}")
                max_violation = max(max_violation, v)

    # Check non-overlap: dist(i,j) >= r_i + r_j
    for i in range(n):
        xi, yi, ri = circles[i]
        if ri <= 0:
            continue
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            if rj <= 0:
                continue
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            min_dist = ri + rj
            min_dist_sq = min_dist ** 2
            # Use squared distances for comparison to avoid sqrt when possible
            if dist_sq < min_dist_sq:
                dist = math.sqrt(dist_sq)
                overlap = min_dist - dist
                if overlap > tol:
                    violations.append(
                        f"Circles {i},{j}: overlap violation = {overlap:.2e} "
                        f"(dist={dist:.10f}, r_i+r_j={min_dist:.10f})"
                    )
                    max_violation = max(max_violation, overlap)

    metric = sum(r for _, _, r in circles)
    valid = len(violations) == 0

    return {
        "valid": valid,
        "metric": metric if valid else 0.0,
        "raw_metric": metric,
        "max_violation": max_violation,
        "violations": violations,
        "n": n,
    }


def load_solution(path: str) -> list[tuple[float, float, float]]:
    """Load a solution from a JSON file.

    Expected format: {"circles": [[x1, y1, r1], [x2, y2, r2], ...]}
    or just: [[x1, y1, r1], [x2, y2, r2], ...]
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        circles_raw = data.get("circles", data.get("solution", []))
    elif isinstance(data, list):
        circles_raw = data
    else:
        raise ValueError(f"Unexpected format in {path}")

    circles = []
    for item in circles_raw:
        if len(item) != 3:
            raise ValueError(f"Expected [x, y, r], got {item}")
        circles.append((float(item[0]), float(item[1]), float(item[2])))

    return circles


def run_sanity_checks(config_path: str = "research/eval/config.yaml") -> bool:
    """Run sanity checks defined in config.yaml. Returns True if all pass."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    checks = config.get("sanity_checks", [])
    all_pass = True

    for check in checks:
        name = check["name"]
        print(f"\n--- Sanity check: {name} ---")
        print(f"  Description: {check.get('description', 'N/A')}")

        if check.get("expected") == "error":
            # Expect an error when loading/validating
            try:
                circles = load_solution(check["input"])
                result = validate_packing(circles)
                if result["valid"]:
                    print(f"  FAIL: Expected error but got valid packing (metric={result['metric']:.6f})")
                    all_pass = False
                else:
                    print(f"  PASS: Invalid packing as expected")
            except Exception as e:
                print(f"  PASS: Got expected error: {e}")
            continue

        # Normal check: load, validate, check range
        try:
            circles = load_solution(check["input"])
            result = validate_packing(circles)

            if not result["valid"]:
                print(f"  FAIL: Packing is invalid. Violations: {result['violations'][:3]}")
                all_pass = False
                continue

            metric = result["metric"]
            expected = check.get("expected_range", [None, None])
            lo, hi = expected

            if lo is not None and metric < lo - 1e-6:
                print(f"  FAIL: metric {metric:.6f} below expected minimum {lo}")
                all_pass = False
            elif hi is not None and metric > hi + 1e-6:
                print(f"  FAIL: metric {metric:.6f} above expected maximum {hi}")
                all_pass = False
            else:
                print(f"  PASS: metric = {metric:.6f} (expected [{lo}, {hi}])")
        except Exception as e:
            print(f"  FAIL: {e}")
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Circle packing evaluator")
    parser.add_argument("solution", nargs="?", help="Path to solution JSON file")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity checks from config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed violations")
    args = parser.parse_args()

    if args.sanity_check:
        ok = run_sanity_checks()
        sys.exit(0 if ok else 1)

    if not args.solution:
        parser.error("Either provide a solution file or use --sanity-check")

    circles = load_solution(args.solution)
    result = validate_packing(circles)

    print(f"n = {result['n']}")
    print(f"valid = {result['valid']}")
    print(f"metric (sum of radii) = {result['raw_metric']:.10f}")
    if not result["valid"]:
        print(f"INVALID — score = 0.0")
        print(f"max_violation = {result['max_violation']:.2e}")
        if args.verbose:
            for v in result["violations"]:
                print(f"  {v}")
    else:
        print(f"VALID — score = {result['metric']:.10f}")

    # Output machine-readable result
    print(f"\n__RESULT__={json.dumps({'valid': result['valid'], 'metric': result['metric'], 'n': result['n']})}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
