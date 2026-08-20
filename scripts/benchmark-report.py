#!/usr/bin/env python
"""Cross-reference BENCHMARKS.md against the tests that enforce it.

A green `pytest -m benchmark` only proves the tests that exist pass. This asks
the other question: does every gate written down in BENCHMARKS.md actually have
a test behind it, and did that test run? A gate with no test is the failure mode
worth catching — it reads as covered and is not.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = re.compile(r"^\|\s*([XRBMSAP]\d)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
SECTION = re.compile(r"^##\s+(.*)$")
IN_DOC = re.compile(r"\b([XRBMSAP]\d)\b")


def gates() -> dict[str, tuple[str, str, str]]:
    """id -> (section, gate, threshold), in file order."""
    out, section = {}, "Cross-cutting"
    for line in (ROOT / "BENCHMARKS.md").read_text().splitlines():
        header = SECTION.match(line)
        if header:
            section = header.group(1).split("·")[-1].strip()
            continue
        hit = GATE.match(line)
        if hit:
            out[hit.group(1)] = (section, hit.group(2), hit.group(3))
    return out


def tests_by_gate() -> tuple[dict[str, list[str]], list[str]]:
    """gate id -> test names, plus benchmark tests naming no gate."""
    mapping: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            marks = ast.unparse(ast.Module(node.decorator_list, []))
            if "benchmark" not in marks:
                continue
            doc = ast.get_docstring(node) or ""
            found = sorted(set(IN_DOC.findall(doc.split("\n\n")[0])))
            if found:
                for gate in found:
                    mapping.setdefault(gate, []).append(node.name)
            else:
                unmapped.append(f"{path.name}::{node.name}")
    return mapping, unmapped


def outcomes() -> dict[str, str]:
    """test function name -> worst outcome across its parametrisations."""
    proc = subprocess.run(
        # -o addopts= clears the "-q" in pyproject.toml, which would otherwise
        # suppress the per-test lines this function exists to read.
        [sys.executable, "-m", "pytest", "-m", "benchmark", "-o", "addopts=",
         "-v", "-p", "no:warnings", "--no-header", "-rN"],
        cwd=ROOT, capture_output=True, text=True)
    result: dict[str, str] = {}
    node_count = 0
    for line in proc.stdout.splitlines():
        hit = re.match(r"tests/\S+::(\w+)(?:\[.*\])?\s+(PASSED|FAILED|ERROR|SKIPPED)", line)
        if hit:
            node_count += 1
            name, verdict = hit.group(1), hit.group(2)
            rank = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2, "ERROR": 3}
            if name not in result or rank[verdict] > rank[result[name]]:
                result[name] = verdict
    result["__nodes__"] = str(node_count)
    return result


def main() -> int:
    catalogue = gates()
    mapping, unmapped = tests_by_gate()
    ran = outcomes()
    node_count = int(ran.pop("__nodes__", "0"))

    section = None
    uncovered, failing = [], []
    for gate, (sec, text, threshold) in catalogue.items():
        if sec != section:
            section = sec
            print(f"\n\033[1m{section}\033[0m")
        names = mapping.get(gate, [])
        verdicts = {ran.get(n, "NOT RUN") for n in names} or {"NO TEST"}
        worst = ("NO TEST" if not names else
                 "FAILED" if "FAILED" in verdicts or "ERROR" in verdicts else
                 "NOT RUN" if "NOT RUN" in verdicts else
                 "SKIPPED" if "SKIPPED" in verdicts else "PASS")
        mark = {"PASS": "\033[32m✓\033[0m", "SKIPPED": "\033[33m~\033[0m",
                "FAILED": "\033[31m✗\033[0m", "NO TEST": "\033[31m!\033[0m",
                "NOT RUN": "\033[31m?\033[0m"}[worst]
        if worst == "NO TEST":
            uncovered.append(gate)
        elif worst in ("FAILED", "NOT RUN"):
            failing.append(gate)
        print(f"  {mark} {gate}  {text[:58]:<58} {len(names)} test(s)")
        if worst != "PASS":
            print(f"      threshold: {threshold}")

    print(f"\n\033[1mSummary\033[0m")
    print(f"  gates documented   {len(catalogue)}")
    print(f"  gates with a test  {len(catalogue) - len(uncovered)}")
    print(f"  benchmark tests    {len(ran)} functions "
          f"({node_count} including parametrisations), "
          f"{sum(1 for v in ran.values() if v == 'PASSED')} passed, "
          f"{sum(1 for v in ran.values() if v == 'SKIPPED')} skipped")
    if unmapped:
        print(f"  extra coverage     {len(unmapped)} benchmark test(s) naming no gate:")
        for name in unmapped:
            print(f"      {name}")
    if uncovered:
        print(f"  \033[31mUNCOVERED GATES: {', '.join(uncovered)}\033[0m")
    if failing:
        print(f"  \033[31mFAILING GATES:   {', '.join(failing)}\033[0m")
    return 1 if (uncovered or failing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
