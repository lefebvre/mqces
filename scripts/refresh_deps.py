#!/usr/bin/env python3
"""Query upstream git remotes for the latest stable release tag of each
FetchContent dependency declared in cmake/Dependencies.cmake, and either
print a diff or rewrite the file in place.

Dependencies are pinned by commit (MQCES_DEP_*_COMMIT); the tag
(MQCES_DEP_*_TAG) records which release that commit came from. A bump
rewrites both. The dry run also re-resolves each current tag and reports
any whose upstream commit no longer matches the pin, i.e. a moved tag.

Usage:
    python3 scripts/refresh_deps.py            # dry-run: print current vs latest
    python3 scripts/refresh_deps.py --apply    # rewrite cmake/Dependencies.cmake
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPS_FILE = REPO_ROOT / "cmake" / "Dependencies.cmake"

PRERELEASE_RE = re.compile(r"(rc|alpha|beta|nightly|pre)", re.IGNORECASE)
SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?$")


@dataclass(frozen=True)
class Dep:
    var: str            # the MQCES_DEP_*_TAG variable name
    url: str            # upstream git URL

    @property
    def commit_var(self) -> str:
        return self.var.removesuffix("_TAG") + "_COMMIT"


DEPS: tuple[Dep, ...] = (
    Dep("MQCES_DEP_EIGEN_TAG",     "https://gitlab.com/libeigen/eigen.git"),
    Dep("MQCES_DEP_GTEST_TAG",     "https://github.com/google/googletest.git"),
    Dep("MQCES_DEP_BENCHMARK_TAG", "https://github.com/google/benchmark.git"),
    Dep("MQCES_DEP_NANOBIND_TAG",  "https://github.com/wjakob/nanobind.git"),
    Dep("MQCES_DEP_JSON_TAG",      "https://github.com/nlohmann/json.git"),
)


def semver_key(tag: str) -> tuple[int, int, int] | None:
    m = SEMVER_RE.match(tag)
    if not m:
        return None
    major, minor, patch = m.groups()
    return (int(major), int(minor), int(patch or 0))


def latest_stable_tag(url: str) -> str:
    """Return the highest stable semver tag advertised by `url`."""
    out = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", url],
        check=True, capture_output=True, text=True,
    ).stdout
    candidates: list[tuple[tuple[int, int, int], str]] = []
    for line in out.splitlines():
        ref = line.split("\t", 1)[1] if "\t" in line else ""
        if not ref.startswith("refs/tags/"):
            continue
        tag = ref[len("refs/tags/"):]
        if PRERELEASE_RE.search(tag):
            continue
        key = semver_key(tag)
        if key is None:
            continue
        candidates.append((key, tag))
    if not candidates:
        raise RuntimeError(f"no stable semver tag found at {url}")
    candidates.sort()
    return candidates[-1][1]


def resolve_commit(url: str, tag: str) -> str:
    """Return the commit `tag` points at, peeling annotated tags."""
    out = subprocess.run(
        ["git", "ls-remote", url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        check=True, capture_output=True, text=True,
    ).stdout
    refs: dict[str, str] = {}
    for line in out.splitlines():
        if "\t" in line:
            sha, ref = line.split("\t", 1)
            refs[ref] = sha
    sha = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if sha is None:
        raise RuntimeError(f"tag {tag} not found at {url}")
    return sha


def read_current(var: str, contents: str) -> str | None:
    pattern = re.compile(
        rf'^\s*set\(\s*{re.escape(var)}\s+"([^"]+)"',
        re.MULTILINE,
    )
    m = pattern.search(contents)
    return m.group(1) if m else None


def rewrite(var: str, new_tag: str, contents: str) -> str:
    pattern = re.compile(
        rf'(^\s*set\(\s*{re.escape(var)}\s+")[^"]+(")',
        re.MULTILINE,
    )
    replaced, n = pattern.subn(rf'\g<1>{new_tag}\g<2>', contents)
    if n != 1:
        raise RuntimeError(f"could not locate {var} in {DEPS_FILE}")
    return replaced


def update_resolved_date(contents: str) -> str:
    today = _dt.date.today().isoformat()
    return re.sub(
        r"Resolved \d{4}-\d{2}-\d{2}",
        f"Resolved {today}",
        contents,
        count=1,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="rewrite cmake/Dependencies.cmake in place")
    args = p.parse_args()

    contents = DEPS_FILE.read_text()
    changed = False
    new_contents = contents

    moved: list[str] = []

    width = max(len(d.var) for d in DEPS)
    print(f"{'variable'.ljust(width)}  {'current':<10}  -> latest")
    print("-" * (width + 30))

    for dep in DEPS:
        current = read_current(dep.var, contents) or "?"
        pinned = read_current(dep.commit_var, contents)
        latest = latest_stable_tag(dep.url)
        marker = "  (bump)" if latest != current else ""
        if current != "?" and pinned is not None and resolve_commit(dep.url, current) != pinned:
            marker += f"  (tag {current} moved upstream; pin is {pinned[:12]})"
            moved.append(dep.var)
        print(f"{dep.var.ljust(width)}  {current:<10}  -> {latest}{marker}")
        if latest != current:
            new_contents = rewrite(dep.var, latest, new_contents)
            new_contents = rewrite(dep.commit_var, resolve_commit(dep.url, latest), new_contents)
            changed = True

    if moved:
        print("\nWarning: upstream tags no longer match their pinned commits: "
              + ", ".join(moved) + ". The pins were left unchanged.")

    if changed and args.apply:
        new_contents = update_resolved_date(new_contents)
        DEPS_FILE.write_text(new_contents)
        print(f"\nWrote {DEPS_FILE.relative_to(REPO_ROOT)}")
    elif changed:
        print("\nRe-run with --apply to write changes.")
    else:
        print("\nAll dependencies up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
