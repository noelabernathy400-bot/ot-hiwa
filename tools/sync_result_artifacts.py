"""Commit and push only explicitly named experiment artifacts.

This helper intentionally refuses to run when another staged change exists.
It lets experiment runners publish their JSON and figures without absorbing
unrelated work from a shared research checkout.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _has_staged_diff() -> bool:
    return subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=ROOT,
        check=False,
    ).returncode != 0


def sync(paths: list[Path], message: str) -> None:
    staged = _git("diff", "--cached", "--name-only").stdout.strip()
    if staged:
        raise RuntimeError(
            "refusing automatic result sync because other staged changes exist:\n"
            f"{staged}"
        )
    relative_paths = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file() or ROOT not in resolved.parents:
            raise ValueError(f"artifact must be an existing file inside the repository: {path}")
        relative_paths.append(str(resolved.relative_to(ROOT)))
    _git("add", "--", *relative_paths)
    if not _has_staged_diff():
        return
    _git("commit", "-m", message)
    branch = _git("branch", "--show-current").stdout.strip()
    _git("push", "origin", branch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--message", default="record experiment results")
    args = parser.parse_args()
    sync(args.paths, args.message)


if __name__ == "__main__":
    main()
