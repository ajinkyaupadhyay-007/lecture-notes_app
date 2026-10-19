"""Evaluation package — public API."""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Lazy entry point to avoid double-loading run module during CLI execution."""
    from lecturenotes.evaluation.run import main as _main

    _main(argv)
