"""Future CLI. Default invocation is denied."""

from __future__ import annotations

from .authorization import ExecutionNotAuthorized


def main() -> None:
    raise ExecutionNotAuthorized("NOT_RUN")
