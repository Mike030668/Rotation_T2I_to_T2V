"""A–F dispatcher. It raises until a separate release calls the cases."""

from __future__ import annotations

from stage_a_integrated.authorization import ExecutionNotAuthorized
from stage_a_integrated.checkpoints import unpack
from stage_a_integrated.spine import run_synthetic

A_MACROS = {32: 0.50, 64: 0.40, 96: 0.45, 128: 0.47}


def case_a(result: dict, best: bytes, last: bytes) -> None:
    if result["macros"] != A_MACROS or unpack(best)["header"]["completed"] != 64 or unpack(last)["header"]["completed"] != 128:
        raise ValueError("A mismatch")


def case_b(previous: bytes, after: bytes) -> None:
    if unpack(after)["header"]["completed"] != unpack(previous)["header"]["completed"]:
        raise ValueError("B replaced previous BEST")


def case_c(payload: bytes) -> None:
    unpack(payload)


def case_d(left: float, right: float) -> None:
    if abs(left - right) > 1e-6:
        raise ValueError("D macro changed")


def case_e(left: dict, right: dict) -> None:
    for key in ("model", "adamw", "python_rng", "numpy_rng", "torch_rng", "sampler", "exposure"):
        if left[key] != right[key]:
            raise ValueError(f"E {key} mismatch")


def case_f(events: list[int]) -> None:
    if events != [32, 64, 96, 128]:
        raise ValueError("F events are not completed-update boundaries")


def run_all(*_args) -> None:
    raise ExecutionNotAuthorized("A-F NOT_RUN")
