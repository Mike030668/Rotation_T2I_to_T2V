"""BEST/LAST checkpoint. CURRENT_BEST moves only after readback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SECTIONS = ("model", "adamw", "python_rng", "numpy_rng", "torch_rng", "sampler", "exposure")


def pack(header: dict, sections: dict[str, bytes]) -> bytes:
    if set(sections) != set(SECTIONS) or "completed" not in header:
        raise ValueError("checkpoint header is incomplete")
    body = bytearray()
    table = []
    for name in SECTIONS:
        raw = sections[name]
        table.append({"name": name, "offset": len(body), "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        body += raw
    stored = dict(header)
    stored["sections"] = table
    encoded = json.dumps(stored, sort_keys=True).encode("utf-8")
    return b"STCA" + len(encoded).to_bytes(4, "little") + encoded + bytes(body)


def unpack(payload: bytes) -> dict:
    if not payload.startswith(b"STCA"):
        raise ValueError("checkpoint magic mismatch")
    header_len = int.from_bytes(payload[4:8], "little")
    header = json.loads(payload[8 : 8 + header_len].decode("utf-8"))
    body = payload[8 + header_len :]
    if [item["name"] for item in header["sections"]] != list(SECTIONS):
        raise ValueError("checkpoint section set mismatch")
    decoded = {"header": header}
    cursor = 0
    for item in header["sections"]:
        raw = body[cursor : cursor + item["length"]]
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("checkpoint section mismatch")
        decoded[item["name"]] = raw
        cursor += item["length"]
    if cursor != len(body):
        raise ValueError("checkpoint has trailing bytes")
    return decoded


def publish(root: Path, generation_id: str, role: str, payload: bytes) -> None:
    if role not in {"BEST", "LAST"} or "_scored_event_pkg" in str(root):
        raise ValueError("checkpoint path rejected")
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    stage = root / "stage" / f"{generation_id}-{role}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "checkpoint.bin").write_bytes(payload)
    final = generations / f"{generation_id}-{role}"
    os.rename(stage, final)
    if (final / "checkpoint.bin").read_bytes() != payload:
        raise ValueError("checkpoint readback mismatch")
    if role == "BEST":
        partial = root / "CURRENT_BEST.partial"
        partial.write_text(generation_id + "\n", encoding="utf-8")
        os.replace(partial, root / "CURRENT_BEST")
