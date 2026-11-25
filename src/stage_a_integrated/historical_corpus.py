"""Header-only selective 500_1 reader, causal sample/DEV builders, and held-out byte isolation."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import struct

from . import CANDIDATE_SHA256, HORIZONS_SHA256, LOSS_SHA256
from .dev import (
    EXPECTED_DEV_PAIRS,
    EXPECTED_DEV_VIDEOS,
    HORIZON_NAMES,
    PREDECLARED_ABSENT_MID_VIDEOS,
    load_rows,
)
from .exposure import HEX_CHARS, canonical_json_bytes, strict_json_loads
from .forward_binding import (
    FORWARD_BINDING_VERSION,
    REPRESENTATION_FIREWALL,
    WIDTH,
    canonical_forward_kwargs_digest,
    canonical_supervision_digest,
    synthetic_progress_f32,
    synthetic_schedule_f32,
    validate_causal_transition_meta,
    validate_supervision,
)
from .plan import REMAINDER_POLICY, require_exact_int, verify_sampler_pin

HISTORICAL_SOURCE_ID = "HISTORICAL_500_1_SAFETENSORS_V1"
HISTORICAL_PROVIDER_TRAIN = "HISTORICAL_500_1_TRAIN"
HISTORICAL_PROVIDER_DEV = "HISTORICAL_500_1_DEV"
HISTORICAL_TEMPORAL_AXIS_SEMANTICS = "HISTORICAL_STORED_FRAME_LABELS_ORDINAL_ONLY_NO_FPS_NO_PHYSICAL_TIME"
TEXT_MASK_POLICY = "ALL_TRUE_77_LEGACY_NO_STORED_MASK_ASSUMPTION"

SOURCE_PTH_SHA256 = "e75ec2a939b54ea301b42033d7526b4ebdc899edbdb529968ff53ccba5d19fef"
SOURCE_PTH_BYTES = 139_748_154
PORTABLE_MANIFEST_SHA256 = "9d32fdb5c038738cabd570217bb5d82e073e1a4d95fecad247cb202ed28a4a60"
PORTABLE_SHARD_ATTESTED_SHA256 = "6ee8f828e2d8fc0515baae788855a89d1f4950b324c541ec0f1f1bf66a47db0a"
CONVERSION_READBACK_LOG_SHA256 = "898544f4f295fe3a80da503b8a36b63459da3bc52740ff05513ea383f68b6465"
CONVERSION_STAGE1_SHA256 = "f5dfd92f2b6f1811fd3b3f21e7c856fac2e21a0b180bb37c2c16b5d50c4f6af2"
CONVERSION_STAGE2_SHA256 = "ff92826ecd856846efe536aaa53a95377e2768850ae6e5693c22669a3e97279f"

TRAIN_IDS_SHA256 = "d8e41e20c61c1e08470f8e191022f3ba75ad5b978086cd9a2ae922d3f5826276"
DEV_IDS_SHA256 = "8dff4562b28fadfba850ec9daa1198f69b53ad1e9ebf528ef52ec9a40bf48aef"
HELDOUT_IDS_SHA256 = "85d283549ce75a19b9070103ea5217812e735852536a36eebae6954181d98435"

EXPECTED_SPLIT_COUNTS = {"train": 400, "dev": 50, "heldout": 50}
SPLIT_FILENAMES = {
    "train": ("train_ids.txt", TRAIN_IDS_SHA256, 400),
    "dev": ("dev_ids.txt", DEV_IDS_SHA256, 50),
    "heldout": ("final_heldout_ids.txt", HELDOUT_IDS_SHA256, 50),
}
TOTAL_CORPUS_RECORDS = 500
MIN_ELIGIBLE_TRAIN_FRAMES = 6
TEXT_SEQ_LEN = 77
TENSOR_SUFFIXES = ("img", "unclip_embed", "text_embed", "last_hidden_state")
ALLOWED_READER_SPLITS = frozenset({"train", "dev"})

MAX_SPLIT_FILE_BYTES = 1 * 1024 * 1024
MAX_MANIFEST_FILE_BYTES = 8 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
MAX_SINGLE_TENSOR_BYTES = 16 * 1024 * 1024


def _read_bounded_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path.name}")
    size = path.stat().st_size
    if size < 1 or size > max_bytes:
        raise ValueError(f"{label} size out of bounds ({size})")
    raw = path.read_bytes()
    if len(raw) != size or len(raw) > max_bytes:
        raise ValueError(f"{label} read size mismatch")
    return raw


def _require_nonempty_str(value: object, name: str, *, max_len: int = 128) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a trimmed nonempty str")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8") from exc
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds max length {max_len}")
    return value


def _require_sha256_hex(value: object, name: str) -> str:
    text = _require_nonempty_str(value, name, max_len=64)
    if len(text) != 64 or any(ch not in HEX_CHARS for ch in text):
        raise ValueError(f"{name} must be a 64-char lowercase hex sha256")
    return text


def load_verified_split_ids(split_dir: Path | str) -> dict:
    """Load and pin-verify R002 400/50/50 train/dev/heldout ID lists from text files only."""
    base = Path(split_dir)
    if not base.is_dir():
        raise ValueError("split_dir does not exist")

    ordered_lists: dict[str, list[str]] = {}
    id_sets: dict[str, set[str]] = {}
    digests: dict[str, str] = {}

    for split_name, (filename, expected_sha, expected_count) in SPLIT_FILENAMES.items():
        fpath = base / filename
        raw = _read_bounded_bytes(fpath, max_bytes=MAX_SPLIT_FILE_BYTES, label=filename)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha:
            raise ValueError(f"{filename} sha256 mismatch against R002 split pin")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{filename} is not valid UTF-8") from exc
        ids = [line.strip() for line in text.splitlines() if line.strip()]
        if len(ids) != expected_count:
            raise ValueError(f"{filename} count {len(ids)} != {expected_count}")
        if len(set(ids)) != expected_count:
            raise ValueError(f"{filename} contains duplicate video IDs")
        for vid in ids:
            _require_nonempty_str(vid, f"{split_name}.id_movi", max_len=64)
        ordered_lists[split_name] = ids
        id_sets[split_name] = set(ids)
        digests[split_name] = digest

    if not id_sets["train"].isdisjoint(id_sets["dev"]):
        raise ValueError("train/dev split overlap detected in R002 ID lists")
    if not id_sets["train"].isdisjoint(id_sets["heldout"]):
        raise ValueError("train/heldout split overlap detected in R002 ID lists")
    if not id_sets["dev"].isdisjoint(id_sets["heldout"]):
        raise ValueError("dev/heldout split overlap detected in R002 ID lists")

    return {
        "train_ids": ordered_lists["train"],
        "dev_ids": ordered_lists["dev"],
        "heldout_ids": ordered_lists["heldout"],
        "train_set": frozenset(id_sets["train"]),
        "dev_set": frozenset(id_sets["dev"]),
        "heldout_set": frozenset(id_sets["heldout"]),
        "digests": digests,
    }


def load_verified_portable_manifest(manifest_path: Path | str, splits: dict) -> dict:
    """Load portable 500_1 manifest.json, verify SHA-256 and 400/50/50 split partition without touching tensors."""
    path = Path(manifest_path)
    raw = _read_bounded_bytes(path, max_bytes=MAX_MANIFEST_FILE_BYTES, label="manifest.json")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != PORTABLE_MANIFEST_SHA256:
        raise ValueError("manifest.json sha256 mismatch against pinned PORTABLE_MANIFEST_SHA256")

    doc = strict_json_loads(raw, max_bytes=MAX_MANIFEST_FILE_BYTES)
    if type(doc) is not dict or set(doc.keys()) != {"records", "shard", "shard_sha256", "n_records"}:
        raise ValueError("portable manifest does not match expected schema keys")
    if doc["shard"] != "records.safetensors":
        raise ValueError("portable manifest shard filename mismatch")
    if _require_sha256_hex(doc["shard_sha256"], "manifest.shard_sha256") != PORTABLE_SHARD_ATTESTED_SHA256:
        raise ValueError("portable manifest shard_sha256 disagrees with attested Phase 1A pin")
    n_records = require_exact_int(doc["n_records"], "manifest.n_records", minimum=TOTAL_CORPUS_RECORDS, maximum=TOTAL_CORPUS_RECORDS)
    rows = doc["records"]
    if type(rows) is not list or len(rows) != n_records:
        raise ValueError("portable manifest records list length is not 500")

    train_set = splits["train_set"]
    dev_set = splits["dev_set"]
    heldout_set = splits["heldout_set"]
    expected_union = train_set | dev_set | heldout_set
    if len(expected_union) != TOTAL_CORPUS_RECORDS:
        raise ValueError("split union size is not 500")

    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    split_counts = {"train": 0, "dev": 0, "heldout": 0}
    annotated_records: list[dict] = []
    record_by_id: dict[str, dict] = {}
    record_by_key: dict[str, dict] = {}

    for rec in rows:
        if type(rec) is not dict or set(rec.keys()) != {"id_movi", "ids_frames", "key"}:
            raise ValueError("manifest record must contain exact keys {id_movi, ids_frames, key}")
        id_movi = _require_nonempty_str(rec["id_movi"], "record.id_movi", max_len=64)
        key = _require_nonempty_str(rec["key"], "record.key", max_len=32)
        frames = rec["ids_frames"]
        if type(frames) is not list or len(frames) < 2 or any(type(v) is not int for v in frames):
            raise ValueError(f"record {id_movi} ids_frames must be a strictly increasing int list of len >= 2")
        if len(set(frames)) != len(frames) or frames != sorted(frames):
            raise ValueError(f"record {id_movi} ids_frames is not strictly increasing")
        if frames[-1] - frames[0] <= 0:
            raise ValueError(f"record {id_movi} ids_frames span must be positive")
        if id_movi in seen_ids:
            raise ValueError(f"duplicate id_movi in portable manifest: {id_movi}")
        if key in seen_keys:
            raise ValueError(f"duplicate key in portable manifest: {key}")
        seen_ids.add(id_movi)
        seen_keys.add(key)

        in_train = id_movi in train_set
        in_dev = id_movi in dev_set
        in_held = id_movi in heldout_set
        if int(in_train) + int(in_dev) + int(in_held) != 1:
            raise ValueError(f"record {id_movi} split membership invalid")
        split = "train" if in_train else ("dev" if in_dev else "heldout")
        split_counts[split] += 1

        entry = {
            "id_movi": id_movi,
            "ids_frames": list(frames),
            "key": key,
            "split": split,
        }
        annotated_records.append(entry)
        record_by_id[id_movi] = entry
        record_by_key[key] = entry

    if seen_ids != expected_union:
        raise ValueError("portable manifest video IDs disagree with 500-video split union")
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"manifest split counts {split_counts} != {EXPECTED_SPLIT_COUNTS}")

    return {
        "manifest_sha256": manifest_sha256,
        "shard_name": doc["shard"],
        "attested_shard_sha256": doc["shard_sha256"],
        "n_records": n_records,
        "split_counts": split_counts,
        "records": annotated_records,
        "record_by_id": record_by_id,
        "record_by_key": record_by_key,
    }


def _expected_tensor_shape(suffix: str, n_frames: int) -> tuple[int, ...]:
    if suffix == "img":
        return (n_frames, 1, WIDTH)
    if suffix in ("unclip_embed", "text_embed"):
        return (1, WIDTH)
    if suffix == "last_hidden_state":
        return (1, TEXT_SEQ_LEN, WIDTH)
    raise ValueError(f"unsupported tensor suffix: {suffix}")


class SelectiveSafetensorsReader:
    """Header-only plain-file safetensors reader with strict non-overlapping held-out byte seal.

    Never uses mmap, torch.load, or pickle. Reads ONLY the 8-byte header length + bounded JSON
    header at open(), proves disjoint tensor intervals across all 2,000 tensors, and refuses any
    held-out split selection or byte-range overlap prior to seeking or reading tensor payloads.
    """

    def __init__(
        self,
        shard_path: Path | str,
        manifest: dict,
        *,
        allowed_splits: frozenset[str] | set[str],
    ) -> None:
        splits_frozen = frozenset(allowed_splits)
        if not splits_frozen:
            raise ValueError("allowed_splits must be non-empty")
        if "heldout" in splits_frozen or not splits_frozen.issubset(ALLOWED_READER_SPLITS):
            raise PermissionError(f"forbidden split in allowed_splits (held-out is sealed): {sorted(splits_frozen)}")

        self._shard_path = Path(shard_path)
        if not self._shard_path.is_file():
            raise ValueError("records.safetensors file does not exist")
        if self._shard_path.name != manifest["shard_name"]:
            raise ValueError("shard filename disagrees with portable manifest")
        self._file_size = int(self._shard_path.stat().st_size)
        if self._file_size < 16:
            raise ValueError("safetensors file truncated before header")

        self._allowed_splits = splits_frozen
        self._handle = open(self._shard_path, "rb")
        self._closed = False
        try:
            len_bytes = self._handle.read(8)
            if len(len_bytes) != 8:
                raise ValueError("failed to read 8-byte safetensors header length")
            header_len = int.from_bytes(len_bytes, "little")
            if header_len < 2 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"safetensors header length out of bounds: {header_len}")
            data_start = 8 + header_len
            if data_start > self._file_size:
                raise ValueError("safetensors header exceeds file size")
            header_bytes = self._handle.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError("truncated safetensors header")
            self._header_sha256 = hashlib.sha256(header_bytes).hexdigest()
            self._data_start = data_start

            header_doc = strict_json_loads(header_bytes, max_bytes=MAX_SAFETENSORS_HEADER_BYTES)
            if type(header_doc) is not dict:
                raise ValueError("safetensors header must be a JSON object")

            expected_tensor_keys: set[str] = set()
            key_to_record: dict[str, tuple[dict, str]] = {}
            for rec in manifest["records"]:
                for suffix in TENSOR_SUFFIXES:
                    tkey = f"{rec['key']}.{suffix}"
                    expected_tensor_keys.add(tkey)
                    key_to_record[tkey] = (rec, suffix)

            header_tensor_keys = {k for k in header_doc.keys() if k != "__metadata__"}
            if header_tensor_keys != expected_tensor_keys or len(header_tensor_keys) != TOTAL_CORPUS_RECORDS * len(TENSOR_SUFFIXES):
                raise ValueError("safetensors header tensor key set disagrees with manifest 2000 keys")

            intervals: list[tuple[int, int, str]] = []
            tensor_meta: dict[str, dict] = {}
            allowed_keys: set[str] = set()
            forbidden_keys: set[str] = set()
            heldout_abs_ranges: list[tuple[int, int]] = []

            for tkey in sorted(header_tensor_keys):
                entry = header_doc[tkey]
                if type(entry) is not dict or set(entry.keys()) != {"dtype", "shape", "data_offsets"}:
                    raise ValueError(f"malformed safetensors header entry for {tkey}")
                if entry["dtype"] != "F16":
                    raise ValueError(f"expected F16 dtype for {tkey}, got {entry['dtype']!r}")
                rec, suffix = key_to_record[tkey]
                expected_shape = _expected_tensor_shape(suffix, len(rec["ids_frames"]))
                raw_shape = entry["shape"]
                if type(raw_shape) is not list or any(type(d) is not int or d <= 0 for d in raw_shape):
                    raise ValueError(f"invalid shape in safetensors header for {tkey}")
                shape = tuple(raw_shape)
                if shape != expected_shape:
                    raise ValueError(f"shape mismatch for {tkey}: {shape} != {expected_shape}")

                offsets = entry["data_offsets"]
                if type(offsets) is not list or len(offsets) != 2:
                    raise ValueError(f"invalid data_offsets for {tkey}")
                rel_start = require_exact_int(offsets[0], f"{tkey}.start", minimum=0)
                rel_end = require_exact_int(offsets[1], f"{tkey}.end", minimum=rel_start + 1)
                expected_nbytes = math.prod(shape) * 2
                if rel_end - rel_start != expected_nbytes:
                    raise ValueError(f"byte length mismatch for {tkey}: {rel_end - rel_start} != {expected_nbytes}")
                if expected_nbytes > MAX_SINGLE_TENSOR_BYTES:
                    raise ValueError(f"tensor {tkey} exceeds MAX_SINGLE_TENSOR_BYTES")
                abs_start = data_start + rel_start
                abs_end = data_start + rel_end
                if abs_end > self._file_size:
                    raise ValueError(f"tensor {tkey} offsets escape shard file")

                intervals.append((rel_start, rel_end, tkey))
                split = rec["split"]
                tensor_meta[tkey] = {
                    "split": split,
                    "dtype": "F16",
                    "shape": shape,
                    "rel_start": rel_start,
                    "rel_end": rel_end,
                    "abs_start": abs_start,
                    "abs_end": abs_end,
                    "nbytes": expected_nbytes,
                }
                if split in splits_frozen:
                    allowed_keys.add(tkey)
                else:
                    forbidden_keys.add(tkey)
                    if split == "heldout":
                        heldout_abs_ranges.append((abs_start, abs_end))

            intervals.sort(key=lambda item: (item[0], item[1]))
            prev_end = 0
            for rel_start, rel_end, tkey in intervals:
                if rel_start < prev_end:
                    raise ValueError(f"overlapping safetensors tensor byte ranges detected near {tkey}")
                prev_end = rel_end
            if data_start + prev_end != self._file_size:
                raise ValueError("safetensors file length disagrees with tensor data_offsets extent")

            self._tensor_meta = tensor_meta
            self._allowed_keys = frozenset(allowed_keys)
            self._forbidden_keys = frozenset(forbidden_keys)
            self._heldout_abs_ranges = tuple(sorted(heldout_abs_ranges))
            self._opened_keys_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "heldout": set()}
            self._bytes_read_by_split: dict[str, int] = {"train": 0, "dev": 0, "heldout": 0}
            self._read_ops_by_split: dict[str, int] = {"train": 0, "dev": 0, "heldout": 0}
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "SelectiveSafetensorsReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._handle.close()
            self._closed = True

    @property
    def header_sha256(self) -> str:
        return self._header_sha256

    @property
    def shard_file_bytes(self) -> int:
        return self._file_size

    def header_summary(self) -> dict:
        counts = {"train": 0, "dev": 0, "heldout": 0}
        bytes_by_split = {"train": 0, "dev": 0, "heldout": 0}
        for meta in self._tensor_meta.values():
            sp = meta["split"]
            counts[sp] += 1
            bytes_by_split[sp] += meta["nbytes"]
        return {
            "header_sha256": self._header_sha256,
            "data_start": self._data_start,
            "shard_file_bytes": self._file_size,
            "tensor_key_count": len(self._tensor_meta),
            "tensor_keys_by_split": counts,
            "payload_bytes_by_split": bytes_by_split,
            "heldout_intervals_sealed": len(self._heldout_abs_ranges),
            "all_intervals_disjoint": True,
        }

    def read_raw_bytes(self, key: str) -> bytes:
        if self._closed:
            raise ValueError("SelectiveSafetensorsReader is closed")
        if key in self._forbidden_keys or key not in self._allowed_keys:
            raise PermissionError(f"forbidden tensor key read rejected under split seal: {key}")
        meta = self._tensor_meta[key]
        if meta["split"] == "heldout" or meta["split"] not in self._allowed_splits:
            raise PermissionError(f"held-out or unauthorized split key rejected: {key}")

        abs_start = meta["abs_start"]
        abs_end = meta["abs_end"]
        nbytes = meta["nbytes"]
        for h_start, h_end in self._heldout_abs_ranges:
            if abs_start < h_end and h_start < abs_end:
                raise PermissionError(f"byte range for {key} overlaps sealed held-out interval")

        self._handle.seek(abs_start, os.SEEK_SET)
        raw = self._handle.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError(f"short read for tensor {key}: {len(raw)} != {nbytes}")

        sp = meta["split"]
        self._opened_keys_by_split[sp].add(key)
        self._bytes_read_by_split[sp] += nbytes
        self._read_ops_by_split[sp] += 1
        return raw

    def decode_f32_from_raw(self, key: str, raw: bytes):
        import numpy
        import torch

        meta = self._tensor_meta.get(key)
        if meta is None:
            raise KeyError(f"unknown tensor key: {key}")
        if len(raw) != meta["nbytes"]:
            raise ValueError(f"raw byte length mismatch for {key}: {len(raw)} != {meta['nbytes']}")
        arr = numpy.frombuffer(raw, dtype="<f2").reshape(meta["shape"]).copy()
        tensor = torch.from_numpy(arr).to(torch.float32)
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"decoded tensor {key} is not CPU float32")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"decoded tensor {key} contains nonfinite values")
        return tensor

    def read_f32_tensor(self, key: str):
        raw = self.read_raw_bytes(key)
        return self.decode_f32_from_raw(key, raw)

    def read_video_tensors(self, rec: dict) -> tuple:
        if rec["split"] == "heldout" or rec["split"] not in self._allowed_splits:
            raise PermissionError(f"refusing to read tensors for unauthorized split {rec['split']}")
        key_prefix = rec["key"]
        raw_img = self.read_f32_tensor(f"{key_prefix}.img")
        expected_len = len(rec["ids_frames"])
        if tuple(raw_img.shape) != (expected_len, 1, WIDTH):
            raise ValueError(f"{rec['id_movi']} img shape mismatch")
        image_f32 = raw_img[:, 0, :].contiguous()

        unclip_f32 = self.read_f32_tensor(f"{key_prefix}.unclip_embed")
        if tuple(unclip_f32.shape) != (1, WIDTH):
            raise ValueError(f"{rec['id_movi']} unclip_embed shape mismatch")

        text_embed_f32 = self.read_f32_tensor(f"{key_prefix}.text_embed")
        if tuple(text_embed_f32.shape) != (1, WIDTH):
            raise ValueError(f"{rec['id_movi']} text_embed shape mismatch")

        text_hidden_f32 = self.read_f32_tensor(f"{key_prefix}.last_hidden_state")
        if tuple(text_hidden_f32.shape) != (1, TEXT_SEQ_LEN, WIDTH):
            raise ValueError(f"{rec['id_movi']} last_hidden_state shape mismatch")

        return image_f32, unclip_f32, text_embed_f32, text_hidden_f32

    def audit_counters(self) -> dict:
        return {
            "allowed_splits": sorted(self._allowed_splits),
            "heldout_keys_opened": len(self._opened_keys_by_split["heldout"]),
            "heldout_payload_bytes_read": int(self._bytes_read_by_split["heldout"]),
            "full_shard_rehashed": False,
            "full_shard_hash_source": "ATTESTED_PHASE1A_MANIFEST_AND_RECEIPT_ONLY_TO_AVOID_READING_HELDOUT_BYTES",
            "header_sha256": self._header_sha256,
            "shard_file_bytes": self._file_size,
            "opened_key_counts_by_split": {k: len(v) for k, v in self._opened_keys_by_split.items()},
            "bytes_read_by_split": dict(self._bytes_read_by_split),
            "read_ops_by_split": dict(self._read_ops_by_split),
        }


def build_historical_item(
    id_movi: str,
    ids_frames: list[int],
    history_ordinals: list[int],
    current_ordinal: int,
    target_ordinal: int,
    horizon: str,
    *,
    image_f32,
    unclip_f32,
    text_hidden_f32,
) -> dict:
    """Assemble one strictly causal Stage A item from historical float32 tensors on CPU."""
    import torch

    _require_nonempty_str(id_movi, "id_movi", max_len=64)
    if type(ids_frames) is not list or len(ids_frames) < 2:
        raise ValueError("ids_frames must have length >= 2")
    if type(history_ordinals) is not list or not history_ordinals:
        raise ValueError("history_ordinals must be a nonempty list")

    n_frames = len(ids_frames)
    cur_ord = require_exact_int(current_ordinal, "current_ordinal", minimum=0, maximum=n_frames - 2)
    tgt_ord = require_exact_int(target_ordinal, "target_ordinal", minimum=cur_ord + 1, maximum=n_frames - 1)
    hist_ords = [require_exact_int(o, "history_ordinal", minimum=0, maximum=cur_ord) for o in history_ordinals]
    if hist_ords != sorted(set(hist_ords)) or hist_ords[0] != 0 or hist_ords[-1] != cur_ord:
        raise ValueError("history_ordinals must be a causal prefix from 0 ending at current_ordinal")
    if tgt_ord in hist_ords:
        raise ValueError("target_ordinal leaked into history_ordinals")

    history_labels = [ids_frames[o] for o in hist_ords]
    current_label = ids_frames[cur_ord]
    target_label = ids_frames[tgt_ord]
    validate_causal_transition_meta(
        id_movi,
        ids_frames,
        history_labels,
        current_label,
        target_label,
        horizon,
        anchor_ordinal=0,
        anchor=ids_frames[0],
    )

    for name, tensor, expected_shape in (
        ("image_f32", image_f32, (n_frames, WIDTH)),
        ("unclip_f32", unclip_f32, (1, WIDTH)),
        ("text_hidden_f32", text_hidden_f32, (1, TEXT_SEQ_LEN, WIDTH)),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != expected_shape
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"{name} must be a finite CPU float32 tensor with shape {expected_shape}")

    image_c = image_f32.detach().contiguous()
    unclip_c = unclip_f32.detach().contiguous()
    text_c = text_hidden_f32.detach().contiguous()

    # Strict representation firewall: only indices <= current_ordinal enter model_kwargs.
    e0 = image_c[0:1]
    e_hist = image_c[hist_ords]
    e_cur = image_c[cur_ord : cur_ord + 1]

    proxy_hist = unclip_c + (e_hist - e0)
    proxy_cur = unclip_c + (e_cur - e0)

    kwargs = {
        "history_latents": proxy_hist.unsqueeze(0).clone(),
        "history_progress": torch.tensor([synthetic_progress_f32(ids_frames, history_labels)], dtype=torch.float32),
        "history_valid_mask": torch.ones(1, len(hist_ords), dtype=torch.bool),
        "text_hidden_states": text_c.clone(),
        "text_valid_mask": torch.ones(1, TEXT_SEQ_LEN, dtype=torch.bool),
        "current_latent": proxy_cur.clone(),
        "anchor_latent": unclip_c.clone(),
        "transition": torch.tensor([synthetic_schedule_f32(ids_frames, current_label, target_label)], dtype=torch.float32),
        "mode": "hybrid",
    }

    # Future frame e_target enters supervision ONLY, never model_kwargs.
    e_target = image_c[tgt_ord : tgt_ord + 1]
    supervision = {
        "target_delta_local": (e_target - e_cur).clone(),
        "target_delta_anchor": (e_target - e0).clone(),
        "target_latent_proxy": (unclip_c + (e_target - e0)).clone(),
    }

    forward_input_sha256, frozen_kwargs = canonical_forward_kwargs_digest(kwargs)
    validate_supervision(supervision, forbidden_kwargs=kwargs, frozen_kwargs=frozen_kwargs)
    supervision_sha256 = canonical_supervision_digest(supervision)

    return {
        "id_movi": id_movi,
        "ids_frames": list(ids_frames),
        "anchor_ordinal": 0,
        "anchor": ids_frames[0],
        "selected_ordinals": hist_ords,
        "selected_labels": history_labels,
        "history": history_labels,
        "current_ordinal": cur_ord,
        "target_ordinal": tgt_ord,
        "current": current_label,
        "target": target_label,
        "horizon": horizon,
        "forward_binding": FORWARD_BINDING_VERSION,
        "representation_firewall": REPRESENTATION_FIREWALL,
        "temporal_axis_semantics": HISTORICAL_TEMPORAL_AXIS_SEMANTICS,
        "text_mask_policy": TEXT_MASK_POLICY,
        "forward_input_sha256": forward_input_sha256,
        "supervision_sha256": supervision_sha256,
        "model_kwargs": kwargs,
        "supervision": supervision,
    }


def assert_historical_no_target_leak(
    item: dict,
    *,
    image_f32,
    unclip_f32,
    text_hidden_f32,
) -> dict:
    """Verify counterfactual target-frame mutation leaves model_kwargs digest unchanged and alters supervision."""
    import torch

    actual_digest, frozen_kwargs = canonical_forward_kwargs_digest(item["model_kwargs"])
    if item.get("forward_input_sha256") != actual_digest:
        raise ValueError("historical item forward_input_sha256 disagrees with model_kwargs")
    validate_supervision(item["supervision"], forbidden_kwargs=item["model_kwargs"], frozen_kwargs=frozen_kwargs)

    mutated = image_f32.detach().clone()
    mutated[item["target_ordinal"]] += 1.0
    dirty = build_historical_item(
        item["id_movi"],
        item["ids_frames"],
        item["selected_ordinals"],
        item["current_ordinal"],
        item["target_ordinal"],
        item["horizon"],
        image_f32=mutated,
        unclip_f32=unclip_f32,
        text_hidden_f32=text_hidden_f32,
    )
    if dirty["forward_input_sha256"] != actual_digest:
        raise ValueError("target-frame mutation altered historical forward_input_sha256")
    for key, val in item["model_kwargs"].items():
        other = dirty["model_kwargs"][key]
        if hasattr(val, "equal"):
            if not bool(torch.equal(val, other)):
                raise ValueError(f"target-frame mutation altered model_kwargs[{key}]")
        elif val != other:
            raise ValueError(f"target-frame mutation altered model_kwargs[{key}]")
    for sup_key in ("target_delta_local", "target_delta_anchor", "target_latent_proxy"):
        if bool(torch.equal(item["supervision"][sup_key], dirty["supervision"][sup_key])):
            raise ValueError(f"target-frame mutation did not alter supervision[{sup_key}]")
    return {
        "forward_input_sha256": actual_digest,
        "clean_supervision_sha256": item["supervision_sha256"],
        "dirty_supervision_sha256": dirty["supervision_sha256"],
        "target_mutation_preserved_digest": True,
    }


def eligible_train_records(manifest: dict, *, min_frames: int = MIN_ELIGIBLE_TRAIN_FRAMES) -> tuple[list[dict], list[dict], str]:
    """Partition 400 train records into length >= min_frames eligible pool vs < min_frames ineligible without silent alteration."""
    min_req = require_exact_int(min_frames, "min_frames", minimum=2, maximum=1000)
    train_recs = sorted((r for r in manifest["records"] if r["split"] == "train"), key=lambda r: r["id_movi"])
    eligible = [r for r in train_recs if len(r["ids_frames"]) >= min_req]
    ineligible = [r for r in train_recs if len(r["ids_frames"]) < min_req]
    if len(train_recs) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError("train split count in manifest is not 400")
    eligible_ids = [r["id_movi"] for r in eligible]
    eligible_ids_sha256 = hashlib.sha256(canonical_json_bytes(eligible_ids, max_bytes=1_048_576)).hexdigest()
    return eligible, ineligible, eligible_ids_sha256


def advance_historical_window(
    epoch: int,
    cursor: int,
    batch_size: int,
    eligible_size: int,
) -> tuple[int, int, int]:
    """Deterministic rollover_before_overflow batch window over actual eligible train count."""
    epoch_val = require_exact_int(epoch, "epoch", minimum=0)
    size_val = require_exact_int(eligible_size, "eligible_size", minimum=1, maximum=EXPECTED_SPLIT_COUNTS["train"])
    batch_val = require_exact_int(batch_size, "batch_size", minimum=1, maximum=size_val)
    cursor_val = require_exact_int(cursor, "cursor", minimum=0, maximum=size_val)
    if cursor_val + batch_val > size_val:
        next_epoch = epoch_val + 1
        cursor_before = 0
    else:
        next_epoch = epoch_val
        cursor_before = cursor_val
    cursor_after = cursor_before + batch_val
    return next_epoch, cursor_before, cursor_after


def next_historical_train_batch(
    reader: SelectiveSafetensorsReader,
    eligible_records: list[dict],
    *,
    eligible_ids_sha256: str,
    seed: int,
    epoch: int,
    cursor_before: int,
    batch_size: int,
    tensor_cache: dict[str, tuple] | None = None,
) -> tuple[list[dict], int, list[str]]:
    """Assemble one authorized historical train batch using pinned stage_a_sampler_v2 semantics."""
    sampler_digest = verify_sampler_pin()
    from .stage_a_sampler_v2 import sample

    n_elig = len(eligible_records)
    if n_elig < 1:
        raise ValueError("eligible_records cannot be empty")
    batch_val = require_exact_int(batch_size, "batch_size", minimum=1, maximum=n_elig)
    before_val = require_exact_int(cursor_before, "cursor_before", minimum=0, maximum=n_elig - 1)
    if before_val + batch_val > n_elig:
        raise ValueError("cursor window exceeds eligible train pool size")
    seed_val = require_exact_int(seed, "seed", minimum=0)
    epoch_val = require_exact_int(epoch, "epoch", minimum=0)

    selected_recs = [eligible_records[(seed_val + epoch_val + before_val + off) % n_elig] for off in range(batch_val)]
    items: list[dict] = []
    chosen_ids: list[str] = []
    for rec in selected_recs:
        vid = rec["id_movi"]
        if rec["split"] != "train":
            raise PermissionError(f"non-train record requested in train batch: {vid} ({rec['split']})")
        frames = rec["ids_frames"]
        ordinals, history = sample(frames, vid, epoch_val, seed_val)
        cur_ord, tgt_ord = int(ordinals[-2]), int(ordinals[-1])
        hist_ords = [int(x) for x in history[cur_ord]]

        if tensor_cache is not None and vid in tensor_cache:
            image_f32, unclip_f32, _, text_hidden_f32 = tensor_cache[vid]
        else:
            tensors = reader.read_video_tensors(rec)
            if tensor_cache is not None:
                tensor_cache[vid] = tensors
            image_f32, unclip_f32, _, text_hidden_f32 = tensors

        item = build_historical_item(
            vid,
            frames,
            hist_ords,
            cur_ord,
            tgt_ord,
            "near",
            image_f32=image_f32,
            unclip_f32=unclip_f32,
            text_hidden_f32=text_hidden_f32,
        )
        item["epoch"] = epoch_val
        item["seed"] = seed_val
        item["provider"] = HISTORICAL_PROVIDER_TRAIN
        item["source_id"] = HISTORICAL_SOURCE_ID
        item["source_sha256"] = eligible_ids_sha256
        item["sampler_sha256"] = sampler_digest
        item["candidate_sha256"] = CANDIDATE_SHA256
        item["loss_sha256"] = LOSS_SHA256
        item["remainder_policy"] = REMAINDER_POLICY
        items.append(item)
        chosen_ids.append(vid)
    return items, before_val + batch_val, chosen_ids


def build_historical_dev_items(
    reader: SelectiveSafetensorsReader,
    manifest: dict,
    horizons_path: Path | str,
    *,
    verify_target_leak_invariance: bool = False,
) -> dict:
    """Build exact 147 frozen DEV items over 50 DEV videos from SelectiveSafetensorsReader (ZERO train/heldout read)."""
    rows = load_rows(Path(horizons_path))
    if len(rows) != EXPECTED_DEV_VIDEOS:
        raise ValueError("frozen dev manifest is not 50 videos")

    by_id = manifest["record_by_id"]
    items: list[dict] = []
    expected_keys: set[tuple] = set()
    expected_digests: dict[tuple, str] = {}
    invariance_checked = 0

    for row in rows:
        vid = row["id_movi"]
        rec = by_id.get(vid)
        if rec is None:
            raise ValueError(f"DEV manifest video {vid} missing from corpus manifest")
        if rec["split"] != "dev":
            raise PermissionError(f"DEV manifest video {vid} belongs to split {rec['split']} != dev")
        n_frames = int(row["n"])
        if len(rec["ids_frames"]) != n_frames:
            raise ValueError(f"DEV video {vid} frame count {len(rec['ids_frames'])} != manifest n={n_frames}")
        if rec["ids_frames"] != list(range(n_frames)):
            raise ValueError(
                f"DEV video {vid} stored ids_frames differs from ordinal 0..n-1: {rec['ids_frames'][:6]}"
            )

        image_f32, unclip_f32, _, text_hidden_f32 = reader.read_video_tensors(rec)
        hist_ords = [int(x) for x in row["history"]]
        cur_ord = int(row["current"])

        for horizon in HORIZON_NAMES:
            raw_tgt = row.get(horizon)
            if raw_tgt is None:
                continue
            tgt_ord = int(raw_tgt)
            item = build_historical_item(
                vid,
                rec["ids_frames"],
                hist_ords,
                cur_ord,
                tgt_ord,
                horizon,
                image_f32=image_f32,
                unclip_f32=unclip_f32,
                text_hidden_f32=text_hidden_f32,
            )
            item["provider"] = HISTORICAL_PROVIDER_DEV
            if verify_target_leak_invariance:
                assert_historical_no_target_leak(
                    item,
                    image_f32=image_f32,
                    unclip_f32=unclip_f32,
                    text_hidden_f32=text_hidden_f32,
                )
                invariance_checked += 1
            key = (vid, cur_ord, tgt_ord, horizon)
            if key in expected_keys:
                raise ValueError(f"duplicate DEV key encountered: {key}")
            expected_keys.add(key)
            expected_digests[key] = item["forward_input_sha256"]
            items.append(item)

    if len(expected_keys) != EXPECTED_DEV_PAIRS or len(items) != EXPECTED_DEV_PAIRS:
        raise ValueError(f"historical DEV support mismatch: {len(items)} != {EXPECTED_DEV_PAIRS}")

    sorted_digests = "".join(expected_digests[k] for k in sorted(expected_digests))
    dev_forward_digests_sha256 = hashlib.sha256(sorted_digests.encode("ascii")).hexdigest()
    return {
        "rows": rows,
        "items": items,
        "expected_keys": expected_keys,
        "expected_digests": expected_digests,
        "dev_forward_digests_sha256": dev_forward_digests_sha256,
        "horizons_sha256": HORIZONS_SHA256,
        "predeclared_absent_mid_videos": sorted(PREDECLARED_ABSENT_MID_VIDEOS),
        "invariance_checked_pairs": invariance_checked,
    }


def _pack_hash_array_f32(tensor) -> str:
    """Canonical binary32 little-endian SHA-256 of a finite CPU float32 tensor."""
    flat = [float(v) for v in tensor.reshape(-1).tolist()]
    if any(not math.isfinite(v) for v in flat):
        raise ValueError("nonfinite value during tensor digest computation")
    packed = struct.pack(f"<{len(flat)}f", *(0.0 if v == 0.0 else v for v in flat))
    return hashlib.sha256(packed).hexdigest()


def _pack_raw_bytes_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def summarize_tensor_mask_and_digests(rec: dict, reader: SelectiveSafetensorsReader) -> dict:
    """Read authorized train/dev record tensors via single-pass SelectiveSafetensorsReader for ABI audit ONLY."""
    import torch

    if rec["split"] == "heldout":
        raise PermissionError("refusing heldout tensor summary")
    prefix = rec["key"]
    img_raw = reader.read_raw_bytes(f"{prefix}.img")
    unclip_raw = reader.read_raw_bytes(f"{prefix}.unclip_embed")
    text_emb_raw = reader.read_raw_bytes(f"{prefix}.text_embed")
    text_hid_raw = reader.read_raw_bytes(f"{prefix}.last_hidden_state")

    raw_img = reader.decode_f32_from_raw(f"{prefix}.img", img_raw)
    if tuple(raw_img.shape) != (len(rec["ids_frames"]), 1, WIDTH):
        raise ValueError(f"{rec['id_movi']} img shape mismatch")
    image_f32 = raw_img[:, 0, :].contiguous()
    unclip_f32 = reader.decode_f32_from_raw(f"{prefix}.unclip_embed", unclip_raw)
    text_embed_f32 = reader.decode_f32_from_raw(f"{prefix}.text_embed", text_emb_raw)
    text_hidden_f32 = reader.decode_f32_from_raw(f"{prefix}.last_hidden_state", text_hid_raw)
    if tuple(unclip_f32.shape) != (1, WIDTH) or tuple(text_embed_f32.shape) != (1, WIDTH) or tuple(text_hidden_f32.shape) != (1, TEXT_SEQ_LEN, WIDTH):
        raise ValueError(f"{rec['id_movi']} embedding shape mismatch")

    zero_row_count = int((text_hidden_f32[0].abs().sum(dim=-1) == 0.0).sum().item())
    trailing_equal_rows = 0
    last_row = text_hidden_f32[0, -1]
    for idx in range(TEXT_SEQ_LEN - 2, -1, -1):
        if bool(torch.equal(text_hidden_f32[0, idx], last_row)):
            trailing_equal_rows += 1
        else:
            break

    img_raw_sha = _pack_raw_bytes_digest(img_raw)
    unclip_raw_sha = _pack_raw_bytes_digest(unclip_raw)
    text_emb_raw_sha = _pack_raw_bytes_digest(text_emb_raw)
    text_hid_raw_sha = _pack_raw_bytes_digest(text_hid_raw)

    rev_img_sha = _pack_hash_array_f32(image_f32.flip(0).contiguous())
    fwd_img_sha = _pack_hash_array_f32(image_f32)
    first_frame_sha = _pack_hash_array_f32(image_f32[0:1])
    last_frame_sha = _pack_hash_array_f32(image_f32[-1:])

    record_payload_sha = hashlib.sha256(
        "|".join((img_raw_sha, unclip_raw_sha, text_emb_raw_sha, text_hid_raw_sha)).encode("ascii")
    ).hexdigest()

    return {
        "id_movi": rec["id_movi"],
        "split": rec["split"],
        "key": rec["key"],
        "n_frames": len(rec["ids_frames"]),
        "ids_frames_is_ordinal": rec["ids_frames"] == list(range(len(rec["ids_frames"]))),
        "record_payload_sha256": record_payload_sha,
        "fwd_img_f32_sha256": fwd_img_sha,
        "rev_img_f32_sha256": rev_img_sha,
        "first_frame_sha256": first_frame_sha,
        "last_frame_sha256": last_frame_sha,
        "unclip_raw_sha256": unclip_raw_sha,
        "text_embed_raw_sha256": text_emb_raw_sha,
        "text_hidden_raw_sha256": text_hid_raw_sha,
        "text_zero_token_rows": zero_row_count,
        "text_trailing_repeated_rows": trailing_equal_rows,
        "image_f32": image_f32,
        "unclip_f32": unclip_f32,
        "text_hidden_f32": text_hidden_f32,
    }
