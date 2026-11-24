"""Canonical exposure-v1 checks. Used before a step and before resume."""

from __future__ import annotations

PINNED_SYNTHETIC_LABELS = [0, 2, 9, 10, 40]
REQUIRED = (
    "selected_ordinals",
    "selected_labels",
    "current_ordinal",
    "target_ordinal",
    "current",
    "target",
    "horizon",
    "provider",
)


def _ordinal(value) -> None:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError("ordinal is not a nonnegative int")


def _bounded(ordinals: list, width: int) -> None:
    if not ordinals:
        raise ValueError("observed prefix is empty")
    for value in ordinals:
        _ordinal(value)
        if value >= width:
            raise ValueError("ordinal is outside ids_frames")


def _labels_match(ordinals: list[int], labels: list[int], frame_labels: list[int]) -> bool:
    _bounded(ordinals, len(frame_labels))
    if len(ordinals) != len(labels):
        return False
    return all(frame_labels[ordinal] == label for ordinal, label in zip(ordinals, labels))


def require_item(item: dict) -> None:
    if item.get("provider") != "SYNTHETIC_APPARATUS" or any(key not in item for key in REQUIRED):
        raise ValueError("batch item is not exposure-v1")
    if not _labels_match(item["selected_ordinals"], item["selected_labels"], item["ids_frames"]):
        raise ValueError("selected ordinal does not match the original frame label")
    _bounded([item["current_ordinal"], item["target_ordinal"]], len(item["ids_frames"]))
    ordinals = item["selected_ordinals"]
    if any(type(value) is not int or isinstance(value, bool) or value < 0 for value in ordinals):
        raise ValueError("ordinal is not a nonnegative int")
    if ordinals != sorted(set(ordinals)) or ordinals[-1] != item["current_ordinal"] or item["target_ordinal"] <= item["current_ordinal"]:
        raise ValueError("history is not a strict prefix ending at current")
    if item["current_ordinal"] not in item["selected_ordinals"] or item["target_ordinal"] in item["selected_ordinals"]:
        raise ValueError("observed prefix must contain current and exclude target")
    if item["ids_frames"][item["current_ordinal"]] != item["current"] or item["ids_frames"][item["target_ordinal"]] != item["target"]:
        raise ValueError("current or target ordinal does not match its label")


def receipt(batch: list[dict], completed: int, epoch: int, cursor_before: int, cursor_after: int, seed: int) -> dict:
    for item in batch:
        require_item(item)
    return {
        "schema": "exposure-v1",
        "completed": completed,
        "videos": [item["id_movi"] for item in batch],
        "selected_ordinals": [item["selected_ordinals"] for item in batch],
        "selected_labels": [item["selected_labels"] for item in batch],
        "current_ordinal": [item["current_ordinal"] for item in batch],
        "target_ordinal": [item["target_ordinal"] for item in batch],
        "current": [item["current"] for item in batch],
        "target": [item["target"] for item in batch],
        "ids_frames": [item["ids_frames"] for item in batch],
        "horizon": [item["horizon"] for item in batch],
        "epoch": epoch,
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
        "seed": seed,
        "provider": "SYNTHETIC_APPARATUS",
    }


def replay_ordinals(record: dict) -> None:
    from .stage_a_sampler_v2 import sample

    for video, ordinals, frame_labels, current_ord, target_ord in zip(
        record["videos"],
        record["selected_ordinals"],
        record["ids_frames"],
        record["current_ordinal"],
        record["target_ordinal"],
    ):
        sampled, history = sample(frame_labels, video, record["epoch"], record["seed"])
        if sampled[-2] != current_ord or sampled[-1] != target_ord or list(history[sampled[-2]]) != ordinals:
            raise ValueError("exposure selection does not match the frozen sampler")


def require_record(record: dict, expected_step: int) -> None:
    require_records([dict(record, completed=1)], 1)
    replay_ordinals(record)
    if record.get("completed") != expected_step:
        raise ValueError("proposed exposure step is not the next update")


def require_records(records: list, completed: int) -> None:
    if len(records) != completed:
        raise ValueError("exposure length does not match completed")
    for index, record in enumerate(records, start=1):
        if type(record) is not dict:
            raise ValueError("exposure record is not an object")
        if type(record.get("videos")) is not list:
            raise ValueError("exposure videos is not a list")
        if record.get("schema") != "exposure-v1" or record.get("completed") != index or record.get("provider") != "SYNTHETIC_APPARATUS":
            raise ValueError("exposure schema or step mismatch")
        for key in ("videos", "selected_ordinals", "selected_labels", "current_ordinal", "target_ordinal", "current", "target", "ids_frames", "horizon", "epoch", "cursor_before", "cursor_after", "seed"):
            if key not in record or record[key] in (None, []):
                raise ValueError("exposure record is missing required fields")
        rows = zip(
            record["selected_ordinals"],
            record["selected_labels"],
            record["current_ordinal"],
            record["target_ordinal"],
            record["current"],
            record["target"],
            record["ids_frames"],
        )
        if any(len(row) == 0 for row in (record["selected_ordinals"], record["selected_labels"])):
            raise ValueError("exposure row is empty")
        if len({len(record[key]) for key in ("videos", "selected_ordinals", "selected_labels", "current_ordinal", "target_ordinal", "current", "target", "ids_frames", "horizon")}) != 1:
            raise ValueError("exposure columns have different lengths")
        for ordinals, labels, current_ord, target_ord, current, target, frame_labels in rows:
            if frame_labels != PINNED_SYNTHETIC_LABELS:
                raise ValueError("ids_frames does not match the pinned synthetic source")
            if type(ordinals) is not list or not ordinals:
                raise ValueError("observed prefix is empty")
            _bounded([current_ord, target_ord, *ordinals], len(frame_labels))
            if any(type(value) is not int or isinstance(value, bool) or value < 0 for value in ordinals):
                raise ValueError("ordinal is not a nonnegative int")
            if ordinals != sorted(set(ordinals)) or ordinals[-1] != current_ord or target_ord <= current_ord:
                raise ValueError("history is not a strict prefix ending at current")
            if current_ord not in ordinals or target_ord in ordinals:
                raise ValueError("observed prefix must contain current and exclude target")
            if len(ordinals) != len(labels):
                raise ValueError("selected ordinals and labels have different lengths")
            if any(frame_labels[ordinal] != label for ordinal, label in zip(ordinals, labels)):
                raise ValueError("selected ordinal does not match the pinned source label")
            if frame_labels[current_ord] != current or frame_labels[target_ord] != target:
                raise ValueError("current or target label does not match the pinned source")
        replay_ordinals(record)
