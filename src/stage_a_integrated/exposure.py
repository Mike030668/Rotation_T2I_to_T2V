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


def _labels_match(ordinals: list[int], labels: list[int], frame_labels: list[int]) -> bool:
    if len(ordinals) != len(labels):
        return False
    return all(frame_labels[ordinal] == label for ordinal, label in zip(ordinals, labels))


def require_item(item: dict) -> None:
    if item.get("provider") != "SYNTHETIC_APPARATUS" or any(key not in item for key in REQUIRED):
        raise ValueError("batch item is not exposure-v1")
    if not _labels_match(item["selected_ordinals"], item["selected_labels"], item["ids_frames"]):
        raise ValueError("selected ordinal does not match the original frame label")
    if item["current_ordinal"] not in item["selected_ordinals"] or item["target_ordinal"] in item["selected_ordinals"]:
        raise ValueError("observed prefix must contain current and exclude target")
    if item["ids_frames"][item["current_ordinal"]] != item["current"] or item["ids_frames"][item["target_ordinal"]] != item["target"]:
        raise ValueError("current or target ordinal does not match its label")


def require_records(records: list, completed: int) -> None:
    if len(records) != completed:
        raise ValueError("exposure length does not match completed")
    for index, record in enumerate(records, start=1):
        if record.get("schema") != "exposure-v1" or record.get("completed") != index or record.get("provider") != "SYNTHETIC_APPARATUS":
            raise ValueError("exposure schema or step mismatch")
        for key in ("videos", "selected_ordinals", "selected_labels", "current_ordinal", "target_ordinal", "current", "target", "ids_frames", "horizon", "epoch", "cursor", "seed"):
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
            if current_ord not in ordinals or target_ord in ordinals:
                raise ValueError("observed prefix must contain current and exclude target")
            if len(ordinals) != len(labels):
                raise ValueError("selected ordinals and labels have different lengths")
            if any(frame_labels[ordinal] != label for ordinal, label in zip(ordinals, labels)):
                raise ValueError("selected ordinal does not match the pinned source label")
            if frame_labels[current_ord] != current or frame_labels[target_ord] != target:
                raise ValueError("current or target label does not match the pinned source")
