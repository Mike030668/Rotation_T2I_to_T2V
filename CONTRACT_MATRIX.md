# Stage A Invariant Ledger (`CONTRACT_MATRIX.md`)

Date: 2026-11-24
Scope: Stage A synthetic apparatus (`stage_a_integrated`)
Code Commit: `7ed83698014a01007baa480f173a482b69a5d9c7` (tree `ac5a271cd7287d40f72dc521761b0cedcc75eb37`)
Execution Status: `SOURCE_ONLY / NOT_RUN`

> Status definitions (never collapse across states):
> - `NOT_IMPLEMENTED`: contract not yet written in code.
> - `SOURCE_IMPLEMENTED`: written and statically traced in source; **not** yet DS-accepted or executed.
> - `SOURCE_REVIEW_ACCEPTED`: reserved for DS disposition only.
> - `RUNTIME_PASS`: reserved for future separately authorized execution only.

## 1. Package C1 — Exposure, Global Sampler Sequence, and Pre-Mutation Journal Reconciliation

| ID | Originating Requirement | Owning Function & Location | Prior Accepted SHA | Positive & Negative Fixtures (`PREPARED_NOT_RUN`) | Expected Result / Error | Severity | Status | Downstream Dependency |
|---|---|---|---|---|---|---|---|---|
| `C1-01` | Strict typed/size-bounded top-level list, record dict, 9 columns, video strings, nested ordinal/label lists, positive exact-int step, nonnegative exact-int seed/epoch/cursors; strict JSON duplicate-key/nonfinite refusal; explicit legacy schema policy | `strict_json_loads` (`exposure.py:138-163`), `canonical_json_bytes` (`exposure.py:166-179`), `_validate_row_semantics` (`exposure.py:186-224`), `validate_record_structure` (`exposure.py:398-462`), `require_records` (`exposure.py:549-561`), `_admit` (`spine.py:214-247`) | `8007b849`, `f1c0cb80`, `e1a8969e` | `POS-01`, `POS-08`; `NEG-02`, `NEG-12`, `NEG-13`, `NEG-14`, `NEG-15`, `NEG-16`, `NEG-17`, `NEG-19`, `NEG-20` | Rejects non-dict/list/str/exact-int, `bool`, `float`, `NaN`/`Infinity`, duplicate JSON keys, oversized payloads, and `exposure-v1`/untyped legacy sampler with `ValueError` **before** sampler invocation or state mutation | P0/P1 | `SOURCE_IMPLEMENTED` | C2, C4 |
| `C1-02` | Preserve original frozen sparse sampler bytes unchanged (`SAMPLER_SHA256`), distinguish tensor ordinals from `ids_frames` labels, replay observed prefix + `current_ordinal` + `target_ordinal` | `stage_a_sampler_v2.py:6-32`, `verify_sampler_pin` (`plan.py:29-34`), `next_batch` (`plan.py:91-117`), `replay_ordinals` (`exposure.py:327-347`), `validate_record_structure` (`exposure.py:450-472`) | `4523e09e`, `1dc6d48c`, `61f75d9`, `e23850c`, `b6635bf`, `4eccbbf`, `a319c78`, `9ca0667`, `c915639`, `f4808de`, `cbaf403`, `116e7e7` | `POS-01`, `POS-02`, `POS-09`; `NEG-10`, `NEG-11` | `stage_a_sampler_v2.py` SHA256 pinned to `3dcdebeb...`; sampler imported only after pin check; enforces `sampled[-2]==current_ord`, `sampled[-1]==target_ord`, exact prefix & label correspondence | P0 | `SOURCE_IMPLEMENTED` | C2, C3, C4 |
| `C1-03` | Capture actual pre-call `cursor_before`, `cursor_after`, `batch_size`, `epoch`; enforce `0 <= before < after <= 400` and `delta == len(videos) == trusted_batch` with explicit rollover policy | `advance_window` (`plan.py:55-77`), `receipt` (`exposure.py:257-311`), `validate_record_structure` (`exposure.py:432-449`), `run_synthetic` (`spine.py:122-158`) | `f1c0cb80`, `e1a8969e`, `e61c30e9` | `POS-01`, `POS-03`, `POS-06`, `POS-07`; `NEG-01`, `NEG-02`, `NEG-03`, `NEG-04` | Captures pre-call `(step_epoch, cursor_before, expected_cursor_after)` via `advance_window` before `next_batch`; rejects negative, overflow `>400`, or non-matching delta vs trusted `expected_batch` | P0/P1 | `SOURCE_IMPLEMENTED` | C2, C4 |
| `C1-04` | Independently reconstruct ordered `syn-*` IDs (`SYNTHETIC_TRAIN_IDS_SHA256`) and every epoch rollover from `(completed=0, epoch=0, cursor=0)` across the full journal; reject valid per-video sample at wrong cursor | `train_ids` & `verify_train_ids_pin` (`plan.py:37-52`), `select_video_ids` (`plan.py:80-88`), `validate_record_structure` (`exposure.py:467-470`), `require_records` (`exposure.py:562-585`) | `74606337`, `e61c30e9` | `POS-03`, `POS-04`, `POS-05`, `POS-06`, `POS-07`, `POS-08`; `NEG-05`, `NEG-06`, `NEG-07`, `NEG-08`, `NEG-09` | Reconstructs `(step_epoch, exp_before, exp_after)` and `select_video_ids(...)` from `(0, 0)` for steps `1..completed`; rollover at `400 -> (epoch+1, 0)`; rejects good per-video sample placed at wrong global cursor or wrong epoch/seed | P0 | `SOURCE_IMPLEMENTED` | C2, C4 |
| `C1-05` | At `resume`, strictly decode, type, and reconcile `sampler-v2` section, header, full `exposure-v2` journal, seed, source/sampler pins, and external config **before** any model/AdamW/RNG mutation | `build_sampler_section_bytes` (`exposure.py:613-654`), `reconcile_checkpoint_state` (`exposure.py:657-807`), `resume` (`spine.py:257-312`) | `4eccbbf` (partial), `3d9a5bb` (partial) | `POS-05`, `POS-08`; `NEG-17`, `NEG-18`, `NEG-19`, `NEG-20` | `resume` runs `_admit(decoded)` then `reconcile_checkpoint_state(...)` against external `expected_config` BEFORE `model.load_state_dict`, `opt.load_state_dict`, and RNG restoration | P0 | `SOURCE_IMPLEMENTED` | C2, C4 |
| `C1-06` | Freeze bounded canonical pre-step receipt bytes + SHA256 digest (`receipt_sha256`, `exposure_journal_sha256`); document future forward-tensor binding deferred to C3 | `canonical_json_bytes` (`exposure.py:166-179`), `receipt_body_digest` (`exposure.py:182-183`), `build_receipt_bytes` (`exposure.py:314-324`), `require_record` (`exposure.py:475-526`), `run_synthetic` (`spine.py:127-200`) | `b09b3bd6` (partial) | `POS-01`, `POS-05`, `POS-10`; `NEG-16` | Pre-step canonical ASCII JSON bytes (`<=65536` B) with `receipt_sha256` validated pre-step and persisted unchanged into `exposure` section; `forward_binding="DEFERRED_TO_C3"` explicitly declared | P1 | `SOURCE_IMPLEMENTED` | C3, C4 |

## 2. Prior Accepted Narrow Fixes Retained (Regression Checklist)

| Prior Commit | Retained Invariant | Exact Location in `7ed83698` |
|---|---|---|
| `4523e09e` | Exact byte copy of `stage_a_sampler_v2.py` | `src/stage_a_integrated/stage_a_sampler_v2.py:1-56` + `verify_sampler_pin` (`plan.py:29-34`) |
| `1dc6d48c` | Ordinal-to-label conversion (`selected_labels = [labels[i] ...]`) | `plan.py:103-105` |
| `61f75d9` | Separate `selected_ordinals` and `selected_labels` recorded | `plan.py:106-107`, `exposure.py:298-299` |
| `c93ba56` | Complete required exposure fields before step & resume | `exposure.py:41-103`, `407-413`, `688-702` |
| `f203c69a` | Pre-optimizer admission placement | `spine.py:136-150` |
| `3d9a5bb` | Pre-weight-load exposure admission placement on resume | `spine.py:278-297` |
| `b311e559` | Single shared schema validator across writer & resume | `validate_record_structure` (`exposure.py:371-472`) |
| `e23850c8`, `b6635bf` | `current` in prefix, `target` excluded, source label match | `_validate_row_semantics` (`exposure.py:217-224`) |
| `4eccbbf` | Pinned `ids_frames == [0, 2, 9, 10, 40]` enforced | `exposure.py:201-202` |
| `a319c78`, `9ca0667`, `c915639` | Nested lengths, ascending unique prefix ending at `current`, `target > current` | `_validate_row_semantics` (`exposure.py:203-224`) |
| `73767c42`, `892460a6`, `b9aa305e` | Exact-int (`type is int`), bounds `[0, len-1]`, nonempty nested prefix | `plan.py:19-26`, `exposure.py:203-214` |
| `4543d273`, `74606337`, `b09b3bd6` | Pre-step receipt snapshot & exact next positive step (`completed_next`) | `spine.py:126-149`, `exposure.py:410-412` |
| `cbaf4035`, `116e7e77`, `f4808de4` | Schema checks before frozen sampler replay of prefix + `sampled[-2]` + `sampled[-1]` | `exposure.py:327-347`, `398-471` |
| `8007b849`, `f1c0cb80`, `e1a8969e`, `e61c30e9` | Malformed containers, per-record `cursor_before`/`cursor_after`, exact-int bounds `0 <= before < after <= 400` | `exposure.py:398-449`, `549-560` |

## 3. Downstream Packages (Out of C1 Scope — Open Blockers)

| ID | Package | Scope Summary | Current Owner / File | Status |
|---|---|---|---|---|
| `C2-01`..`C2-04` | **C2** | Bounded versioned `checkpoints.unpack` (offsets/limits/strict JSON header), staged pre-mutation deserialization + rollback receipt for model/AdamW/RNG, crash-safe BEST/LAST with directory fsync & prior-BEST retention | `checkpoints.py:13-70`, `spine.py:291-297` | `NOT_IMPLEMENTED` |
| `C3-01`..`C3-03` | **C3** | Historical 400/50/50 split & tensor ABI crosswalk; bind actual forwarded `model_kwargs` content/shape/dtype to pre-step receipt (`forward_binding`); exception-safe `dev.evaluate` train-mode restore; deliberate same-forward leak negative | `samples.py:30-86`, `trainer.py:9-24`, `dev.py:37-61` | `NOT_IMPLEMENTED` |
| `C4-01`..`C4-03` | **C4** | Owner-independent Ed25519 trust anchor, trusted clock/nonce/revocation ledger, full resolved config (`clip`, `horizons_path`), genuinely wired A–F acceptance harness | `authorization.py:16-51`, `tests/stage_a/test_a_f.py:12-49` | `NOT_IMPLEMENTED / PENDING_OWNER` |
