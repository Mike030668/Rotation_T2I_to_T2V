# Static Admission, Transaction, and Rollback Argument (`STATIC_ADMISSION_ARGUMENT.md`)

**Branch:** `local/stage-a-git-first`  
**C1 Accepted Baseline:** Doc commit `7bb1c73f0c4bd252aa9dd42d5604e69ed7fe3bb6` / Code commit `7ed83698014a01007baa480f173a482b69a5d9c7`  
**C2 Code Commits:** `19e034d2a5ef07443f9a9626c127b97ef48d23e0` -> `6622078391736777edc62ea4a61d33c430bd3dc1`  
**Execution Status:** `SOURCE_ONLY / NO_IMPORT / NO_COMPILE / NO_TEST / NO_EXECUTION`

---

## 1. Complete C1 + C2 Call Graphs

### 1.1 Writer & Durable Publish Call Graph (`spine.py:run_synthetic:189-417`)

```text
run_synthetic(release_path, trust_anchor_path, expected, config)
  ├─> authorization.load(release_path, trust_anchor_path, expected)            [spine.py:190] (C4 trust root remains PENDING_OWNER)
  ├─> _validate_run_config(config)                                             [spine.py:66-104,193]
  ├─> verify_sampler_pin() & verify_train_ids_pin(train_ids())                 [spine.py:196-197]
  ├─> file_sha256(horizons_path, max_bytes=16777216)                           [spine.py:200; checkpoints.py:86-105]
  ├─> _canonical_effective_config_bytes(...)                                   [spine.py:107-140,201-208]
  ├─> build_expected_provenance(effective_config_bytes, horizons_path, None)   [spine.py:209-214; checkpoints.py:136-173]
  ├─> [if resume_bytes or resume_role:] pre-construction preflight BEFORE model/opt init:
  │     ├─ resolve_pointer(root, role) if resume_role                          [spine.py:233-235; checkpoints.py:638-694]
  │     ├─ _preflight_and_reconcile(resume_payload, expected_resume_cfg)       [spine.py:143-186,241]
  │     ├─ enforce completed <= max_updates                                    [spine.py:242-243]
  │     └─ on any exception -> write_failure_receipt(root, "resume_preflight", ...) -> re-raise
  ├─> model = model_cls(config_cls()); opt = torch.optim.AdamW(...)            [spine.py:256-258]
  ├─> [if resume_payload is not None:] resume(..., failure_root=root)          [spine.py:270-286,500-584]
  └─> while completed < max_updates:                                           [spine.py:288-409]
        ├─ step_epoch, cursor_before, expected_cursor_after = advance_window(epoch, cursor, batch_size)
        ├─ batch, cursor_after, chosen = next_batch(seed, step_epoch, cursor_before, batch_size)
        ├─ proposed_bytes = build_receipt_bytes(...)                           [exposure-v2 canonical bytes]
        ├─ record, frozen_bytes = require_record(proposed_bytes, ...)          [full C1 validation BEFORE optimizer_step]
        ├─ step_inc = optimizer_step(model, batch, opt, clip)                  [trainer.py:9-24; forward_binding DEFERRED_TO_C3]
        ├─ state advance & exposure_frames.append(frozen_bytes)
        └─ if completed in dev_events:                                         [spine.py:326-409]
              ├─ try: value = evaluate(model, items, keys)
              │  except: write_failure_receipt(root, "dev_eval_failure", ...); raise
              │  finally: model.train()                                        [guarantees train-mode restore]
              ├─ if nonfinite/bool(value):
              │     write_failure_receipt(root, "dev_nonfinite", ...); raise ValueError
              ├─ update macros[completed], best, events
              ├─ step_provenance = build_expected_provenance(..., generation_id=f"u{completed}")
              ├─ payload = pack(header_with_prov_v1, _sections(...))           [stca-v2; checkpoints.py:261-297]
              └─ try:
                    publish(root, f"u{completed}", "LAST", payload)            [CURRENT_LAST]
                    if best[0] == completed: publish(root, f"u{completed}", "BEST", payload)
                 except: write_failure_receipt(root, "checkpoint_publish", ...); raise
```

### 1.2 2-Phase All-State Resume & Rollback Call Graph (`spine.py:resume:500-584`)

```text
resume(payload, model, opt, expected_config, *, failure_root=None)
  │
  ├─ PHASE 0 — ENVELOPE & EXPOSURE PREFLIGHT (_preflight_and_reconcile, zero side effects)
  │   ├─ checkpoints.unpack(payload)                                           [checkpoints.py:300-370]
  │   ├─ spine._admit(decoded)                                                 [spine.py:420-453]
  │   ├─ verify_checkpoint_provenance / verify_baseline_provenance             [checkpoints.py:176-258]
  │   ├─ exposure.reconcile_checkpoint_state(...)                              [exposure.py:657-807]
  │   ├─ validate_python_rng_bytes / validate_numpy_rng_bytes / validate_torch_rng_bytes [checkpoints.py:373-423]
  │
  ├─ PHASE 1 — ALL-STATE STAGING (pure local copies; zero mutation of live model/opt/RNGs)
  │   ├─ staged_model = stage_model_state(decoded["model"], model)             [checkpoints.py:448-469]
  │   ├─ staged_adamw = stage_adamw_state(decoded["adamw"], opt, completed=N)  [checkpoints.py:521-563]
  │   ├─ staged_py_rng = stage_python_rng(decoded["python_rng"])               [checkpoints.py:426-427]
  │   ├─ staged_np_rng = stage_numpy_rng(decoded["numpy_rng"])                 [checkpoints.py:430-434]
  │   └─ staged_torch_rng = stage_torch_rng(decoded["torch_rng"])              [checkpoints.py:437-445; isolated Generator.set_state probe]
  │       └─ ANY failure in Phase 0 or 1:
  │             [if failure_root:] write_failure_receipt(..., "resume_stage", ...)
  │             raise (live model, live opt, and global Python/NumPy/Torch RNGs are 100% untouched)
  │
  ├─ SNAPSHOT CAPTURE (detached tensor clones + optimizer deepcopy + RNG copies)
  │   └─ snap_model, snap_adamw, snap_py_rng, snap_np_rng, snap_torch_rng      [spine.py:537-542]
  │
  └─ PHASE 2 — TRANSACTIONAL APPLY & ROLLBACK
      ├─ model.load_state_dict(staged_model, strict=True)
      ├─ opt.load_state_dict(staged_adamw)
      ├─ random.setstate(staged_py_rng)
      ├─ numpy.random.set_state(staged_np_rng)
      ├─ torch.set_rng_state(staged_torch_rng)
      └─ on ANY exception:
            _rollback_live_state(...) restores all 5 live snapshots            [spine.py:470-497]
            finally: [if failure_root:] write_failure_receipt(..., "resume_apply", ...)
            re-raise original exception
```

### 1.3 Crash-Safe Generation Publish & Pointer Protocol (`checkpoints.py:publish:766-894`)

1. Validate `role in {"BEST", "LAST"}`, reject `"_scored_event_pkg" in str(root)`, validate canonical `generation_id == f"u{step}"`.
2. Verify `payload` via `unpack(payload)`, check `header["completed"] == step` and `verify_baseline_provenance(header)["generation_id"] == generation_id`.
3. Create `generations/` and `stage/`, `_fsync_dir(root)`, verify `generations.stat().st_dev == stage_root.stat().st_dev` (same-filesystem guarantee).
4. **Immutable generation enforcement:** if `generations / f"{generation_id}-{role}"` already exists, refuse overwrite (`ValueError`).
5. Read existing `CURRENT_{role}` (`_parse_pointer_bytes`) if present to extract `previous_generation_id` and `previous_checkpoint_sha256` without mutating any prior generation directory.
6. Build canonical `ckpt-manifest-v1` JSON binding `checkpoint_bytes`, `checkpoint_sha256`, `header_sha256`, `provenance_sha256`, and `provenance`.
7. Allocate fresh crash-recovery-safe temporary directory `stage / f"{generation_id}-{role}.tmp-{seq:04d}"`, `_fsync_dir(stage_root)`, write `checkpoint.bin` (`_fsync_file`), write `manifest.json` (`_fsync_file`), `_fsync_dir(stage)`.
8. **Pre-rename verification:** read back `checkpoint.bin` (exact bytes + SHA256), read back `manifest.json`, and run full `unpack(staged_readback)`.
9. **Atomic publish:** `os.rename(stage, final)`, `_fsync_dir(generations)`, `_fsync_dir(stage_root)`.
10. **Post-rename verification:** read back `(final / "checkpoint.bin")` and `(final / "manifest.json")` before touching any pointer.
11. **Atomic pointer commit (`ptr-v1`):** write `root / f"CURRENT_{role}.partial"`, `_fsync_file`, read back, `os.replace(partial, root / f"CURRENT_{role}")`, `_fsync_dir(root)`, read back.

---

## 2. Trust Boundary: Externally Authoritative vs Checkpoint-Internal Data (C1 + C2)

| Surface | Authority Class | Source / Binding | Security Limit Under C2 |
|---|---|---|---|
| `expected_seed`, `expected_batch`, `effective_config_sha256` | **Externally supplied caller config** | Validated in `_validate_run_config`, hashed via `_canonical_effective_config_bytes`, bound into `prov-v1` | Reconciles resumed checkpoints with effective config; full CLI config & Ed25519 root await **C4** (`PENDING_OWNER_C4`) |
| `SYNTHETIC_SOURCE_ID`, `SYNTHETIC_TRAIN_IDS_SHA256`, `CANDIDATE_SHA256`, `LOSS_SHA256`, `HORIZONS_SHA256`, `SAMPLER_SHA256` | **Code-pinned synthetic constants** | Bound in `prov-v1` (`checkpoints.py:152-173`) and runtime `file_sha256(horizons_path)` / `verify_*_pin()` | Certifies the synthetic 400-ID apparatus only; does **not** certify `500_1` historical tensor ABI (`C3`) |
| `header`, `sections`, `sampler`, `exposure`, `provenance` | **Untrusted serialized bytes until reconciled** | `unpack` (`stca-v2`) + `_admit` + `verify_checkpoint_provenance` (`prov-v1`) + `reconcile_checkpoint_state` (`sampler-v2`/`exposure-v2`) | Self-consistency inside checkpoint is necessary, never sufficient; all metadata checked against external pins and global replay before any state staging |
| `model`, `adamw`, `python_rng`, `numpy_rng`, `torch_rng` bytes | **Untrusted serialized state until Phase 1 & Phase 2** | Pure structural validators + isolated Torch `Generator` probe + shape/dtype/finiteness checks + 5-way live snapshot rollback | Late corruption in `torch_rng` or `adamw` leaves `model`, `opt`, and all live RNGs untouched or rolled back, with `fail-receipt-v1` written |
| `CURRENT_BEST`, `CURRENT_LAST`, `manifest.json`, `checkpoint.bin` on disk | **Untrusted durable artifacts until `resolve_pointer`** | Bounded read (`_read_bounded_bytes`), relative-path traversal guard (`_safe_generation_relpath`), manifest/payload SHA256 chain, `unpack`, `prov-v1` check | Prevents path escape, silent pointer/payload split, or partial file consumption |
| `model_kwargs` forwarded in `trainer.optimizer_step` | **Deferred to C3** | `prov-v1` and `exposure-v2` explicitly pin `"forward_binding": "DEFERRED_TO_C3"` | Forward tensor content/shape/dtype digest binding is NOT claimed in C2 |
| Ed25519 release root, clock, nonce ledger | **Deferred to C4 (`PENDING_OWNER`)** | `prov-v1` and `configs/stage_a_synthetic.json` pin `"owner_trust_root_status": "PENDING_OWNER_C4"`, `"release_created": false` | Zero Owner keys or execution releases exist or are generated |

---

## 3. Static Counterexample Rejection Matrix (C1 + C2 Summary)

1. **Envelope & Framing Corruption (`C2-NEG-01..04`):** Non-`STCA` magic, `header_len` < 2 or > `MAX_HEADER_BYTES`, duplicate JSON keys or `NaN` in header, missing `checkpoint_schema="stca-v2"`, overlapping/gapped section `offset`, section exceeding `SECTION_MAX_BYTES`, section/body SHA256 mismatch, or trailing bytes fail immediately in `unpack` before `_admit` or any Torch/NumPy load.
2. **Pre-Construction & Late-Section Corruption (`C2-NEG-05..09`):** Valid exposure + corrupt `python_rng` / `numpy_rng` fails in `_preflight_and_reconcile` **before** `model_cls(config_cls())` advances Torch RNG. Corrupt `model` (bad key/shape/dtype/`NaN`), corrupt `adamw` (missing `exp_avg`/`exp_avg_sq`/`step` when `completed > 0`, `NaN` moments, unknown param id), or wrong `torch_rng` length/bytes fail in Phase 1 `resume` **before** touching live weights or live RNGs, leaving `resume_stage` failure receipt.
3. **Mid-Apply Fault (`C2-NEG-09`):** Exception during `model.load_state_dict`, `opt.load_state_dict`, or `setstate` triggers `_rollback_live_state`, restoring all 5 snapshots identically and writing `resume_apply` receipt.
4. **Provenance & Budget Tampering (`C2-NEG-10`):** Altered `seed`, `clip`, `dev_events`, `horizons_file`, `generation_id != f"u{completed}"`, `release_created=True`, or checkpoint `completed > max_updates` fails preflight before model creation.
5. **Nonfinite DEV & Publish/Pointer Failure (`C2-NEG-11..14`):** Exception or `NaN`/`Inf` from `evaluate` always restores `model.train()` via `finally`, writes `dev_eval_failure`/`dev_nonfinite` receipt, and never modifies `macros`, `events`, `best`, `CURRENT_BEST`, or `CURRENT_LAST`. Re-publishing existing `u32-BEST`, failed staged readback, or tampered `CURRENT_BEST` pointer refuses mutation without corrupting prior `generations/u*-BEST`.

---

## 4. Non-Execution & Scope Ceiling Attestation

1. Static proofs hold by direct code inspection under `SOURCE_ONLY`; **zero** Python imports, compiles, unit tests, model forwards, optimizer steps, or filesystem checkpoint writes were executed (`STC_STAGE_A_SINGLE_GIT_INTEGRATION_STATIC_REVIEW` remains `NOT PASSED`).
2. `forward_binding="DEFERRED_TO_C3"`, `500_1` historical corpus remains `UNBOUND / NOT_AUTHORIZED`, and `owner_trust_root_status="PENDING_OWNER_C4"`.
