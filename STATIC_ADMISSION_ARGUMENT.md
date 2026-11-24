# Static Admission, Transaction, and Rollback Argument (`STATIC_ADMISSION_ARGUMENT.md`)

**Branch:** `local/stage-a-git-first`  
**C1 Accepted Baseline:** `7bb1c73f0c4bd252aa9dd42d5604e69ed7fe3bb6` / `7ed83698014a01007baa480f173a482b69a5d9c7`  
**Prior Reviewed C2 Candidate:** `b6cd585f4f91f77ef227d51fe4d06503ba23500d`  
**Current C2-AMENDMENT-01 Code Commit:** `e04d81b53ddf69adb955f49508d8046c1aa15c02`  
**Execution Status:** `SOURCE_ONLY / NO_IMPORT / NO_COMPILE / NO_TEST / NO_EXECUTION`

---

## 1. Call Graphs (`e04d81b53ddf69adb955f49508d8046c1aa15c02`)

### 1.1 Writer & Authoritative Atomic Commit Graph (`spine.py:run_synthetic:242-530`)

```text
run_synthetic(release_path, trust_anchor_path, expected, config)
  ├─> authorization.load(release_path, trust_anchor_path, expected)            [spine.py:243] (C4 trust root remains PENDING_OWNER_C4)
  ├─> _validate_run_config(config)                                             [spine.py:83-151,246] (pins schemas, RNG scope, fixed AdamW hyperparams)
  ├─> verify_sampler_pin() & verify_train_ids_pin(train_ids())                 [spine.py:249-250]
  ├─> horizons_file_sha256 == HORIZONS_SHA256                                  [spine.py:253-255; checkpoints.py:160-179,246-247]
  ├─> _canonical_effective_config_bytes(...) -> build_expected_provenance(...) [spine.py:154-195,256-269; checkpoints.py:230-271]
  ├─> [if resume_bytes or resume_role:] PRE-CONSTRUCTION PREFLIGHT (before torch import / model init):
  │     ├─ if resume_role: recover_commit_state(root, expected_provenance=base_provenance) [checkpoints.py:1153-1176]
  │     │                 -> resolve_pointer(root, role, expected_provenance=base_provenance) -> resolve_commit(...)
  │     ├─ _preflight_and_reconcile(resume_payload, expected_resume_cfg)       [spine.py:198-239,297]
  │     ├─ enforce completed <= max_updates                                    [spine.py:298-299]
  │     └─ on any exception -> write_failure_receipt("resume_preflight") -> raise (zero Torch/caller RNG side effect)
  ├─> caller_rng_snapshot = capture_cpu_rng_snapshot() if resume_payload else None [spine.py:312; checkpoints.py:513-523]
  ├─> model_cls(config_cls()); opt = AdamW(..., lr=1e-4, wd=1e-4, betas=(0.9,0.999), eps=1e-8, amsgrad=False)
  │     └─ on pre-resume setup failure when resuming -> restore_cpu_rng_snapshot(caller_rng_snapshot)
  │           -> if rollback errors: write_failure_receipt("fatal_rollback") + raise FatalRollbackError
  │           -> else: write_failure_receipt("resume_stage") + re-raise
  ├─> [if resume_payload is not None:] resume(..., failure_root=root, caller_rng_snapshot=caller_rng_snapshot)
  └─> while completed < max_updates:                                           [spine.py:375-521]
        ├─ C1 exposure pre-step checks & optimizer_step(model, batch, opt, clip) [forward_binding="DEFERRED_TO_C3"]
        └─ if completed in dev_events:
              ├─ try: value = evaluate(...) except: write_failure_receipt("dev_eval_failure"); raise finally: model.train()
              ├─ if nonfinite/bool(value): write_failure_receipt("dev_nonfinite"); raise ValueError
              ├─ compute next_macros, is_new_best, next_best, next_best_history, next_events in locals
              ├─ pack(stca-v2 + prov-v1, _sections(...))
              └─ commit_event(root, generation_id, payload, is_new_best=is_new_best, expected_provenance=step_provenance)
                    ├─ PostCommitError -> write_failure_receipt("post_commit_failure", commit_durable=True) -> raise (fail-stop)
                    ├─ other Exception -> write_failure_receipt("checkpoint_publish", commit_durable=False) -> raise (fail-stop)
                    └─ only on success -> advance macros, best, best_history, events
```

### 1.2 Single Authoritative `CURRENT_COMMIT` & Idempotent Recovery Graph (`C2-A01`, `C2-A02`)

```text
commit_event(root, generation_id, payload, is_new_best, expected_provenance)   [checkpoints.py:1376-1539]
  ├─ unpack(payload) + header["completed"] == gen_step + verify_checkpoint_provenance(expected_generation_id=generation_id)
  └─ with _single_writer_lock(root) [fcntl.flock(LOCK_EX | LOCK_NB) on root/.writer.lock]:
        ├─ if CURRENT_COMMIT exists:
        │     ├─ parse CURRENT_COMMIT (commit-v1), verify effective_config_sha256
        │     ├─ if generation_id == prev_commit["last"]["generation_id"]:
        │     │     verify exact LAST/BEST SHA256 + is_new_best match (else ValueError),
        │     │     idempotent _materialize_generation verify/reuse, _sync_derived_mirrors -> return "idempotent_reused"
        │     └─ else: enforce gen_step > prev_completed & earliest-tie is_new_best == (macro < prev_best) & header coherence
        ├─ _materialize_generation(root, generation_id, "LAST", ...)
        │     ├─ if generations/{id}-LAST exists: exact files + bytes + manifest + unpack + prov-v1 check -> reuse or refuse conflict
        │     └─ else: stage/{id}-LAST.tmp-NNNN -> write+fsync files/dir -> readback+unpack -> os.rename -> fsync dirs -> readback
        ├─ if is_new_best: _materialize_generation(root, generation_id, "BEST", ...)  (prior BEST dirs untouched)
        │  else: _verify_materialized_record(root, prev_commit["best"], "BEST", ...)
        ├─ verify header["best_history"] == committed best_history summary
        ├─ write CURRENT_COMMIT.partial + _fsync_file + readback
        ├─ os.replace(CURRENT_COMMIT.partial, CURRENT_COMMIT)        <== SINGLE ATOMIC COMMIT POINT
        └─ try: _fsync_dir(root) + readback + _sync_derived_mirrors(CURRENT_LAST, CURRENT_BEST)
           except: raise PostCommitError(generation_id=..., commit_sha256=...)
```

**Single-Commit-Point Crash Matrix (`C2-A01`, `C2-A02`):**

| Crash / Failure Point | On-Disk State | Reader Behavior (`resolve_commit` / `resolve_pointer`) | Recovery Behavior (`recover_commit_state` / retry `commit_event`) |
|---|---|---|---|
| Before `os.rename(stage, final)` | `CURRENT_COMMIT` points to `prev` (`u64`); `stage/u96-LAST.tmp-NNNN` incomplete | Resolves coherent `prev` (`LAST=u64`, `BEST=u64` or prior) | `recover_commit_state` inventories `stage/*` as `orphaned_stages`; retry allocates `.tmp-(N+1)` cleanly |
| After `os.rename(u96-LAST)` before `BEST` rename or `CURRENT_COMMIT` replace | `generations/u96-LAST` unreferenced on disk; `CURRENT_COMMIT` still points to `u64` | Resolves coherent `u64`; **ignores** unreferenced `u96-LAST` | `recover_commit_state` lists `u96-LAST` under `unreferenced_generations` (never guesses/promotes); exact retry verifies & reuses `u96-LAST` idempotently; conflicting payload refused without overwrite |
| After `os.rename(u96-LAST)` and `os.rename(u96-BEST)` before `CURRENT_COMMIT` `os.replace` | Both `u96-LAST` and `u96-BEST` unreferenced on disk; `CURRENT_COMMIT` still points to `u64` | Resolves coherent `u64` (`header.best_step` and `BEST` both at `u64`) | Same idempotent verification & atomic `CURRENT_COMMIT` completion on exact retry; `u64-BEST` bytes remain preserved |
| Initial event (`u32`) fails before `CURRENT_COMMIT` exists | `CURRENT_COMMIT` absent; `u32-LAST` or `u32-BEST` may exist in `generations/` | Fails with `CURRENT_COMMIT file missing` (no promotion by guessing) | `recover_commit_state` returns `committed=False` without promoting; exact `commit_event` retry verifies/reuses existing dirs and writes initial `CURRENT_COMMIT` |
| Failure AFTER `os.replace(CURRENT_COMMIT.partial, CURRENT_COMMIT)` (e.g., `_sync_derived_mirrors` error) | `CURRENT_COMMIT` durably reflects `u96`; `CURRENT_BEST`/`CURRENT_LAST` mirror files may lag | `resolve_pointer` delegates ONLY to `resolve_commit(CURRENT_COMMIT)`, so readers STILL see coherent `u96` | `commit_event` raises `PostCommitError`; `spine` emits `"post_commit_failure"` (`commit_durable=True`) & stops; `recover_commit_state` resyncs derived mirrors strictly from `CURRENT_COMMIT` |

### 1.3 End-to-End CPU RNG & Semantic AdamW Resume Graph (`C2-A03`, `C2-A04`)

1. **Source Justification for `AdamW` Updated Parameter Participation (`C2-A03`):**
   By direct inspection of frozen `pinned_splitter_v3.py:174-440` (`SplitterV3.__init__` and `forward`, default `mode="hybrid"`) and `pinned_losses_v3.py:52-109` (`SplitterV3Loss.forward` returning `"total"`):
   - Every parameter in `SplitterV3` created in `__init__` (`state_proj`, `delta_proj`, `accel_proj`, `anchor_proj`, `text_proj`, `progress_embed`, `transition_encoder`, `trajectory_blocks[*]` [`norm1`, `film_attn`, `attn`, `norm2`, `film_ffn`, `ffn`], `semantic_query_norm`, `text_norm`, `semantic_attn`, `traj_film`, `sem_film`, `fusion`, `writer_norm`, `writer`, `local_direction_head`, `anchor_direction_head`, `hybrid_gate`) lies on the computational graph feeding `delta_local`, `delta_anchor`, `pred_local`, `pred_anchor`, and `pred_hybrid`, all of which contribute positive non-zero terms to `loss["total"]` in `trainer.py:17`.
   - `spine.py:673-684` explicitly derives `expected_updated_pids` from `opt.state_dict()["param_groups"]` paired with `opt.param_groups` parameters where `requires_grad=True`, and `stage_adamw_state` (`checkpoints.py:655-756`) enforces exact equality `set(raw_state.keys()) == expected_pids` when `completed > 0`, and `raw_state == {}` when `completed == 0`.
2. **Exact AdamW Group & Slot Validation (`C2-A03`):**
   - `stage_adamw_state` checks exact group key sets, finite frozen hyperparameters (`lr=1e-4`, `weight_decay=1e-4`, `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`), exact type+value equality of all remaining group options against live `opt.state_dict()["param_groups"]`, and exact sequence equality `grp["params"] == live_pids`.
   - Fixed live `param_groups` are cloned cleanly; checkpoint group dicts are NEVER adopted blindly.
   - `_validate_adamw_slot` requires exact keys `{"step", "exp_avg", "exp_avg_sq"}` (NO scalar fallback branch), CPU floating-point `exp_avg` / `exp_avg_sq` matching live parameter shape & dtype and finiteness, strict non-negativity `(exp_avg_sq >= 0).all()`, and integral CPU scalar Tensor or exact int `step == completed`.
3. **End-to-End CPU-Only RNG Transaction & Fatal Rollback (`C2-A04`):**
   - Pre-construction preflight runs BEFORE `import torch` or `model_cls()`.
   - When resuming, `run_synthetic:312` captures `caller_rng_snapshot = capture_cpu_rng_snapshot()` BEFORE `model_cls(config_cls())` initializes weights.
   - `resume` (`spine.py:646-765`) admits 100% of returned metadata (`completed`, `epoch`, `cursor`, `last_cursor_before`, `last_receipt_sha256`, `best`, `best_history`, `events`, `macros`, `exposure`, `exposure_frames`, `sampler`, `provenance`) and stages all 5 CPU states BEFORE touching live `model`, `opt`, or RNGs.
   - Any failure restores `restore_rng = caller_rng_snapshot` (plus `snap_model` / `snap_adamw` on Phase 2 failure). If restoration itself raises any error, `resume` writes a `"fatal_rollback"` receipt recording BOTH `original_error` and `rollback_errors` with `fatal=True, must_terminate_process=True`, and raises `FatalRollbackError`. Training never continues after any resume or commit error (`fail-stop`).
   - **Explicit Device Scope (`rng_scope="cpu_only"`):** Only CPU Python/NumPy/Torch RNGs and CPU tensors are governed (`stage_model_state`, `_validate_adamw_slot`, and `stage_torch_rng` explicitly require `device.type == "cpu"`); **zero GPU parity claim is made.**

---

## 2. External Provenance & Trust Boundary (`C2-A05`)

| Surface | Authority Class | Enforcing Code (`e04d81b53ddf69adb955f49508d8046c1aa15c02`) | Security Boundary |
|---|---|---|---|
| `expected_config`, `effective_config_sha256`, `horizons_file_sha256 == HORIZONS_SHA256` | **Externally constructed from caller run config & frozen pins** | `_validate_run_config`, `_canonical_effective_config_bytes`, `build_expected_provenance`, `run_synthetic:253-269` | Never inferred from checkpoint payload; `horizons_path` runtime digest verified equal to `HORIZONS_SHA256` |
| `expected_provenance` on preflight, resume, commit & resolve | **Mandatory external verifier** | `_preflight_and_reconcile:211-217`, `commit_event:1390-1394`, `resolve_commit:1066-1110` | Missing or mismatched `expected_provenance` raises immediately before state mutation |
| `CURRENT_COMMIT`, `generations/*`, `header`, `sections` | **Untrusted serialized bytes until verified against external pins + `stca-v2` + `prov-v1` + `sampler-v2`/`exposure-v2` + Phase 1** | `unpack`, `_admit`, `verify_checkpoint_provenance`, `reconcile_checkpoint_state`, `stage_*` | Derived `CURRENT_BEST`/`CURRENT_LAST` mirror files never override `CURRENT_COMMIT` |
| `model_kwargs` in `trainer.optimizer_step` | **Deferred to C3** | `prov-v1` and `exposure-v2` pin `"forward_binding": "DEFERRED_TO_C3"` | Forward tensor content/shape/dtype binding NOT claimed in C2 |
| Ed25519 trust anchor, clock, nonce/revocation ledger | **Deferred to C4 (`PENDING_OWNER_C4`)** | `prov-v1`, `commit-v1`, `fail-receipt-v1`, `configs/stage_a_synthetic.json` pin `"owner_trust_root_status": "PENDING_OWNER_C4"`, `"release_created": false` | Integrity binding only; NO Owner keys or execution releases generated |

---

## 3. Non-Execution Attestation

- All claims above hold by direct static inspection under `SOURCE_ONLY`; ZERO Python imports, compiles, unit tests, forward passes, optimizer steps, or checkpoint writes have executed (`STC_STAGE_A_SINGLE_GIT_INTEGRATION_STATIC_REVIEW` remains `NOT PASSED`).
