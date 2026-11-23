# EGM Consolidated Codebase & Drift Review Report: Medieval II RL & Fast Battle Simulator

**Target Repository:** [`/home/tomek/medieval2-docker/`](file:///home/tomek/medieval2-docker/)  
**Review Protocol:** Elephant-Goldfish Model (`egm-review` + `egm-drift-report`)  
**Goldfish Personas Executed in Parallel:**
- `egm-codebase-details` (Pedantic Technical Quality & Sim-to-Real Parity Auditor)
- `egm-design` (Hostile Design & Mathematical/RL Architecture Critic)
- `egm-zoom-out` (Skeptical Architect — Simplicity, Coherence & Dead-Path Auditor)

---

## Executive Summary

Weights trained in the Python battle simulator ([`sim/sim_train.py`](file:///home/tomek/medieval2-docker/sim/sim_train.py) + [`sim/rl_env.py`](file:///home/tomek/medieval2-docker/sim/rl_env.py)) **cannot transfer cleanly to the real Medieval II M2TWEOP engine** ([`workspace/rl_battle_ai.lua`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua)) in the current state of the repository. Furthermore, several silent mathematical and training bugs degrade or completely disable portions of the simulator training pipeline itself.

Across the three independent Goldfish audits, **24 verified findings** were identified and deduplicated into:
- **8 🔴 Blockers** (silent training no-ops, inverted feature polarities, incompatible feature indices `252..292` and `50..77`, duplicate sparse grid emissions, and broken off-policy batch BPTT updates)
- **12 🟡 Major Issues** (BPTT Jacobian/bias/recurrence math errors, hardcoded siege action hijacking, 2v1 melee counter-attack scaling, holding units failing to turn toward enemies, and EDU parser bugs)
- **4 🟢 Nits & Design Document Drift Items**

---

## Part 1: Ranked Findings & Concrete Fix Recommendations

### 🔴 BLOCKERS (Must Fix for Valid Training & Sim-to-Real Weight Transfer)

#### 🔴 B1. Phase 1 (`vanilla_vs_vanilla`) Is a 100-Battle Complete No-Op in `sim_train.py` (`update_phase1` Never Called; `update_from_episode` Skips All `a == -1` Steps)
- **Citations:**
  - [`sim/sim_train.py:113, 164-166`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L164-L166)
  - [`sim/rl_env.py:520-523, 588-595`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L520-L523)
  - [`workspace/trainer/sarsa_trainer.py:120-174, 248-251`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L120-L174)
- **Mechanism:**
  1. During Phase 1 (`phase == 1`, `mode == "vanilla_vs_vanilla"`), [`sim/rl_env.py:523`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L523) records every learner transition with `"a": -1` (plus `"disp"` and `"vq"` for displacement imitation).
  2. However, [`sim/sim_train.py:164`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L164) unconditionally calls `st.update_from_episode(w, ep, phase, None, llm_terminal, weight_mult)` and **never calls `st.update_phase1(w, ep)`**.
  3. Inside [`sarsa_trainer.py:249-251`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L249-L251), the backward pass executes:
     ```python
     a = acts[t]
     if a < 0 or a >= NUM_ACTIONS:
         continue
     ```
     Because `a == -1` for every step in Phase 1, **100% of steps are skipped**, `d_W1, d_Wh, d_b1, d_W2` remain all zeros, and not a single gradient update occurs across the 100 battles of Phase 1 (only `rw["W2"][:, 16:]` is multiplied by `0.9985`, which is already `0.0`).
  4. **Secondary Bug in `update_phase1`:** If `update_phase1` ([`sarsa_trainer.py:167-173`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L167-L173)) *is* called as currently written, it updates **all 32 hidden neurons (`0..31`)** and mean-centers `rw["b2"] -= np.mean(rw["b2"])`, which destroys the KBANN symbolic rule backbone (`neurons 0..15`) and centering biases (`b2[act] += wt`) constructed in [`kbann_init.py:22-35`](file:///home/tomek/medieval2-docker/workspace/trainer/kbann_init.py#L22-L35) and frozen in Phase 2+ ([`sarsa_trainer.py:269-273`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L269-L273)). Also, the end-of-battle flush in [`sim/rl_env.py:588-595`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L588-L595) omits `"disp"` and `"vq"`, giving the final step `disp=None, vq=None`.
- **Concrete Fix:**
  - Either set `--phase1-battles 0` by default (since KBANN initialization in `kbann_init.py` already compiles Vanilla tactical rules directly into `0..15`), **or** in [`sim/sim_train.py:164`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L164) branch on `if phase == 1: st.update_phase1(w, ep)` while restricting [`sarsa_trainer.py:167-173`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L167-L173) to update only residual neurons `[16:]` without altering `b2`.

---

#### 🔴 B2. Commander RNN Input Features `phi[252..292]` & Squad Partitioning Are Completely Incompatible Between `sim/rl_env.py`, `workspace/rl_battle_ai.lua`, and `workspace/trainer/kbann_init.py`
- **Citations:**
  - [`sim/rl_env.py:387-406, 453-481`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L387-L406)
  - [`workspace/rl_battle_ai.lua:589-691`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L589-L691)
  - [`workspace/trainer/kbann_init.py:173-175`](file:///home/tomek/medieval2-docker/workspace/trainer/kbann_init.py#L173-L175)
- **Mechanism:**
  1. **Feature Index Clash on `252..292`:**
     - **In `rl_battle_ai.lua:615-671`:**
       - `252`: Step-0 global command indicator (`1.0`).
       - `253..262`: One-hot current `globalCmd` (`253 + globalCmd`) on squad steps.
       - `263..272`: One-hot `prevGlobalCmd` (step 0) or `prevStepCmd` (squad steps).
       - `273..282`: One-hot `prevGlobalCmd` (step 0) or `ownPrev` (`S.prevSubCmds[sk]`).
       - `283..289`: One-hot squad role (`282 + sq.ri` for `ri in 1..7`).
       - `290, 291, 292`: `[sqDx, sqDy, sqHp]`.
     - **In `sim/rl_env.py:456-470`:**
       - Step 0 sets `[(252, 1.0)]` (omitting `263..282`).
       - Squad steps `sq in (1, 2, 3)` set `[(252 + sq, 1.0), (263 + g_cmd, 1.0)]`!
       - Here, indices `253, 254, 255` encode **squad index 1..3** in Python, colliding directly with **one-hot `globalCmd` (`253..262`)** in Lua!
       - Indices `256..262` and `273..292` are **permanently `0.0` in Python**!
       - Worse, Commander KBANN Rule 3 (`[(6, -3.0), (290, -2.0)]`) and Rule 4 (`[(7, -3.0), (290, 2.0)]`) in [`kbann_init.py:173-175`](file:///home/tomek/medieval2-docker/workspace/trainer/kbann_init.py#L173-L175) depend on feature `290` (`sqDx`), which is never populated in `sim/rl_env.py`.
  2. **Squad Partitioning & World-X Flank Inversion:**
     - `sim/rl_env.py:387-406` (`_assign_squads`) dynamically splits the army into 3 spatial zones using raw world X (`u.x < inf_cx - 45.0` $\to$ 1, `u.x > inf_cx + 45.0` $\to$ 3, else 2). Because it uses world `u.x` instead of army-frame lateral `rx = (u.x - inf_cx) * fr["rx"] + (u.y - inf_cy) * fr["ry"]`, **Side 1 (facing `-Y`) has its left and right flanks swapped (`sq=1` vs `sq=3`)** during Phase 4 self-play!
     - `rl_battle_ai.lua:589-602` (`assignSquads`) statically groups units once at battle start **by role** in chunks of `MAX_SQUAD_SIZE = 3` (`line_inf_1`, `line_inf_2`, `spear_inf_1`, ...), and sets `lead` to the leftmost `line_inf_1` squad (`rl_battle_ai.lua:678-680`) rather than the center-most `line_inf` unit (`rl_env.py:403-405`).
- **Concrete Fix:**
  - Unify squad assignment and Commander RNN feature building `252..292` between `sim/rl_env.py` and `workspace/rl_battle_ai.lua` so both populate `253..292` with identical semantics and compute lateral offsets in the army coordinate frame `(fr["rx"], fr["ry"])`.

---

#### 🔴 B3. Commander RNN Receives Identically Zero Local Step Rewards (`dk=0, dl=0, coh=0`) in `sim/rl_env.py`, and Interleaved BPTT Trajectories Corrupt Recurrent Gradients
- **Citations:**
  - [`sim/rl_env.py:456-474`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L456-L474)
  - [`workspace/trainer/sarsa_trainer.py:89-93, 312-366`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L312-L366)
  - [`workspace/rl_battle_ai.lua:804-830, 930-939`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L804-L830)
- **Mechanism:**
  1. **Zero Local Advantage (`loc_adv == 0`) in Simulator:**
     In [`sim/rl_env.py:462, 470`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L462), `st["cmd_steps"].append({"phi": ..., "h": ..., "a": ...})` never attaches `"dk"`, `"dl"`, or `"coh"`. In [`sarsa_trainer.py:322, 338`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L322), `step_reward` returns `0.0` for all $t < T-1$, and `r_loc = rewards[t] - (term_total if t == T - 1 else 0.0)` is **identically `0.0` for every step $t \in [0, T-1]$**. Thus `local_ret` and `loc_adv` are identically zero across the entire episode, giving every Commander decision from $t=0$ to $t=T-1$ the exact same flat terminal scalar `ep_adv`.
  2. **Broken Recurrent BPTT Chain:**
     - In [`sim/rl_env.py:457-470`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L457-L470), 4 decisions at the same timestamp (`global`, `sq=1`, `sq=2`, `sq=3`) are chained through `st["cmd_h"]` and pushed into one flat `cmd_steps` list, so `g_loc = r_loc + 0.85 * g_loc` discounts across squad index within the same tick.
     - In [`workspace/rl_battle_ai.lua:622-682, 930-939`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L622-L682), Lua chains a single shared `h` across `globalCmd` and all squads within a tick, **but splits the logged steps into separate `"commander"` entries** (`S.cmdSteps` + each `S.subSteps[sk]`). When `update_commander_rnn` backpropagates `rnn["Wh"].T @ dh_next` along each split list, `hs[t]` did not produce `hps[t+1]` of that list because 5–10 other squad steps mutated `h` in between!
- **Concrete Fix:**
  - Record per-interval `(dk, dl, coh)` on Commander steps in `sim/rl_env.py` when flushing at the next `CMD_INTERVAL` (and at battle end), and maintain either one unified recurrent trajectory or separate per-detachment hidden states `cmd_h[sq]`.

---

#### 🔴 B4. Morale Feature (`l[19]` / Feature Index `69`) and Fatigue Feature (`l[20]` / Feature Index `70`) Scales Are Inverted/Mismatched Between Python and Lua
- **Citations:**
  - [`sim/combat.py:825-839`](file:///home/tomek/medieval2-docker/sim/combat.py#L825-L839)
  - [`sim/rl_env.py:196-197`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L196-L197)
  - [`workspace/rl_battle_ai.lua:20, 84, 213, 525`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L20)
  - [`m2tw_rl_ai_design_doc.md:86, 114`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L114)
- **Mechanism:**
  1. **Morale Polarity Inversion (`l[19]` / index `69`):**
     - In M2TWEOP ([`rl_battle_ai.lua:20, 84`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L20)), `moraleLevel == 0` is `berserk` (highest morale) and `moraleLevel == 6` is `routing` (lowest morale). Thus in Lua ([`rl_battle_ai.lua:213`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L213)), `clamp((r.u.moraleLevel or 0) / 6, 0, 1)` is **`0.0` at highest morale and `1.0` when routing**.
     - In [`sim/combat.py:826-839`](file:///home/tomek/medieval2-docker/sim/combat.py#L826-L839), `u.morale_level` is mapped with the **exact opposite polarity**: `0` when routing and `6` when morale is highest (`>= 11.5`). Thus in Python ([`sim/rl_env.py:196`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L196)), `l[19]` is **`1.0` at highest morale and `0.0` when routing**!
  2. **Fatigue Scale Mismatch (`l[20]` / index `70`):**
     - In Python ([`sim/combat.py:484, 592`](file:///home/tomek/medieval2-docker/sim/combat.py#L484)), `u.fatigue` is an integer tier `0..6` (`int(u.fatigue_val / 16.0)`), so `l[20] = clamp(u.fatigue / 6.0, 0, 1)` spans `[0, 1]`.
     - In M2TWEOP ([`rl_battle_ai.lua:525`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L525)), `r.u.fatigue` is the raw engine fatigue counter (e.g., `r.u.fatigue = 30` for merely winded), so `clamp((r.u.fatigue or 0) / 6, 0, 1)` at [`rl_battle_ai.lua:213`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L213) saturates to `1.0` almost immediately!
- **Concrete Fix:**
  - Invert `l[20]` in `rl_battle_ai.lua:213` to `clamp((6 - (r.u.moraleLevel or 3)) / 6, 0, 1)` (or invert `combat.py:826-839`), and scale raw M2TWEOP `r.u.fatigue` by its engine maximum (or map to `0..6` fatigue tiers before dividing by `6`).

---

#### 🔴 B5. Local Unit Combat Stat Features `l[10]`, `l[11]`, `l[13]` (Indices `60, 61, 63`) Use Real EDU Stats in Python vs Hardcoded/Wrong Constants in Lua
- **Citations:**
  - [`sim/rl_env.py:185-190`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L185-L190)
  - [`workspace/rl_battle_ai.lua:203-208, 223-225`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L203-L208)
- **Mechanism:**
  - **Feature `60` (`l[10]` / Lua `l[11]` — Charge Bonus):** Python computes `clamp(t.melee_chg / 15.0, 0.0, 1.0)`. Lua hardcodes `local chg = 4` ([`rl_battle_ai.lua:204`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L204)), feeding a constant `4/15 = 0.2667` for every unit.
  - **Feature `61` (`l[11]` / Lua `l[12]` — Missile Attack):** Python computes `clamp((t.pri_atk if t.is_missile else 0.0) / 20.0, 0.0, 1.0)`. Lua ([`rl_battle_ai.lua:223`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L223)) feeds `clamp((isMissile and atk or 0) / 20, 0, 1)` where `atk` is `aiUnitValuePerSoldier` (overall unit value ~5–18), not missile attack!
  - **Feature `63` (`l[13]` / Lua `l[14]` — Total Defense):** Python computes `clamp((t.armour + t.def_skill + t.shield) / 30.0, 0.0, 1.0)`. Lua ([`rl_battle_ai.lua:225`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L225)) hardcodes `l[14] = 0.5` for every unit.
  - **Feature `55` (`l[5]` / Lua `l[6]` — Spearmen flag):** Python checks `t.pri_spear or t.pri_long_pike`; Lua checks `combo == 3 or cls == 3`, missing heavy-class spear/halberd units.
- **Concrete Fix:**
  - Read real EDU stats in `rl_battle_ai.lua` (`r.u.attackInBattle`, `r.u.armourInBattle`, `e.charge`, `e.defence`, `e.shield`) or align both sides to the exact same EDU fields.

---

#### 🔴 B6. Egocentric Grid (`78..461`) Emits Duplicate Sparse Indices in `rl_battle_ai.lua` and Conflates Moving Units With Holding Units in `sim/rl_env.py`
- **Citations:**
  - [`workspace/rl_battle_ai.lua:123-134, 232-246, 275`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L232-L246)
  - [`sim/rl_env.py:158-165, 213-231`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L158-L165)
- **Mechanism:**
  1. **Per-Unit `seen` Table in Lua Multiplies Grid Weights by Unit Count in Cell:**
     In [`rl_battle_ai.lua:232-244`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L232-L244), `local seen = {}` is declared **inside** `local function cell(o, isEnemy)` instead of outside the unit loop (`for _, o in ipairs(...)`). When $k$ units occupy the same $25\text{m} \times 25\text{m}$ cell, `out` receives $k$ duplicate `{i, 1}` entries, and `unitRNNForward` ([`rl_battle_ai.lua:275`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L275)) sums them $k$ times (`k * W1[j, i]`), whereas [`sim/rl_env.py:213`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L213) deduplicates across all units via a single outer `seen = set()` (`1 * W1[j, i]`).
  2. **Moving Units Always Tagged as `status_hold` in Python vs No Status Channel in Lua:**
     In [`rl_battle_ai.lua:128-134`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L128-L134), moving units (`dx^2 + dy^2 >= 4`) return `st = 0` and set only the allegiance bit (`if st >= 3 then once(base + st - 1) end`). In [`sim/rl_env.py:158-165`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L158-L165), `_unit_status(u)` has no moving case and returns `3` (`hold`) for all non-routing, non-engaged, non-firing units, setting channel 2 (`status_hold`) for every moving unit on the battlefield!
- **Concrete Fix:**
  - Move `local seen = {}` outside `cell(o, isEnemy)` in [`rl_battle_ai.lua:232`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L232).
  - In [`sim/rl_env.py:158-165, 227`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L158-L165), return `0` from `_unit_status(u)` when `u.order_kind in ("move", "attack", "withdraw")` and the unit is moving, and only emit `base + st - 1` when `st >= 3`.

---

#### 🔴 B7. 24-Battle Batch Off-Policy Sequential In-Place Gradient Updates With Stale Hidden States in `sim_train.py` & `sarsa_trainer.py`
- **Citations:**
  - [`sim/sim_train.py:115, 148-206`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L148-L206)
  - [`workspace/trainer/sarsa_trainer.py:200-274, 312-372`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L200-L274)
- **Mechanism:**
  1. [`sim/sim_train.py:148-155`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L148-L155) simulates a batch of `bsz = 24` battles in parallel under a single snapshot of weights `(w, rnn)`.
  2. When the 24 episodes finish, [`sim/sim_train.py:156-167`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L156-L167) calls `update_from_episode` and `update_commander_rnn` 24 times sequentially, and inside `update_from_episode` ([`sarsa_trainer.py:200-274`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L200-L274)), `rw` is mutated **in-place after every individual unit** ($24 \times 5 = 120$ sequential in-place mutations of `rw` per batch for `line_inf`!).
  3. On updates $2 \dots 120$, `update_from_episode` ([`sarsa_trainer.py:219-220`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L219-L220)) computes `probs` using the **already-mutated weights `rw`** while feeding the **stale rollout hidden state** `hp = _dense_unit_h(s.get("h"))` from the old policy! That means:
     - `hs[t]` (computed with mutated `rw` from stale `hps[t]`) is **not** equal to `hps[t+1]`, breaking the BPTT chain rule through `rw["Wh"].T @ dh_next` at [`sarsa_trainer.py:260`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L260).
     - Applying on-policy REINFORCE gradient `adv[t] * (one_hot - probs)` 120 times sequentially without an importance-sampling ratio $\pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ causes severe policy overshoot.
  4. **Mid-Batch Phase Transition Bleed:** Phase transitions (`1 -> 2 -> 3 -> 4`) are triggered inside the `for ep, verdict in results:` loop ([`sim/sim_train.py:186-206`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L186-L206)), so when a transition occurs mid-batch, the remaining episodes in `results` (simulated under the *old* phase's mode) are evaluated and recorded into `s["recent"]` under the *new* phase!
- **Concrete Fix:**
  - Accumulate gradients across all units/episodes in the batch evaluated at the rollout policy $\theta_{\text{old}}$ (or roll `h_curr` forward self-consistently from $t=0$) and apply a single averaged update per batch.
  - Clamp batch size `bsz` so batches do not straddle `args.phase1_battles`.

---

#### 🔴 B8. Battering Ram & Siege Ladder Units Freeze Permanently in `sim/combat.py` If Ordered to `"attack"`
- **Citations:**
  - [`sim/combat.py:457-485, 514-538`](file:///home/tomek/medieval2-docker/sim/combat.py#L457-L485)
- **Mechanism:**
  In [`sim/combat.py:457-485`](file:///home/tomek/medieval2-docker/sim/combat.py#L457-L485):
  ```python
  if u.order_kind == "attack" and u.target_uid is not None and not (u.has_ram and self.gate_hp > 0) and not u.has_ladder:
      ...
  elif u.order_kind in ("move", "withdraw"):
      gx, gy = u.target_x, u.target_y
  else:
      # Hold: recover fatigue slowly, face nearest active enemy
      u.fatigue_val = max(0.0, u.fatigue_val - 1.0 * dt)
      u.fatigue = min(6, int(u.fatigue_val / 16.0))
      return
  ```
  Because line 457 excludes `(u.has_ram and self.gate_hp > 0)` and `u.has_ladder`, any ram or ladder unit whose `u.order_kind == "attack"` fails the `if`, fails the `elif u.order_kind in ("move", "withdraw")`, enters the `else:` branch, and **immediately `return`s** before reaching the siege ram/ladder movement logic at lines 514–538!
  Additionally, if the 2 ram carriers and 2 ladder carriers rout or die under tower/archer focus fire ([`sim/combat.py:146-160`](file:///home/tomek/medieval2-docker/sim/combat.py#L146-L160)), no other unit ever picks up the ram or ladders, permanently deadlocking the siege if artillery runs out of ammo or is absent.
- **Concrete Fix:**
  - Remove `and not (u.has_ram and self.gate_hp > 0) and not u.has_ladder` from line 457 (since lines 517–537 already override `gx, gy` for ram/ladder carriers), and reassign dropped rams/ladders when a carrier routs or dies.

---

### 🟡 MAJOR ISSUES (Mathematical, Simulation, & Sim-to-Real Behavioral Bugs)

#### 🟡 M1. Softmax Inverse Temperature Mismatch (`2.2` in Python vs `3.5` in Lua) & Missing `inv_temp` Factor in BPTT Softmax Gradient
- **Citations:**
  - [`sim/rl_env.py:25`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L25)
  - [`workspace/trainer/sarsa_trainer.py:108, 153, 255, 354`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L108)
  - [`workspace/rl_battle_ai.lua:21, 292-293, 314-337, 404-409`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L292-L293)
- **Mechanism:**
  1. Python uses `inv_temp = 2.2` in `_softmax_sample` ([`sim/rl_env.py:25`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L25)) and `_softmax` ([`sarsa_trainer.py:108`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L108)), whereas Lua uses `local scale = invTemp or 3.5` ([`rl_battle_ai.lua:293`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L293)).
  2. Since $\pi_a = \frac{\exp(\beta z_a)}{\sum_k \exp(\beta z_k)}$, the gradient with respect to pre-temperature logits $z$ is $\frac{\partial \log \pi_a}{\partial z} = \beta (\mathbf{e}_a - \boldsymbol{\pi}) = 2.2 (\mathbf{e}_a - \boldsymbol{\pi})$, but [`sarsa_trainer.py:153, 255, 354`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L255) omits the $\beta = 2.2$ multiplier.
  3. In [`rl_battle_ai.lua:21, 314, 404`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L21), `EPS = { learner = 0.10, frozen = 0.02 }` is threaded into `stepCommanderRNN` and `chooseAction` as `eps`, which is completely ignored by both functions.
- **Concrete Fix:** Set `invTemp` in `rl_battle_ai.lua:293` to `2.2`, multiply `dz` by `inv_temp` in `sarsa_trainer.py`, and use lower temperature / greedy selection for the frozen opponent when `not is_learner`.

---

#### 🟡 M2. Decision Cadence Mismatch (`6s/12s` in Python vs `18s/18s` in Lua) & Holding/Firing Units Bypass `UNIT_COMMIT_SEC` Every `2.0s` in Python
- **Citations:**
  - [`sim/rl_env.py:20-22, 486-490`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L20-L22)
  - [`workspace/rl_battle_ai.lua:8-11, 845-863`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L8-L11)
- **Mechanism:**
  - `sim/rl_env.py` uses `CMD_INTERVAL = 12.0` and `UNIT_COMMIT_SEC = 6.0`; `rl_battle_ai.lua` uses `CMD_INTERVAL = 18.0` and `UNIT_COMMIT_SEC = 18.0` ($3\times$ slower recurrent stepping).
  - Worse, [`sim/rl_env.py:486`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L486) checks:
    ```python
    if not imitate and sim.t - u.last_cmd_t < UNIT_COMMIT_SEC and (u.engaged or u.order_kind != "hold"):
        continue
    ```
    Because of `and (u.engaged or u.order_kind != "hold")`, any unengaged unit with `u.order_kind == "hold"` (Action 8, OR any missile unit firing in range at line 329!) **bypasses `UNIT_COMMIT_SEC` and re-steps its RNN every `SIM_DT = 2.0s`**, accumulating $3\times$ more trajectory steps and $3\times$ more positive cohesion rewards (`+0.04` per step) than moving units!
- **Concrete Fix:** Replace line 486 in `sim/rl_env.py` with early wakeup only when `u.engaged != getattr(u, "was_engaged", False)` (matching `rl_battle_ai.lua:854`), and unify `UNIT_COMMIT_SEC` and `CMD_INTERVAL` between Python and Lua.

---

#### 🟡 M3. Per-Trajectory Temporal Mean Centering of `local_ret` Punishes Steady-Damage Units at Battle End, and `term_base` Has a $4\times$ Win/Loss Asymmetry
- **Citations:**
  - [`workspace/trainer/sarsa_trainer.py:189, 231-239, 335-343`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L189)
- **Mechanism:**
  1. `loc_adv = (local_ret - np.mean(local_ret)) / (np.std(local_ret) + 2.0)` centers discounted return `local_ret` within each single unit's trajectory. Because `local_ret[t]` is a truncated finite-horizon sum $\sum_{k=t}^{T-1} 0.85^{k-t} r_k$, a unit earning a constant positive reward $r_t = +c > 0$ at every step has `local_ret[0] > np.mean(local_ret) > local_ret[T-1]`, automatically assigning **negative `loc_adv`** to the unit's actions near the end of the battle! Conversely, a unit that sits idle ($r_t = 0$) and takes heavy casualties at one step gets **positive `loc_adv`** on all its idle steps.
  2. At [`sarsa_trainer.py:189`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L189), `term_base = (TERMINAL * 0.25) if (result > 0 and not is_dec) else (TERMINAL * result)` gives `+25.0` for a non-decisive win vs `-100.0` for a non-decisive loss (`-270.0` for a decisive loss), while `DMG_BONUS = 0.0` (line 12).
  3. At [`sarsa_trainer.py:259, 270-273, 358, 368-371`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L270-L273), `d_b2` and `d_bo` are computed but never applied while `W2[:, 16:]` and `Wo[:, 10:]` decay by `0.9985` every unit/episode.
- **Concrete Fix:** Subtract a cross-unit role baseline (or per-step reward mean `r_loc - np.mean(r_loc)` before discounting) instead of centering the truncated discounted sum `local_ret`, and symmetrize `term_base`.

---

#### 🟡 M4. KBANN Rules 10..15 Assume Unscaled `x[30..35] = 1.0`, But `rl_env.py` and `rl_battle_ai.lua` Scale `g[20..49]` by `3.0`
- **Citations:**
  - [`workspace/trainer/kbann_init.py:138-145`](file:///home/tomek/medieval2-docker/workspace/trainer/kbann_init.py#L138-L145)
  - [`sim/rl_env.py:171`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L171)
  - [`workspace/rl_battle_ai.lua:199`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L199)
- **Mechanism:**
  In [`kbann_init.py:138-145`](file:///home/tomek/medieval2-docker/workspace/trainer/kbann_init.py#L138-L145), Rules 10..15 set input weights `[(30, 1.2)] .. [(35, 1.2)]` with bias `-1.8` under the comment `# Gentle initial coupling to Commander detachment order x[30..35]`. However, [`sim/rl_env.py:171`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L171) and [`rl_battle_ai.lua:199`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L199) multiply features `20..49` by `3.0`! Thus `x[30..35]` arrives as `3.0`, making the pre-activation `1.2 * 3.0 - 1.8 = +1.8` (`tanh(+1.8) = +0.947`) instead of `-0.6`, fully saturating the rule neurons and shifting `b2` centering.
- **Concrete Fix:** Adjust the input weight in `kbann_init.py:140-145` from `1.2` to `0.4` (or `1.0` with bias `-1.8`) to account for the `3.0x` feature scaling on `g[20..49]`.

---

#### 🟡 M5. Hardcoded Siege Action Hijacking in `sim/rl_env.py:_issue_action` Masks Policy Actions and Diverges From `rl_battle_ai.lua:issue`
- **Citations:**
  - [`sim/rl_env.py:294-367`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L294-L367)
  - [`workspace/rl_battle_ai.lua:427-458`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L427-L458)
- **Mechanism:**
  1. In [`sim/rl_env.py:298-304`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L298-L304), any attacker with `has_ram` or `has_ladder` that picks any action `a not in (5, 6, 7)` (10 out of 13 actions) has its action forcibly replaced with marching to the gate/wall, while the trajectory records `a` as the action that produced the reward.
  2. In [`sim/rl_env.py:308-314`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L308-L314), any defender inside the fortress picking `a in (9, 10, 11, 12)` while no enemy is inside the wall is forcibly overridden to `"hold"`.
  3. In [`sim/rl_env.py:352-363`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L352-L363), once a breach opens, any attacker picking move actions `0, 1, 2` has its relative offset replaced with absolute coordinates `(sim.plaza_x, sim.plaza_y)`.
  4. For all movement actions `0..7`, [`sim/rl_env.py:366`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L366) sets `u.run = True` unconditionally (burning fatigue across the map), whereas [`rl_battle_ai.lua:438`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L438) runs only when `ed < 150 or u:isUnderFire()`.
  5. When already engaged in melee, [`rl_battle_ai.lua:441`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L441) ignores attack orders `9..12` (`if u:isEngaged() then return end`), whereas `sim/rl_env.py:321-348` allows re-targeting mid-melee.
- **Concrete Fix:** Align `_issue_action` in `sim/rl_env.py` with `issue` in `rl_battle_ai.lua` (including `u.run = (ed < 150 or u.is_under_fire)` on moves `0..7` and no-op on `9..12` when `u.engaged`), and if siege overrides are used in the simulator, log the effective overridden action or encode plaza/gate direction in features rather than silently mutating the semantics of actions `0, 1, 2`.

---

#### 🟡 M6. Centroid and Infantry Centroid Formulas Differ Between `sim/rl_env.py` and `workspace/rl_battle_ai.lua`
- **Citations:**
  - [`sim/rl_env.py:54-76`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L54-L76)
  - [`workspace/rl_battle_ai.lua:87-103`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L87-L103)
- **Mechanism:**
  - `_inf_centroid` in [`sim/rl_env.py:70`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L70) includes **all non-cavalry, non-artillery active units** (`if u.active and not u.tpl.is_cav and u.role != "artillery":`), which **includes `"missile_foot"`** (and any dismounted `"general"`), weighted by `u.alive`.
  - `infantryCentroid` in [`rl_battle_ai.lua:97`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L97) includes **only `r.role == "line_inf" or r.role == "spear_inf"`** (excluding `"missile_foot"`), **unweighted** by soldier count.
  - `_centroid` in [`sim/rl_env.py:57-61`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L57-L61) weights units by `u.alive` (and includes routing units with weight `1`), whereas `centroid` in [`rl_battle_ai.lua:89`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L89) is an unweighted mean over non-routing units.
- **Concrete Fix:** Change `_inf_centroid` in `sim/rl_env.py:70` to `if u.active and u.role in ("line_inf", "spear_inf"):` and weight both Python and Lua centroids consistently by `alive(r)` over `active(r)` units.

---

#### 🟡 M7. Frozen Opponent Commander RNN State Reset & 18s Unit Strobe Bug in `workspace/rl_battle_ai.lua`
- **Citations:**
  - [`workspace/rl_battle_ai.lua:608-613, 764-771, 836-841`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L608-L613)
- **Mechanism:**
  In `rl_battle_ai.lua` when `S.mode == "rl_vs_frozen"`:
  1. `updateCommander` ([`rl_battle_ai.lua:612, 684`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L612)) initializes `local h = (learner and S.cmdH) or zeroHidden()` and only saves `S.cmdH`, `S.prevGlobalCmd`, and `S.prevSubCmds` when `learner == true`. Thus the frozen opponent (`learner == false`) **wipes its Commander RNN hidden state to zeros on every tick** and never receives `prevGlobalCmd` or `prevSubCmds`!
  2. `actSide` ([`rl_battle_ai.lua:840`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L840)) is only called inside the `if t >= (S.nextCmdTick or 0)` block (every `18.0s`), so the frozen opponent's units are never placed in a round-robin queue and never receive early wakeup on melee engagement changes.
- **Concrete Fix:** Store `S.cmdHF`, `S.prevGlobalCmdF`, and `S.prevSubCmdsF` for the frozen side in `rl_battle_ai.lua`.

---

#### 🟡 M8. 2v1 Melee Focus-Fire Gives Outnumbered Unit $k \times$ Free Full-Frontage Counter-Attacks & Side 0 Always Strikes First in `sim/combat.py`
- **Citations:**
  - [`sim/combat.py:326-358`](file:///home/tomek/medieval2-docker/sim/combat.py#L326-L358)
- **Mechanism:**
  1. When $k$ units ($A_1 \dots A_k$) attack a single defender $B$, each pair $(A_i, B)$ is unique in `fought_pairs`. For each pair, [`sim/combat.py:355-357`](file:///home/tomek/medieval2-docker/sim/combat.py#L355-L357) calls `self._apply_melee(u, tgt, dt)` **and** `self._apply_melee(tgt, u, dt)` at 100% of $B$'s frontage! Thus a single unit surrounded by 3 enemies deals **$3\times$ total melee damage per tick**!
  2. Because `self.units` is ordered Side 0 first, Side 1 second, `self._apply_melee(u, tgt, dt)` always kills Side 1 soldiers *before* Side 1's counter-attack `self._apply_melee(tgt, u, dt)` is computed on line 357.
  3. Units ordered via `"move"` (`a in 0..7`) that collide with an enemy and auto-engage at line 333 never set `u.charging = True` (which is only set in the `u.order_kind == "attack"` branch at lines 469–478).
- **Concrete Fix:** First collect all engaged melee pairs in `step(dt)`, split each unit's effective fighter count `u.alive` across its active engaged opponents, and apply damage simultaneously.

---

#### 🟡 M9. Holding Units in `sim/combat.py` Never Turn to Face Nearest Enemy (Disabling Braced Spear Bonus After Lateral Moves)
- **Citations:**
  - [`sim/combat.py:481-485, 594-628`](file:///home/tomek/medieval2-docker/sim/combat.py#L481-L485)
- **Mechanism:**
  In [`sim/combat.py:481-485`](file:///home/tomek/medieval2-docker/sim/combat.py#L481-L485), the `else:` (`"hold"`) branch says:
  ```python
  # Hold: recover fatigue slowly, face nearest active enemy
  u.fatigue_val = max(0.0, u.fatigue_val - 1.0 * dt)
  u.fatigue = min(6, int(u.fatigue_val / 16.0))
  return
  ```
  It **omits `self._turn_toward(...)`**. If a spear unit moves left/right (`a = 3` or `4`, setting `u.fx = ±1, u.fy = 0`) and then holds (`a = 8`), its facing vector `(fx, fy)` stays perpendicular to the enemy army! When enemy cavalry charges from the front, `_hit_quadrant` ([`sim/combat.py:594-605`](file:///home/tomek/medieval2-docker/sim/combat.py#L594-L605)) classifies the frontal charge as a `"left"` or `"right"` flank hit (`dot ≈ 0`), completely disabling `braced_spear` and `braced_bonus` ([`sim/combat.py:615-628, 641`](file:///home/tomek/medieval2-docker/sim/combat.py#L615-L628)) and giving the charging cavalry a flank attack bonus!
- **Concrete Fix:** In `_move_unit`'s `hold` branch (`combat.py:481-485`), find the nearest active enemy and call `self._turn_toward(u, enemy.x - u.x, enemy.y - u.y, dt)` when `not u.engaged`.

---

#### 🟡 M10. Siege Garrison Flank Stacking, Routing Through Walls/Rivers, & Unreachable Artillery Wall Breaches in `sim/combat.py`
- **Citations:**
  - [`sim/combat.py:223-230, 373-384, 439-451, 489-511, 663-720`](file:///home/tomek/medieval2-docker/sim/combat.py#L223-L230)
- **Mechanism:**
  1. **Siege Garrison Infantry Stacking (`combat.py:229-230`):** In `_deploy_side` (`style == "siege_garrison"`), all infantry units with `i >= 3` are placed at `self.gate_x + wing * 95.0 - dx_shift, self.wall_y - 24.0 * sign` where `wing = -1.0 if (i % 2 == 1) else 1.0` has no slot offset, stacking 3–4 infantry units on the exact same `(x, y)` point on each wing.
  2. **Routing Units Ignore Walls & Rivers (`combat.py:439-451`):** `if u.routing:` returns early before the river and siege wall movement constraints, allowing routing units inside or outside a fortress to walk straight through solid walls and deep rivers.
  3. **River Missing From `_blocked_by_barrier` & False River Magnet (`combat.py:373-384, 491`):** `_blocked_by_barrier` only checks siege walls, allowing units on opposite river banks off-bridge to melee across water if `u.radius + e.radius + 5.0` spans the bank gap. Also, `wants_cross` at line 491 triggers whenever `abs(gy - self.river_y) < self.river_half_w` even if `(u.y - self.river_y) * (gy - self.river_y) > 0`, pulling units moving along their own bank laterally toward `best_b["x"]`.
  4. **Artillery Shots Multiplied by Crew Count (`combat.py:706`) & Wall Breach Requires Live Enemy Target (`combat.py:663-681, 713-719`):** At line 706, `shots = u.alive * (3.0 if u.tpl.pri_area else 1.0)` uses `u.alive` (which for `category == "siege"` on `UNIT_SCALE = 2.5` is ~40–60 crewmen!), firing ~120–180 area shots per volley, and artillery cannot fire at walls/gates unless an enemy unit is within range.
- **Concrete Fix:** Add a lateral slot offset `((i - 3) // 2) * 22.0` to `siege_garrison`, check river barriers in `_blocked_by_barrier`, allow siege artillery to target intact gates/walls directly, and scale artillery shots by gun count (`u.tpl.extras` or `2..4` pieces) rather than crew count.

---

#### 🟡 M11. `edu_parser.py` Classifies All Bodyguard-Capable Heavy Cavalry (`general_unit` Attribute) Exclusively as `"general"`, Excluding Them From `"shock_cav"`
- **Citations:**
  - [`sim/edu_parser.py:135-146, 241-243, 258-261, 294-298`](file:///home/tomek/medieval2-docker/sim/edu_parser.py#L135-L146)
  - [`workspace/rl_battle_ai.lua:41`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L41)
- **Mechanism:**
  1. In M2TW's `export_descr_unit.txt`, many main-line heavy cavalry units (e.g., Mailed Knights, Feudal Knights, Chivalric Knights, Royal Mamluks) include `general_unit` in their `attributes` line because campaign generals can use them as bodyguards.
  2. Because `UnitTemplate.role` ([`sim/edu_parser.py:138-141`](file:///home/tomek/medieval2-docker/sim/edu_parser.py#L138-L141)) checks `if self.general_unit: return "general"` *before* `if self.is_cav: return "shock_cav"`, all of those heavy knight templates are locked into the `"general"` pool (capped at 1 per army!) and completely excluded from `"shock_cav"`. Meanwhile, [`rl_battle_ai.lua:41`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L41) checks `(bu.isGeneral == 1) or (u.character ~= nil)`, classifying non-general knight units as `"shock_cav"`.
  3. In [`sim/edu_parser.py:296`](file:///home/tomek/medieval2-docker/sim/edu_parser.py#L296), `stat_fire_delay` is parsed as `float(vals[0])` without multiplying by `0.1` (unlike `pri_delay` at line 259).
  4. In [`sim/edu_parser.py:258-261`](file:///home/tomek/medieval2-docker/sim/edu_parser.py#L258-L261), 11-field `stat_pri` lines where `vals[9]` is a sound/delay modifier string raise `ValueError` inside the `try` block.
- **Concrete Fix:** Allow `general_unit` cavalry templates without `no_custom` to also populate the `"shock_cav"` pool in `build_faction_pools` (setting `u.role = "shock_cav"` when spawned in a `"shock_cav"` slot), parse the numeric delay from `vals[-1]` in `stat_pri`, and scale `stat_fire_delay` by `0.1`.

---

#### 🟡 M12. Out-of-Ammo Foot Archers in `sim/vanilla_ai.py` Fall Through to Frontline Infantry Charge Logic
- **Citations:**
  - [`sim/vanilla_ai.py:118-130, 175-187, 263-297, 333-341`](file:///home/tomek/medieval2-docker/sim/vanilla_ai.py#L263-L297)
- **Mechanism:**
  In `step_vanilla_ai` (`_step_siege_vanilla`, `_step_river_vanilla`, and open-field), the `if u.role == "missile_foot":` and `if u.role == "missile_cav":` blocks only `continue` when `if u.ammo > 0:`. As soon as a missile unit fires its last arrow (`u.ammo == 0`), execution falls through to the bottom of the loop (`d <= 45.0 -> attack`, else march straight at the nearest enemy), causing unarmored archers to suicide-charge into enemy heavy infantry instead of holding behind the infantry line.
- **Concrete Fix:** When `u.role == "missile_foot"` and `u.ammo <= 0`, hold position behind the friendly infantry centroid `(icx - fx * 30.0, icy - fy * 30.0)` unless enemies close within `25.0m`.

---

### 🟢 NITS & CLEANUP ITEMS

#### 🟢 N1. Undefined Global `curEd` & Draw Misclassification as `decisive_loss` in `workspace/rl_battle_ai.lua`
- **Citations:** [`workspace/rl_battle_ai.lua:564-566, 756, 760, 948`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L756)
- **Mechanism:**
  - Lines 756 and 760 assign `ed = curEd`, where `curEd` is an undefined global variable (always `nil`).
  - Line 948 computes `local decisiveLoss = (won - 1 <= 0) and (not S.timedOut)`, which evaluates to `true` on a draw (`won == 1`, `result == 0`), applying the `-170.0` decisive loss penalty to draws (`should be (won - 1 < 0)`).
  - Lines 564–566 initialize `ownStart = 1, enStart = 1` before adding `r.start`, making total start counts off by `+1`.

#### 🟢 N2. Dead Constants, Unused Variables, and Hardcoded `if False` in `sarsa_trainer.py`, `sim_train.py`, and `rl_battle_ai.lua`
- **Citations:**
  - [`workspace/trainer/sarsa_trainer.py:10-12`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L10-L12) (`GAMMA, LAMBDA, ALPHA, W_GLOB, W_MAP, DMG_BONUS` unused)
  - [`sim/sim_train.py:52, 87`](file:///home/tomek/medieval2-docker/sim/sim_train.py#L87) (`scale=1.0` unused; `if False and bid % 10 == 0` disables LLM judge)
  - [`workspace/rl_battle_ai.lua:189-193`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L189-L193) (`globalSparse` unused)
  - [`sim/combat.py:65`](file:///home/tomek/medieval2-docker/sim/combat.py#L65) (`SimUnit.last_feats` unused)

---

## Part 2: Design Document Drift Report (`egm-drift-report`)

## `m2tw_rl_ai_design_doc.md`

- **Algorithm & Architecture Replaced (Linear SARSA($\lambda$) $\to$ Hierarchical KBANN Elman RNNs + Full BPTT Policy Gradient):**
  - [`m2tw_rl_ai_design_doc.md:4, 146, 218-227`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L218-L227) specifies a flat Linear SARSA($\lambda$) policy $Q(s, a) = \mathbf{w}_a^\top \boldsymbol{\phi}(s)$ with eligibility traces $\mathbf{E}_{\text{role}} \in \mathbb{R}^{13 \times 430}$ across 4 unit archetypes (`MeleeInf`, `SpearInf`, `Missile`, `Cavalry`).
  - Reality ([`spec.py:4-20`](file:///home/tomek/medieval2-docker/workspace/trainer/spec.py#L4-L20), [`kbann_init.py:37-180`](file:///home/tomek/medieval2-docker/workspace/trainer/kbann_init.py#L37-L180), [`sarsa_trainer.py:193-372`](file:///home/tomek/medieval2-docker/workspace/trainer/sarsa_trainer.py#L193-L372)): The codebase implements a two-tier hierarchical Recurrent Neural Network architecture initialized with KBANN symbolic rules and trained via Full BPTT Policy Gradient:
    1. A **Commander RNN** (`293 inputs + 48 hidden -> 10 unnamed orders`, with neurons `0..9` frozen).
    2. **7 Per-Role Unit RNNs** (`462 inputs + 32 hidden -> 13 primitive actions` for `line_inf`, `spear_inf`, `missile_foot`, `shock_cav`, `missile_cav`, `general`, `artillery`, with neurons `0..15` frozen).
- **Feature Dimension & Layout Drift (`430` $\to$ `462` Unit Features + `293` Commander Features):**
  - [`m2tw_rl_ai_design_doc.md:68, 95, 145`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L145) specifies `20 Global + 26 Local + 384 Grid = 430` features.
  - Reality ([`spec.py:6-11`](file:///home/tomek/medieval2-docker/workspace/trainer/spec.py#L6-L11)): `50 Global` (`20 army/map + 10 global order + 10 detachment order + 10 lead line_inf order`, scaled by `3.0`) + `28 Local` (`26 unit/local + 2 infantry centroid offset`) + `384 Grid` = `462` features.
  - Within Local Features ([`m2tw_rl_ai_design_doc.md:99-119`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L99-L119) vs [`sim/rl_env.py:178-208`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L178-L208)): Only 9 capability slots `l[0..8]` are allocated instead of 10 (`l[6]` and `l[8]` are permanently `0.0`; `is_siege_or_handler`, `class_skirmish`, and `has_bonus_vs_cav` are omitted).
  - Within Global Features ([`m2tw_rl_ai_design_doc.md:86-91`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L86-L91) vs [`sim/rl_env.py:137-154`](file:///home/tomek/medieval2-docker/sim/rl_env.py#L137-L154) & [`rl_battle_ai.lua:179-185`](file:///home/tomek/medieval2-docker/workspace/rl_battle_ai.lua#L179-L185)): `own_morale_avg` and `enemy_morale_avg` compute the non-routing fraction rather than mean morale level; `ground_control_score` computes global strength ratio `(oS - eS) / tot * 0.5 + 0.5` rather than central tile control; `g[18..20]` are hardcoded `0, 0, 0` in `rl_battle_ai.lua`.
- **Sequential Intra-Tick Sector Balance Updates Unimplemented:**
  - [`m2tw_rl_ai_design_doc.md:178-182`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L178-L182) states that as soon as Unit $i$ selects a movement offset $(\Delta x, \Delta y)$, its projected position is immediately reflected in `left_wing_balance`, `center_str_balance`, `right_wing_balance`, and `reserve_ratio` before Unit $i+1$ acts. Neither `rl_env.py` nor `rl_battle_ai.lua` updates `g` between unit decisions within a tick.
- **Decision Cadence, Action Selection, & Curriculum Drift:**
  - [`m2tw_rl_ai_design_doc.md:47-50, 185-214`](file:///home/tomek/medieval2-docker/m2tw_rl_ai_design_doc.md#L185-L214) specifies `3.0s` synchronous decisions for all units, $\varepsilon$-greedy exploration, a 3-phase Docker/Wine curriculum, and step rewards including $w_{\text{glob}}$ and $w_{\text{map}}$ with $\pm 50.0$ terminal reward.
  - Reality: `rl_battle_ai.lua` uses an `18.0s` commitment window with a `4 units/sec` round-robin queue at `50x` speed (`rl_env.py` uses `6.0s` unit / `12.0s` commander windows), softmax sampling, a 4-phase standalone Python simulator (`sim_train.py` adding Phase 3 `rl_vs_vanilla_disadvantaged`), `W_GLOB = 0.0`, `W_MAP = 0.0`, `TERMINAL = 100.0`, and `DECISIVE_BONUS = 340.0`.

---

## Traceability Matrix (Goldfish Findings $\to$ Consolidated Report)

| Consolidated ID | Severity | `egm-codebase-details` | `egm-design` | `egm-zoom-out` | Primary Target File(s) |
|:---|:---|:---|:---|:---|:---|
| **B1** | 🔴 Blocker | 2.1, 2.2 | 1.1 | 2.1 | `sim/sim_train.py:164`, `sarsa_trainer.py:120-174, 249-251` |
| **B2** | 🔴 Blocker | 1.1, 3.2 | 2.2, 3.1 | 1.2 | `sim/rl_env.py:387-406, 455-470`, `rl_battle_ai.lua:589-691`, `kbann_init.py:173-175` |
| **B3** | 🔴 Blocker | 2.3 | 1.2 | 2.2 | `sim/rl_env.py:462, 470`, `sarsa_trainer.py:312-366`, `rl_battle_ai.lua:814-830, 930-939` |
| **B4** | 🔴 Blocker | 1.2, 3.4 | 2.1 | 1.3 | `sim/combat.py:825-839`, `sim/rl_env.py:196-197`, `rl_battle_ai.lua:20, 213, 525` |
| **B5** | 🔴 Blocker | 1.2 | 2.4 | 1.3 | `sim/rl_env.py:185-190`, `rl_battle_ai.lua:203-208, 223-225` |
| **B6** | 🔴 Blocker | 1.4 | — | — | `rl_battle_ai.lua:123-134, 232-246`, `sim/rl_env.py:158-165, 213-231` |
| **B7** | 🔴 Blocker | — | 1.3 | 3.1 | `sim/sim_train.py:148-206`, `sarsa_trainer.py:200-274, 312-372` |
| **B8** | 🔴 Blocker | 4.1 | 3.3 | 3.3 | `sim/combat.py:146-160, 457-485, 514-538` |
| **M1** | 🟡 Major | 3.1 | 1.3, 2.3 | 1.3, 2.4 | `sim/rl_env.py:25`, `sarsa_trainer.py:108, 255, 354`, `rl_battle_ai.lua:21, 293` |
| **M2** | 🟡 Major | 3.5 | 2.6 | 1.3 | `sim/rl_env.py:20-22, 486`, `rl_battle_ai.lua:8-11, 845-863` |
| **M3** | 🟡 Major | — | 1.3, 1.4 | — | `sarsa_trainer.py:189, 231-239, 270-273, 335-343` |
| **M4** | 🟡 Major | 1.3 | — | — | `kbann_init.py:138-145`, `sim/rl_env.py:171`, `rl_battle_ai.lua:199` |
| **M5** | 🟡 Major | 1.5 | 2.5 | 3.3 | `sim/rl_env.py:294-367`, `rl_battle_ai.lua:427-458` |
| **M6** | 🟡 Major | 3.3 | 2.4 | 1.3 | `sim/rl_env.py:54-76`, `rl_battle_ai.lua:87-103` |
| **M7** | 🟡 Major | — | — | 3.2 | `rl_battle_ai.lua:608-613, 764-771, 836-841` |
| **M8** | 🟡 Major | 4.3 | 3.2 | — | `sim/combat.py:326-358` |
| **M9** | 🟡 Major | 4.4 | — | 3.3 | `sim/combat.py:481-485, 594-628` |
| **M10** | 🟡 Major | 4.2 | 3.3, 3.4 | 3.3 | `sim/combat.py:223-230, 373-384, 439-451, 491, 663-720` |
| **M11** | 🟡 Major | 4.5 | 3.4 | — | `sim/edu_parser.py:135-146, 241-243, 258-261, 296` |
| **M12** | 🟡 Major | 4.6 | — | — | `sim/vanilla_ai.py:118-130, 175-187, 263-297` |
| **N1** | 🟢 Nit | 1.3 | 2.7 | 2.3 | `rl_battle_ai.lua:564-566, 756, 760, 948` |
| **N2** | 🟢 Nit | — | — | 2.3 | `sarsa_trainer.py:10-12`, `sim/sim_train.py:52, 87`, `rl_battle_ai.lua:189` |
| **Drift** | 🟡 Major | Section 5 | — | 1.1 | `m2tw_rl_ai_design_doc.md:1-274` |
