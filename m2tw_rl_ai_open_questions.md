# M2TW Battle AI: RL + LLM Judge — Open Design Questions

> [!NOTE]
> This document captures all open architectural, algorithmic, and practical questions we need to settle before implementing the **Decentralized Rule-Learning Battle AI (SARSA / Q-Learning + LLM Judge)** via **M2TWEOP**.

---

## 1. Engine & Mod Target Environment

1. **Executable / Engine Target:**
   - M2TWEOP hooks the 32-bit `medieval2.exe` / `kingdoms.exe` via `d3d9.dll`, while your root folder currently also has 64-bit **M2EX** files.
   - **Question:** Which mod folder (or vanilla campaign/custom battle setup) should we install `M2TWEOP-3.3.1` into as our dedicated AI sandbox?
   - **answer** we will have a docker runtime for this
2. **Battle Type Scope (Phase 1):**
   - **Question:** Are we starting strictly with **Open Field Battles** (no walls/rivers) so the spatial coordinate frame is clean, and expanding to River/Siege battles later?
   - **answer** the battkle conditions will  be randomized

---

## 2. Training Loop & Opponent Setup

3. **Who Does the RL Agent Play Against?**
   - *Option A:* **RL AI vs. Vanilla GTA AI** (automated fast training—we can even run battles at $3\times$–$6\times$ game speed).
   - *Option B:* **RL AI Self-Play** (Side 1 RL vs. Side 2 RL, both updating or competing against frozen checkpoints).
   - *Option C:* **RL AI vs. Human (You)** (live online learning + post-battle LLM review).
   - **Question:** Which training mode (or combination) do you want as the primary loop?
   - **answer** for now we will want it to play against vanilla ai, also the initial starting position will be the vanilla ai, after it starts winning to much we will have it play against a version of itself, which will be frozen and updater per some cycles
4. **Army Compositions During Training:**
   - **Question:** Should we train on a **fixed balanced army preset** first (e.g., 6 Inf, 4 Spear, 4 Missile, 4 Cav, 1 General) to speed up convergence, or randomized army compositions from the start?
   - **answer** we will want some more or less standard compositions but with some randomness so there is a small chance for a full cavalary battle or something but overall they will be standardized

---

## 3. State Representation ($\mathbf{g}_{\text{battle}}$ & $\mathbf{l}_i$)

5. **Global Battle Context ($\mathbf{g}_{\text{battle}}$):**
   - Currently proposed:
     - Sector strength ratios (*Left Wing, Center, Right Wing, Reserve/Rear*)
     - Capability ratios (*Ranged dominance, Cavalry/Mobility dominance*)
     - Battle momentum (*Total soldiers alive on both sides, Casualty ratio, $\%$ of friendly units engaged in melee*)
   - **Question:** Are there any additional global battlefield metrics you want every unit to know (e.g., general alive/dead, time elapsed, overall morale/fatigue ratio, terrain elevation slope)?
   - **answer** uyeah we want general alive dead time elapsed, morale, maybe also something about capturing ovwerall ground on the battlefield and explicitly something about holding settlements
6. **Local Unit Sensors ($\mathbf{l}_i$):**
   - Own unit category/class & combat stats (`attack`, `charge`, `missileAttack`, `range`, `ammo%`, `defence/armour/shield`, `moveSpeed`, `mass`, capabilities like `canPhalanx`/`hasBonusVsCav`, **excluding target unit stats**).
   - Current sector $(x, y)$ relative to army center, soldier count & casualties, morale, fatigue, `isEngaged`, `isUnderFire`.
   - Local proximity counts + a **low-res 2D map covering $200\text{m} \times 200\text{m}$ around the unit** with three values (`0 = empty`, `1 = friendly`, `2 = enemy`) so it can learn flanking lanes and encirclements.

---

## 4. Primitive Action Space & Rule Discovery

7. **Primitive Action Format ($A$):**
   - Since we want behaviors like *"Join Mainline"* or *"Flank"* to be **learned** rather than hardcoded, what primitive output should a rule produce?
   - *Option A (Relative Grid/Vector):* Move to relative offset $(\Delta x, \Delta y)$ in the army's coordinate frame + stance (`Walk/Run`, `Engage/Hold/Skirmish`).
   - *Option B (Target-Centric Primitives):* Pick a sector (`Far-Left, Left, Center, Right, Far-Right`) $\times$ depth (`Rear, Line, Front, Deep`) $\times$ behavior mode (`Move-Oriented, Attack-Nearest-In-Sector, Fire-At-Range, Brace`).
   - **Question:** Which primitive action representation feels most natural for the rules to output?
   - **answer** we would want to be quite primitve so move on the offset, or attack a unit, obv the default attack mode for both units
8. **How Are New Rules Generated & Updated?**
   - *Option A (Fixed Feature Basis + Linear SARSA($\lambda$)):* Automatic tile-coding / decision-tree splits over $(\mathbf{g}_{\text{battle}}, \mathbf{l}_i)$ where SARSA learns the weight $w_{r, a}$ of every condition-action rule.
   - *Option B (Evolutionary Rule Discovery / XCS + SARSA):* Rules are explicit `IF [feature_min <= x <= feature_max] THEN [Action]` clauses; SARSA updates their $Q$-values during battle, and between battles poorly performing rules are pruned and mutated/proposed (potentially even seeded by the LLM).
   - **Question:** Do you prefer **Linear SARSA over automatic feature rules (Option A)** or **explicit evolving `IF-THEN` rule tables (Option B)**?
   - **answer** lets have linear sarsa
9. **Decision Tick Frequency:**
   - **Question:** How often should units re-evaluate their rules during battle? (e.g., every $3\text{–}5$ seconds of in-game battle time, or event-driven when local state changes significantly?)
   - **answer** every 3 seconds

---

## 5. The Cost / Reward Function & LLM Judge

10. **LLM Judge Provider & Model:**
    - **Question:** Which LLM backend should the Python judge service use?
      - Cloud API (*Gemini, OpenAI, Anthropic*)?
      - Local model (*Ollama / vLLM / LM Studio*)?
    - **answer** we will use gemini flash 3.8, i will provide the api
11. **When Does the LLM Judge Run?**
    - *Option A (Post-Battle / Offline Episodic Judge):* During the battle, M2TWEOP logs tactical snapshots every $10\text{–}15\text{s}$. Immediately when the battle ends (or asynchronously in the background), the LLM Judge scores each $15\text{s}$ window and each rule that fired, and updates the $Q$-table before the next battle starts. *(Zero in-battle lag, low token cost).*
    - *Option B (Live Async Judge During Battle):* A background Python daemon scores snapshots every $15\text{–}20\text{s}$ while the battle is running and streams reward penalties/bonuses back to Lua in real time.
    - **Question:** Do you want **Post-Battle Offline Judging (Option A)**, **Live Async Judging (Option B)**, or both?
    - **answer** we want to have a sort of history of the battle saved maybe in some top down format similar to what i proposed with the 2d map but for this format we also want to remember what kind of units there were and stuff like that, but bascially we want a de facto video of the battle, but we have to whatch out for the output not to be too big, and llm will evaluate this but it will not be made on every run since llms are slow, we will do it every 100 runs and have the pnalties be very large
12. **What Exact Criteria Should the LLM Judge Score?**
    - Currently proposed rubric for $R_{\text{LLM}} \in [-1, +1]$:
      - **Anti-Cheese / Fairness:** Penalize red-line corner camping, endless kiting, 1-rank spaghetti lines, or stacking 10 units on the same coordinate.
      - **Human-Like / Historical Cohesion:** Reward maintaining a recognizable battle line, keeping reserves until needed, protecting flanks, and coordinated wing maneuvers.
    - **Question:** Are there specific "human-like" behaviors you want the LLM Judge to reward, or specific AI habits you hate and want heavily penalized?
    - **answer** for right anti cheese/fairness and human like hsitorical cohesion will be allright
13. **Reward Weight Balance ($\alpha, \gamma, \beta$):**
    - Total reward: $R_{\text{total}} = R_{\text{combat}} + \beta \cdot R_{\text{LLM}}$.
    - **Question:** If a tactic wins the battle ($R_{\text{combat}} > 0$) but the LLM Judge flags it as pure cheese/exploit ($R_{\text{LLM}} = -1$), should the LLM Judge have **veto power** (forcing the net reward negative so the rule is unlearned), or just a moderate penalty?
    - **already answered**

---

## 6. SARSA($\lambda$) Implementation & Where Training Takes Place

14. **Where Does the SARSA($\lambda$) Math & Weight Update Run?**
    
    - *Option A (In-Process LuaJIT inside M2TWEOP):*
      - LuaJIT (`luaPluginScript.lua`) extracts features $\boldsymbol{\phi}(\mathbf{g}_{\text{battle}}, \mathbf{l}_i)$, computes $Q(s, a) = \mathbf{w}_a^\top \boldsymbol{\phi}(s)$, picks actions every $3\text{s}$, updates eligibility traces $\mathbf{e}_a$ and weights $\mathbf{w}_a$ in memory, and writes `weights.json` + `battle_replay_XXXX.json` to `/home/tomek/workspace/` at the end of each battle.
      - Python only runs outside the battle loop (to generate the next randomized battle file, track win rates, manage frozen self-play checkpoints, and run the Gemini 3.8 Flash Judge every 100 runs).
      - *Pros:* Zero socket/IPC latency during battle ticks; LuaJIT is super fast at 1D array dot products.
    - *Option B (Python Training Server + Thin Lua Telemetry/Actuator Client):*
      - Lua inside M2TWEOP only reads game memory every $3\text{s}$, sends the raw state JSON over a localhost socket/pipe to a Python process (running inside the Docker container), and executes the returned unit commands.
      - Python (NumPy/PyTorch) computes the 200m 2D grids, runs Linear SARSA($\lambda$), logs TensorBoard/W&B metrics, and calls Gemini every 100 battles.
      - *Pros:* All math, feature engineering, checkpointing, and LLM penalty updates live in one language (Python) with NumPy vectorization.
    - **Question:** Do you want **SARSA($\lambda$) in LuaJIT + Orchestrator in Python (Option A)**, or **SARSA($\lambda$) + Orchestrator all in Python with Lua as a thin bridge (Option B)**?
    - **answer:** we want it to happen after the battle in python code

15. **Where Does the Orchestrator & Training Loop Run (Host vs. Inside Docker)?**
    
    - Inside `/home/tomek/medieval2-docker`, we have the `medieval2-ubuntu` container with `/home/tomek/workspace` bind-mounted from the host.
    - *Setup:*
      1. A **Battle Runner Loop** generates a randomized `battle_script` / `descr_battle.txt` (maps, weather, time of day, standardized + slightly randomized armies), launches `medieval2.exe` headless under `xvfb-run` + Wine + M2TWEOP inside Docker, sets battle speed to maximum, and exits/restarts (or reloads) when the battle finishes.
      2. Shared volume `/home/tomek/medieval2-docker/workspace` holds `weights_latest.npy/json`, `weights_frozen_opponent.npy/json`, `replays/`, and training logs.
    - **Question:** Should the Python Training & Gemini Judge service run **inside the `medieval2-ubuntu` Docker container** alongside the game, or **on your Host machine** watching the shared `/home/tomek/medieval2-docker/workspace` folder? (And do you ever plan to spin up **multiple Docker containers in parallel** to run e.g. 2–4 battles simultaneously?)
    - **answer:** it will ta;l in a spearate container talking to the medieval 2

16. **How Should We "Warm-Start" Linear SARSA from Vanilla AI (From Your Answer #3)?**
    - **Chosen Method — Way 2 (3-Second Physical Displacement / Inverse Dynamics):**
      - Even though Vanilla AI's internal FSM states live in a different space from our primitive actions, over any 3-second tick $[t, t+3\text{s}]$ a Vanilla-controlled unit physically moves by a 2D displacement vector $(\Delta x_{\text{actual}}, \Delta y_{\text{actual}}) = (x_{t+3} - x_t, y_{t+3} - y_t)$ (or engages an enemy unit).
      - Since our SARSA primitive action space is also 2D offset vectors $(\Delta x, \Delta y)$ + `Attack`, we map the actual 3-second physical displacement of Vanilla units directly into our primitive action space and penalize divergence $-\|a_t - a_{\text{displacement}}\|^2$ during the initial Vanilla reproduction phase, before decaying the imitation penalty and handing control over to Combat + Gemini Judge rewards.
