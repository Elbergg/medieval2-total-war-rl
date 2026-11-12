# Technical Specifications — Medieval II: Total War RL Battle AI

This document provides complete technical, mathematical, and algorithmic specifications for the reinforcement learning battle AI operating inside *Medieval II: Total War* (v1.52 Steam) via M2TWEOP (Medieval II Total War Engine Overhaul Project) v3.3.1.

---

## 1. System Topology & Architecture

```
+---------------------------------------------------------------------------------------+
| HOST LINUX ENVIRONMENT                                                               |
|                                                                                       |
|  +-------------------------------------+     +-------------------------------------+  |
|  | Container: medieval2-ubuntu         |     | Container: medieval2-trainer        |  |
|  | - Ubuntu 24.04 (i386 + amd64)       |     | - Python 3.12 (NumPy, GenAI)        |  |
|  | - Wine32 + Xvfb (:1024x768x24)      |     | - orchestrator.py                   |  |
|  | - medieval2.exe (4GB LAA Patched)   |     | - sarsa_trainer.py                  |  |
|  | - M2TWEOP v3.3.1 (Lua Plugin)       |     | - gemini_judge.py                   |  |
|  | - rl_battle_ai.lua                  |     | - scenario_generator.py             |  |
|  +------------------+------------------+     +------------------+------------------+  |
|                     |                                           |                     |
|                     +---------------------+---------------------+                     |
|                                           |                                           |
|                              SHARED IPC VOLUME: workspace/                            |
|                              - scenario.json                                          |
|                              - state.json                                             |
|                              - weights_latest.json / weights_frozen.json              |
|                              - episodes/episode_{N}.json                              |
|                              - replays/replay_{N}.json & .mp4                         |
|                              - rl_heartbeat.txt                                       |
+---------------------------------------------------------------------------------------+
```

---

## 2. State Space (430 Dimensions)

Each controlled unit observes the battlefield from its egocentric frame of reference ($f_{\text{forward}}, f_{\text{right}}$). The feature vector $\phi \in \mathbb{R}^{430}$ is sparse and composed of three concatenated segments:

$$\phi = [\phi_{\text{global}} \; (20) \quad \Vert \quad \phi_{\text{local}} \; (26) \quad \Vert \quad \phi_{\text{spatial}} \; (384)]$$

### 2.1 Global Features ($N_G = 20$, indices 0–19)
Normalized macroscopic battlefield indicators:
1. `time_elapsed`: $\min(t / 600, 1.0)$
2. `own_strength_ratio`: $\frac{S_{\text{own}}}{S_{\text{own}} + S_{\text{enemy}}}$ (sum of alive soldiers $\times$ edu unit value)
3. `en_strength_ratio`: $\frac{S_{\text{enemy}}}{S_{\text{own}} + S_{\text{enemy}}}$
4. `own_general_alive`: binary $\{0, 1\}$
5. `en_general_alive`: binary $\{0, 1\}$
6–11. `own_role_counts`: normalized count for `[line_inf, spear_inf, missile_foot, shock_cav, missile_cav, general]`
12–17. `en_role_counts`: normalized counts for enemy roles
18. `friendly_losses_delta`: normalized casualties in last step
19. `enemy_losses_delta`: normalized casualties inflicted in last step

### 2.2 Local Unit Features ($N_L = 26$, indices 20–45)
Unit-specific state:
- Unit role one-hot encoding (6 dims: `line_inf`, `spear_inf`, `missile_foot`, `shock_cav`, `missile_cav`, `general`)
- Soldier count ratio: $N_{\text{current}} / N_{\text{start}}$
- Ammunition ratio: $\text{ammo} / \text{ammo}_{\text{max}}$
- Fatigue level: normalized $[0, 1]$
- Morale status: routed / shaken / steady / eager $[0, 1]$
- Engagement status: in melee $\{0, 1\}$, under fire $\{0, 1\}$
- Nearest enemy distance & relative bearing: $[\cos \theta, \sin \theta, \min(d / 300, 1.0)]$
- Local friendly density & enemy density (within 80m radius)

### 2.3 Egocentric Spatial Grid ($N_S = 384$, indices 46–429)
An $8 \times 8$ grid centered on the unit, aligned with its heading ($x \in [-100\text{m}, +100\text{m}]$, $y \in [-100\text{m}, +100\text{m}]$, resolution $25\text{m} \times 25\text{m}$ per cell). Each cell contains 6 binary channels:
1. Friendly unit present
2. Enemy unit present
3. Unit holding position
4. Unit engaging in melee
5. Unit firing missiles
6. Unit routing / fleeing

$$\text{Total Spatial Features} = 8 \times 8 \times 6 = 384$$

---

## 3. Action Space (13 Discrete Actions)

Each unit independently selects an action $a \in \{0, \dots, 12\}$ every decision tick ($\Delta t = 3.0$ in-game seconds):

| Action ID | Name | Semantic Description |
|:---|:---|:---|
| **0** | `MOVE_N` | Move 50m Forward along unit heading |
| **1** | `MOVE_NE` | Move 50m Forward-Right |
| **2** | `MOVE_E` | Move 50m Right (Flank Right) |
| **3** | `MOVE_SE` | Move 50m Rear-Right |
| **4** | `MOVE_S` | Move 50m Backward (Fall back) |
| **5** | `MOVE_SW` | Move 50m Rear-Left |
| **6** | `MOVE_W` | Move 50m Left (Flank Left) |
| **7** | `MOVE_NW` | Move 50m Forward-Left |
| **8** | `HALT` | Stop movement, brace formation, hold line |
| **9** | `ATTACK_FRONT` | Attack nearest enemy in front quadrant ($[-45^\circ, +45^\circ]$) |
| **10** | `ATTACK_RIGHT` | Attack nearest enemy in right quadrant ($[+45^\circ, +135^\circ]$) |
| **11** | `ATTACK_REAR` | Attack nearest enemy in rear quadrant ($[+135^\circ, -135^\circ]$) |
| **12** | `ATTACK_LEFT` | Attack nearest enemy in left quadrant ($[-45^\circ, -135^\circ]$) |

---

## 4. Policy & Learning Algorithm

### 4.1 Value Function Approximation
Linear function approximation parameterized by weight matrices $W_{\text{role}} \in \mathbb{R}^{|A| \times |\Phi|} = \mathbb{R}^{13 \times 430}$:

$$Q(\phi, a; W_r) = \sum_{i \in \text{active}(\phi)} W_r[a, i] \cdot \phi[i]$$

$\epsilon$-greedy exploration:
$$a = \begin{cases} \text{random}(0, 12) & \text{with probability } \epsilon \\ \arg\max_{a'} Q(\phi, a'; W_r) & \text{otherwise} \end{cases}$$

### 4.2 SARSA($\lambda$) True Online Eligibility Traces
- Discount factor: $\gamma = 0.99$
- Trace decay parameter: $\lambda = 0.8$
- Learning rate: $\alpha = 0.01$
- TD Error: $\delta_t = R_{t+1} + \gamma Q(\phi_{t+1}, a_{t+1}) - Q(\phi_t, a_t)$
- Trace update: $e_t \leftarrow \gamma \lambda e_{t-1} + \phi_t$
- Weight update: $W \leftarrow W + \alpha \delta_t e_t$

### 4.3 Reward Formulation
The transition reward $R_t$ blends unit-level tactical feedback and macroscopic battle outcomes:
$$R_t = \Delta K_{\text{kills}} - 1.5 \Delta L_{\text{losses}} + 2.0 \Delta G_{\text{ground}} + R_{\text{terminal}}$$
- $\Delta K$: Enemy soldiers killed by this unit
- $\Delta L$: Friendly soldiers lost by this unit
- $\Delta G$: Change in relative army strength ratio
- $R_{\text{terminal}}$: $+10.0$ for battle victory, $-10.0$ for defeat, $+5.0$ if enemy general killed, $-5.0$ if friendly general killed

---

## 5. Curriculum Progression

1. **Phase 1: Imitation Learning Baseline (Battles 1–50)**:
   - Mode: `vanilla_vs_vanilla`
   - Both armies driven by native Total War engine AI.
   - RL agent passively observes Side 0, extracting trajectory features and warm-starting weights via behavioral displacement matching.
2. **Phase 2: RL vs. Vanilla AI (Battle 51+ until winrate $\ge 75\%$)**:
   - Mode: `rl_vs_vanilla`
   - Our RL agent actively controls Side 0; native engine AI controls Side 1.
   - Evaluated on a 50-battle rolling window. When rolling win rate exceeds 75%, weights freeze into `weights_frozen.json` and advance to Phase 3.
3. **Phase 3: Self-Play (RL vs. Frozen Checkpoint)**:
   - Mode: `rl_vs_frozen`
   - Learner plays against frozen past versions of itself. When rolling win rate exceeds 55% over 100 battles, the frozen opponent is updated.

---

## 6. Execution & Engine Modifications

### 6.1 4GB LAA (Large Address Aware) Patch
Medieval II is a 32-bit executable. Wine restricts 32-bit binaries to 2GB address space unless the `IMAGE_FILE_LARGE_ADDRESS_AWARE` flag (`0x0020`) is set in the PE header Characteristics field.
- Default: `Characteristics = 0x10f` (2GB max virtual memory)
- Patched: `Characteristics = 0x12f` (4GB max virtual memory)
- Prevents out-of-memory DirectX surface allocation crashes (`"A fatal graphics error has occurred"`).

### 6.2 Acceleration & Early Stopping
- Simulation Speed: `b.battleSpeed = 150.0`
- Decision Frequency: 3.0 in-game seconds (pumped at 0.05s DirectX hook)
- Early Stop Conditions:
  - Battle reaches 300 in-game seconds
  - Either army suffers an 80%+ casualty rout
- Cycle Duration: **~20–22 wall-clock seconds per entire battle**.
