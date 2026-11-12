# M2TW RL Battle AI — runbook

```
./setup_game.sh                # hardlinked ./game copy + M2TWEOP 3.3.1 (host Steam install untouched)
docker compose up -d --build   # medieval2-ubuntu (Wine+Xvfb) + medieval2-trainer (Python)
./run_headless.sh              # launch medieval2.exe with M2TWEOP + Lua RL plugin
```

- `workspace/rl_battle_ai.lua` — in-game inference + logging (copy into `game/eopData/eopScripts/`, hooked from `luaPluginScript.lua`)
- `workspace/trainer/` — `orchestrator.py` (curriculum loop), `sarsa_trainer.py`, `scenario_generator.py` (random factions/weather/armies), `gemini_judge.py`, `spec.py` (shared constants)
- IPC (`workspace/`): `scenario.json` -> game, `weights_*.json` -> game, `episodes/` + `replays/` -> trainer
- Set `GEMINI_API_KEY` in the environment before `docker compose up` to enable the every-100-battles judge.
