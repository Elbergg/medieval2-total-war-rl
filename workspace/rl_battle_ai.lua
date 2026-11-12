-- rl_battle_ai.lua : decentralized unit-level Linear SARSA(lambda) *inference + logging* inside M2TWEOP.
-- All learning happens post-battle in Python. Spec constants must match trainer/spec.py.
-- NOTE: written against the M2TWEOP 3.3.1 Lua API; calls marked [VERIFY] must be confirmed in the first in-game smoke test.
local json = require('helpers/dkjson')

local M = {}
local WS = "Z:/home/tomek/workspace/"          -- Wine drive Z: = Linux root; shared volume with trainer
local TICK = 3.0
local NA, NG, NL, GRID, GCH = 13, 20, 26, 8, 6
local NF = NG + NL + GRID * GRID * GCH           -- 430
local ROLES = { "line_inf", "spear_inf", "missile_foot", "shock_cav", "missile_cav", "general" }
local MOVES = { [0] = { 0, 60 }, { -45, 45 }, { 45, 45 }, { -60, 0 }, { 60, 0 }, { -45, -45 }, { 45, -45 }, { 0, -60 }, { 0, 0 } }
local MORALE_ROUTING = 0                         -- [VERIFY] moraleLevel value meaning "routing"
local EPS = { learner = 0.10, frozen = 0.02 }

local S = nil  -- per-battle state

local function readfile(p) local f = io.open(p, "r"); if not f then return nil end; local s = f:read("*a"); f:close(); return s end
local function writefile(p, s) local f = io.open(p, "w"); if f then f:write(s); f:close() end end
local function clamp(x, a, b) if x < a then return a elseif x > b then return b end return x end

-- ---------- unit classification ----------
local function lower(v) return string.lower(tostring(v)) end
local function classify(u)
    local e = u.eduEntry
    local cat, cls = lower(e.category), lower(e.class)           -- [VERIFY] string or int; int fallback below
    local isCav = cat:find("cav") ~= nil or cat == "1"
    local isMissile = e.primaryStats.isMissile and true or false
    local isSpear = cls:find("spear") ~= nil or cls == "3"
    local isHeavy = cls:find("heavy") ~= nil or cls == "1"
    local isGeneral = u.character ~= nil
    local role
    if isGeneral then role = "general"
    elseif isCav and isMissile then role = "missile_cav"
    elseif isCav then role = "shock_cav"
    elseif isMissile then role = "missile_foot"
    elseif isSpear then role = "spear_inf"
    else role = "line_inf" end
    return role, { cav = isCav, missile = isMissile, spear = isSpear, heavy = isHeavy, e = e }
end

local function roleIndex(role) for i, r in ipairs(ROLES) do if r == role then return i end end end

-- ---------- battle roster ----------
local function collect(battle)
    local sides = { {}, {} }
    for si = 0, 1 do
        local side = battle.sides[si]
        for ai = 0, side.battleArmyNum - 1 do
            local army = side:getBattleArmy(ai)
            for ui = 0, army.unitCount - 1 do
                local bu = army:getBattleUnit(ui)
                local u = bu.unit
                local role, info = classify(u)
                table.insert(sides[si + 1], { bu = bu, u = u, role = role, ri = roleIndex(role), info = info,
                    start = math.max(1, bu.soldiersStart), px = u.battlePosX, py = u.battlePosY,
                    lastKilled = 0, lastLost = 0, steps = {} })
            end
        end
    end
    return sides
end

local function alive(r) return r.u.soldierCountBattleMap or 0 end
local function routing(r) return r.u.moraleLevel == MORALE_ROUTING end
local function active(r) return alive(r) > 0 and not routing(r) and not r.u.dead end

local function centroid(list)
    local x, y, n = 0, 0, 0
    for _, r in ipairs(list) do if active(r) then x = x + r.u.battlePosX; y = y + r.u.battlePosY; n = n + 1 end end
    if n == 0 then return 0, 0, 0 end
    return x / n, y / n, n
end

-- ---------- features ----------
local function status(r)
    if routing(r) then return 6 end                              -- fleeing
    if r.u:isFiring() then return 5 end                          -- firing
    if r.u:isEngaged() then return 4 end                         -- attacking
    local dx, dy = r.u.battlePosX - r.px, r.u.battlePosY - r.py
    if dx * dx + dy * dy < 4 then return 3 end                   -- hold
    return 0
end

local function strength(list, pred)
    local s = 0
    for _, r in ipairs(list) do if active(r) and (not pred or pred(r)) then s = s + alive(r) * (1 + r.info.e.primaryStats.attack / 10) end end
    return s
end

local function frame(F)
    local cx, cy = centroid(F.own); local ex, ey = centroid(F.enemy)
    local fx, fy = ex - cx, ey - cy
    local n = math.sqrt(fx * fx + fy * fy); if n < 1e-3 then fx, fy, n = 0, 1, 1 end
    fx, fy = fx / n, fy / n
    return { cx = cx, cy = cy, fx = fx, fy = fy, rx = fy, ry = -fx }
end
local function toRel(fr, x, y) local dx, dy = x - fr.cx, y - fr.cy; return dx * fr.rx + dy * fr.ry, dx * fr.fx + dy * fr.fy end

local function globalFeatures(F, fr, t)
    local own, en = F.own, F.enemy
    local oS, eS = strength(own), strength(en)
    local tot = math.max(1, oS + eS)
    local function zone(list, pred) return strength(list, function(r) local x, y = toRel(fr, r.u.battlePosX, r.u.battlePosY); return pred(x, y) end) end
    local g = {}
    g[1] = 1.0
    g[2] = math.min(1, t / 1800)
    g[3] = F.ownAlive / F.ownStart
    g[4] = F.enAlive / F.enStart
    g[5] = clamp((1 - g[4]) - (1 - g[3]), -1, 1)
    g[6] = math.tanh((zone(own, function(x, y) return math.abs(x) <= 80 and math.abs(y) <= 60 end) - zone(en, function(x, y) return math.abs(x) <= 80 end)) / tot)
    g[7] = math.tanh((zone(own, function(x) return x < -80 end) - zone(en, function(x) return x < -80 end)) / tot)
    g[8] = math.tanh((zone(own, function(x) return x > 80 end) - zone(en, function(x) return x > 80 end)) / tot)
    g[9] = zone(own, function(x, y) return y < -50 end) / math.max(1, oS)
    local om, em = strength(own, function(r) return r.info.missile end), strength(en, function(r) return r.info.missile end)
    g[10] = (om - em) / (om + em + 1)
    local oc, ec = strength(own, function(r) return r.info.cav end), strength(en, function(r) return r.info.cav end)
    g[11] = (oc - ec) / (oc + ec + 1)
    local eng, n = 0, 0
    for _, r in ipairs(own) do if active(r) then n = n + 1; if r.u:isEngaged() then eng = eng + 1 end end end
    g[12] = n > 0 and eng / n or 0
    g[13], g[14] = F.ownGeneralAlive and 1 or 0, F.enGeneralAlive and 1 or 0
    local mo, me, no, ne = 0, 0, 0, 0
    for _, r in ipairs(own) do if alive(r) > 0 then mo = mo + (routing(r) and 0 or 1); no = no + 1 end end
    for _, r in ipairs(en) do if alive(r) > 0 then me = me + (routing(r) and 0 or 1); ne = ne + 1 end end
    g[15], g[16] = no > 0 and mo / no or 0, ne > 0 and me / ne or 0
    g[17] = (oS - eS) / tot * 0.5 + 0.5                          -- ground-control proxy: share of contested strength near centre
    g[18], g[19], g[20] = 0, 0, 0                                -- settlement/plaza flags (open-field first; [VERIFY] battleResidence/plazaData)
    return g
end

local function sparse(gl, r, F, fr)
    local out, idx = {}, 0
    local function put(i, v) if v ~= 0 then out[#out + 1] = { i, v } end end
    for i = 1, NG do put(i - 1, gl[i]) end
    local e, info, p = r.info.e, r.info, r.info.e.primaryStats
    local l = {
        info.cav and 0 or 1, info.cav and 1 or 0, info.heavy and 1 or 0, (not info.heavy and not info.missile) and 1 or 0,
        info.missile and 1 or 0, info.spear and 1 or 0, 0, e.canPhalanx and 1 or 0, 0,
        clamp(p.attack / 25, 0, 1), clamp(p.charge / 15, 0, 1), 0, 0, 0, clamp((e.moveSpeedMod or 1) / 2.5, 0, 1), clamp((e.mass or 1) / 5, 0, 1),
    }
    local rx, ry = toRel(fr, r.u.battlePosX, r.u.battlePosY)
    l[#l + 1] = clamp(rx / 300, -1, 1); l[#l + 1] = clamp(ry / 300, -1, 1)
    l[#l + 1] = alive(r) / r.start
    l[#l + 1] = clamp((r.u.moraleLevel or 0) / 6, 0, 1); l[#l + 1] = clamp((r.u.fatigue or 0) / 6, 0, 1)
    l[#l + 1] = (r.u.maxAmmo or 0) > 0 and r.u.currentAmmo / r.u.maxAmmo or 0
    l[#l + 1] = r.u:isEngaged() and 1 or 0; l[#l + 1] = r.u:isUnderFire() and 1 or 0
    local nf, ne = 0, 0
    for _, o in ipairs(F.own) do if o ~= r and active(o) and math.abs(o.u.battlePosX - r.u.battlePosX) < 80 and math.abs(o.u.battlePosY - r.u.battlePosY) < 80 then nf = nf + alive(o) end end
    for _, o in ipairs(F.enemy) do if active(o) and math.abs(o.u.battlePosX - r.u.battlePosX) < 80 and math.abs(o.u.battlePosY - r.u.battlePosY) < 80 then ne = ne + alive(o) end end
    l[#l + 1] = clamp(nf / 500, 0, 1); l[#l + 1] = clamp(ne / 500, 0, 1)
    -- l has 26 entries: 9 type/capability + 7 stats(+placeholders for missile stats below) + 10 live/proximity
    l[12] = clamp((p.isMissile and (p.attack or 0) or 0) / 20, 0, 1)                      -- missile attack
    local okR, rng = pcall(function() return p.range end)                                 -- [VERIFY] attackStats range field name
    l[13] = okR and clamp((rng or 0) / 250, 0, 1) or 0
    local okD, def = pcall(function() local d = e.primaryDefenseStats; return d.armour + d.shield end)
    l[14] = okD and clamp(def / 35, 0, 1) or 0                                            -- total defence
    for i = 1, NL do put(NG + i - 1, l[i] or 0) end
    -- egocentric 8x8 grid, 6 binary channels: friendly, enemy, hold, attacking, firing, fleeing
    local function cell(o, isEnemy)
        if alive(o) <= 0 then return end
        local dx, dy = o.u.battlePosX - r.u.battlePosX, o.u.battlePosY - r.u.battlePosY
        local gx, gy = dx * fr.rx + dy * fr.ry, dx * fr.fx + dy * fr.fy
        if gx < -100 or gx >= 100 or gy < -100 or gy >= 100 then return end
        local c, rr = math.floor((gx + 100) / 25), math.floor((gy + 100) / 25)
        local base = NG + NL + (rr * GRID + c) * GCH
        local st = status(o)
        local seen = {}
        local function once(i) if not seen[i] then seen[i] = true; out[#out + 1] = { i, 1 } end end
        once(base + (isEnemy and 1 or 0))
        if st >= 3 then once(base + st - 1) end
    end
    for _, o in ipairs(F.own) do if o ~= r then cell(o, false) end end
    for _, o in ipairs(F.enemy) do cell(o, true) end
    return out
end

-- ---------- policy ----------
local function qvalue(W, ri, a, feats)
    local row = W[ri][a + 1]; local q = 0
    for _, f in ipairs(feats) do q = q + (row[f[1] + 1] or 0) * f[2] end
    return q
end

local function chooseAction(W, ri, feats, eps)
    if math.random() < eps then return math.random(0, NA - 1) end
    local best, bq = 0, -1e18
    for a = 0, NA - 1 do local q = qvalue(W, ri, a, feats); if q > bq then bq, best = q, a end end
    return best
end

local function nearestEnemy(F, fr, r, quad)
    local best, bd, any, ad = nil, 1e18, nil, 1e18
    for _, o in ipairs(F.enemy) do
        if active(o) then
            local dx, dy = o.u.battlePosX - r.u.battlePosX, o.u.battlePosY - r.u.battlePosY
            local d = dx * dx + dy * dy
            local gx, gy = dx * fr.rx + dy * fr.ry, dx * fr.fx + dy * fr.fy
            local ang = math.deg(math.atan2(gx, gy))
            local q = (math.abs(ang) <= 45 and 0) or (ang < -45 and ang >= -135 and 1) or (ang > 45 and ang <= 135 and 2) or 3
            if d < ad then ad, any = d, o end
            if q == quad and d < bd then bd, best = d, o end
        end
    end
    return best or any, math.sqrt(ad)
end

local function issue(a, r, F, fr)
    local u = r.u
    if a == 8 then u:halt(); return end
    if a <= 7 then
        local o = MOVES[a]
        local tx = u.battlePosX + o[1] * fr.rx + o[2] * fr.fx
        local ty = u.battlePosY + o[1] * fr.ry + o[2] * fr.fy
        local _, ed = nearestEnemy(F, fr, r, -1)
        u:moveToPosition(tx, ty, ed < 150 or u:isUnderFire())
    else
        local tgt = nearestEnemy(F, fr, r, a - 9)
        if tgt then u:attackUnit(tgt.u, true) else u:halt() end
    end
end

-- ---------- replay ----------
local SYM = { line_inf = "I", spear_inf = "S", missile_foot = "A", shock_cav = "C", missile_cav = "C", general = "G" }
local ST = { [6] = "R", [5] = "F", [4] = "A", [3] = "H" }
local function replayFrame(F, t)
    local units, minx, miny, maxx, maxy = {}, 1e9, 1e9, -1e9, -1e9
    for side, list in ipairs({ F.own, F.enemy }) do
        for i, r in ipairs(list) do
            local x, y = r.u.battlePosX, r.u.battlePosY
            minx, miny, maxx, maxy = math.min(minx, x), math.min(miny, y), math.max(maxx, x), math.max(maxy, y)
            local sym = SYM[r.role]; if side == 2 then sym = string.lower(sym) end
            units[#units + 1] = { side, i, math.floor(x), math.floor(y), math.floor(100 * alive(r) / r.start), r.u.moraleLevel or 0, status(r), r.lastAction or -1, sym }
        end
    end
    local grid = {}
    for i = 1, 24 do grid[i] = {} for j = 1, 24 do grid[i][j] = ".." end end
    local w, h = math.max(1, maxx - minx), math.max(1, maxy - miny)
    for _, u in ipairs(units) do
        local gx, gy = clamp(math.floor((u[3] - minx) / w * 23), 0, 23) + 1, clamp(math.floor((u[4] - miny) / h * 23), 0, 23) + 1
        grid[gy][gx] = u[9] .. (ST[u[7]] or ".")
    end
    local rows = {}
    for i = 24, 1, -1 do rows[#rows + 1] = table.concat(grid[i], " ") end
    return { t = math.floor(t), grid = rows, units = units }
end

local function getBattle()
    if gameDataAll and gameDataAll.get then
        local gd = gameDataAll.get()
        if gd and gd.battleStruct then return gd.battleStruct end
    end
    if M2TW and M2TW.battle then return M2TW.battle end
    if BATTLE then return BATTLE end
    return nil
end

-- ---------- public hooks ----------
function M.onBattleAiCommenced()
    local sc = json.decode(readfile(WS .. "scenario.json") or "{}") or {}
    local wtxt = readfile(WS .. "weights_latest.json")
    S = { battleId = sc.battle_id or 0, mode = sc.mode or "vanilla_vs_vanilla", t = 0, nextTick = TICK, done = false,
        W = wtxt and json.decode(wtxt) or nil, frames = {}, tickN = 0 }
    if S.mode == "rl_vs_frozen" then S.WF = json.decode(readfile(WS .. "weights_frozen.json") or "null") end
    local b = getBattle()
    if not b then
        print("[RL_AI] ERROR: getBattle() returned nil")
        return
    end
    print("[RL_AI] Battle commenced! State=" .. tostring(b.battleState) .. " ID=" .. tostring(S.battleId))
    pcall(function() b.battleSpeed = 15.0 end)                               -- Accelerated sim speed (15x)
    S.F = collect(b)
    local own, en = S.F[1], S.F[2]
    S.F = { own = own, enemy = en, ownStart = 1, enStart = 1 }
    for _, r in ipairs(own) do S.F.ownStart = S.F.ownStart + r.start end
    for _, r in ipairs(en) do S.F.enStart = S.F.enStart + r.start end
    S.prevOwnLost, S.prevEnLost, S.prevGround = 0, 0, 0.5
    S.pending = {}
end

local function totals(F)
    local oa, ea, og, eg = 0, 0, false, false
    for _, r in ipairs(F.own) do oa = oa + alive(r); if r.role == "general" and active(r) then og = true end end
    for _, r in ipairs(F.enemy) do ea = ea + alive(r); if r.role == "general" and active(r) then eg = true end end
    F.ownAlive, F.enAlive, F.ownGeneralAlive, F.enGeneralAlive = oa, ea, og, eg
end

local function actSide(list, W, eps, fr, g, F, learner)
    for _, r in ipairs(list) do
        if active(r) and r.ri then
            local feats = sparse(g, r, F, fr)
            if S.mode ~= "vanilla_vs_vanilla" and W then
                local a = chooseAction(W, r.ri, feats, eps)
                issue(a, r, F, fr); r.lastAction = a
                if learner then r.pending = { phi = feats, a = a } end
            elseif learner then
                r.pending = { phi = feats, a = -1, sx = r.u.battlePosX, sy = r.u.battlePosY, frm = fr }
            end
        end
    end
end

function M.onBattleTick()
    if not S or S.done then return end
    local b = getBattle()
    if not b then return end
    local t = b.secondsPassed or 0
    if b.battleState ~= nil and t < S.nextTick then return end
    pcall(function() b.battleSpeed = 15.0 end)
    -- Hard timeout safeguard: auto-complete if battle exceeds 600s in-game (~40s wall-clock at 15x)
    if t >= 600 then
        print("[RL_AI] Battle reached 600s time cap. Completing episode.")
        M.onBattleFinished()
        return
    end
    S.nextTick = S.nextTick + TICK; S.tickN = S.tickN + 1
    local F = S.F
    totals(F)
    -- close the previous step for the learner side (reward deltas + Way-2 displacement)
    local ol, el = F.ownStart - F.ownAlive, F.enStart - F.enAlive
    local gc = ((el - S.prevEnLost) - (ol - S.prevOwnLost)) / F.ownStart * 10
    local ground = (strength(F.own) - strength(F.enemy)) / math.max(1, strength(F.own) + strength(F.enemy)) * 0.5 + 0.5
    for _, r in ipairs(F.own) do
        local p = r.pending
        if p then
            local st = { phi = p.phi, a = p.a, dk = (r.bu.soldiersKilled - r.lastKilled) / 20, dl = (r.bu.soldiersLost - r.lastLost) / 20,
                gc = gc, gm = ground - S.prevGround }
            if S.mode == "vanilla_vs_vanilla" and p.frm then
                local dx, dy = r.u.battlePosX - p.sx, r.u.battlePosY - p.sy
                st.disp = { dx * p.frm.rx + dy * p.frm.ry, dx * p.frm.fx + dy * p.frm.fy }
                if r.u:isEngaged() then
                    local _, _ = nil, nil
                    local tgt = nearestEnemy(F, p.frm, r, -1)
                    if tgt then
                        local ex, ey = tgt.u.battlePosX - r.u.battlePosX, tgt.u.battlePosY - r.u.battlePosY
                        local ang = math.deg(math.atan2(ex * p.frm.rx + ey * p.frm.ry, ex * p.frm.fx + ey * p.frm.fy))
                        st.vq = (math.abs(ang) <= 45 and 0) or (ang < -45 and ang >= -135 and 1) or (ang > 45 and ang <= 135 and 2) or 3
                    end
                end
            end
            r.steps[#r.steps + 1] = st
            r.pending = nil
        end
        r.lastKilled, r.lastLost = r.bu.soldiersKilled, r.bu.soldiersLost
    end
    S.prevOwnLost, S.prevEnLost, S.prevGround = ol, el, ground
    -- new decisions
    local fr = frame(F)
    local g = globalFeatures(F, fr, t)
    actSide(F.own, S.W, EPS.learner, fr, g, F, true)
    if S.mode == "rl_vs_frozen" and S.WF then
        local efr = frame({ own = F.enemy, enemy = F.own })
        local eF = { own = F.enemy, enemy = F.own, ownStart = F.enStart, enStart = F.ownStart, ownAlive = F.enAlive, enAlive = F.ownAlive,
            ownGeneralAlive = F.enGeneralAlive, enGeneralAlive = F.ownGeneralAlive }
        actSide(F.enemy, S.WF, EPS.frozen, efr, globalFeatures(eF, efr, t), eF, false)
    end
    for _, list in ipairs({ F.own, F.enemy }) do for _, r in ipairs(list) do r.px, r.py = r.u.battlePosX, r.u.battlePosY end end
    if S.tickN % 3 == 1 then S.frames[#S.frames + 1] = replayFrame(F, t) end
end

function M.onBattleFinished()
    if not S or S.done then return end
    S.done = true
    local b = getBattle()
    local won = 1
    pcall(function() won = b and b.sides[0].wonBattle or 1 end)
    local units, roster = {}, {}
    for i, r in ipairs(S.F.own) do
        if #r.steps > 0 then units[#units + 1] = { id = i, role = r.role, steps = r.steps } end
        roster[#roster + 1] = { side = 0, id = i, role = r.role, start = r.start }
    end
    for i, r in ipairs(S.F.enemy) do roster[#roster + 1] = { side = 1, id = i, role = r.role, start = r.start } end
    local ep = { battle_id = S.battleId, mode = S.mode, result = won - 1, tick_seconds = TICK, units = units }
    -- write replay first, episode last (trainer polls for episode files)
    writefile(WS .. string.format("replays/replay_%d.json", S.battleId),
        json.encode({ battle_id = S.battleId, mode = S.mode, roster = roster, frames = S.frames, result = won - 1 }))
    writefile(WS .. string.format("episodes/episode_%d.json", S.battleId), json.encode(ep))
    writefile(WS .. "last_battle_result.txt", string.format("Battle %d finished: result=%d ticks=%d\n", S.battleId, won - 1, S.tickN))
    pcall(function() M2TWEOP.quitGame() end)                                  -- [VERIFY] exit so runner restarts with next scenario
end

return M
