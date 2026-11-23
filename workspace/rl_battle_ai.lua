-- rl_battle_ai.lua : decentralized unit-level Linear SARSA(lambda) *inference + logging* inside M2TWEOP.
-- All learning happens post-battle in Python. Spec constants must match trainer/spec.py.
-- NOTE: written against the M2TWEOP 3.3.1 Lua API; calls marked [VERIFY] must be confirmed in the first in-game smoke test.
local json = require('helpers/dkjson')

local M = {}
local WS = "Z:/home/tomek/workspace/"          -- Wine drive Z: = Linux root; shared volume with trainer
local TICK = 1.0
local UNITS_PER_SEC = 10
local UNIT_COMMIT_SEC = 6.0
local CMD_INTERVAL = 12.0
local NA, NCMD, NG, NL, GRID, GCH = 13, 10, 50, 28, 8, 6
local NF = NG + NL + GRID * GRID * GCH           -- 462
local MAX_SQUAD_SIZE = 3
local ROLES = {
    "line_inf", "spear_inf", "missile_foot", "shock_cav", "missile_cav", "general", "artillery",
    "commander",
}
local MOVES = { [0] = { 0, 60 }, { -45, 45 }, { 45, 45 }, { -60, 0 }, { 60, 0 }, { -45, -45 }, { 45, -45 }, { 0, -60 }, { 0, 0 } }
local MORALE_ROUTING = 6                         -- moraleStatus.routing = 6 (0 is berserk)
local EPS = { learner = 0.10, frozen = 0.02 }

local S = nil  -- per-battle state

local function readfile(p) local f = io.open(p, "r"); if not f then return nil end; local s = f:read("*a"); f:close(); return s end
local function writefile(p, s) local f = io.open(p, "w"); if f then f:write(s); f:close() end end
local function clamp(x, a, b) if x < a then return a elseif x > b then return b end return x end

-- ---------- unit classification ----------
local function lower(v) return string.lower(tostring(v)) end
local function classify(u, bu)
    local e = u.eduEntry
    local cat = e and e.category or -1
    local cls = e and e.class or -1
    local combo = e and e.categoryClassCombo or -1
    local isArtillery = (cat == 2)
    local isCav = (cat == 1) or (combo == 5 or combo == 6 or combo == 7)
    local isMissile = isArtillery or (combo == 4 or combo == 7) or (cls == 4 or cls == 2)
    local isSpear = (combo == 3) or (cls == 3)
    local isHeavy = (combo == 2 or combo == 6) or (cls == 0)
    local isGeneral = (bu and (bu.isGeneral == 1 or bu.isGeneral == true)) or (u.character ~= nil)
    local role
    if isGeneral then role = "general"
    elseif isArtillery then role = "artillery"
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
    local zeroBased = (battle.sides[0] ~= nil)
    for si = 1, 2 do
        local side = zeroBased and battle.sides[si - 1] or battle.sides[si]
        if side then
            local aNum = side.battleArmyNum or side.armiesNum or 1
            for ai = 0, aNum - 1 do
                local ok, army = pcall(function() return side:getBattleArmy(ai) end)
                if ok and army then
                    local uNum = army.unitCount or 0
                    for ui = 0, uNum - 1 do
                        local ok_u, bu = pcall(function() return army:getBattleUnit(ui) end)
                        if ok_u and bu and bu.unit then
                            local u = bu.unit
                            local role, info = classify(u, bu)
                            table.insert(sides[si], { bu = bu, u = u, role = role, ri = roleIndex(role), info = info,
                                start = math.max(1, bu.soldiersStart or 1), px = u.battlePosX or 0, py = u.battlePosY or 0,
                                prevPx = u.battlePosX or 0, prevPy = u.battlePosY or 0,
                                lastKilled = 0, lastLost = 0, steps = {} })
                        end
                    end
                end
            end
        end
    end
    return sides
end

local function alive(r) return r.u.soldierCountBattleMap or 0 end
local function routing(r) return r.u.moraleLevel == MORALE_ROUTING end
local function active(r) return alive(r) > 0 and not routing(r) and (not r.u.dead or r.u.dead == 0) end
local function isEngaged(u) local ok, v = pcall(function() return u:isEngaged() end); return ok and (v == true or v == 1) end
local function isUnderFire(u) local ok, v = pcall(function() return u:isUnderFire() end); return ok and (v == true or v == 1) end

local function centroid(list)
    local x, y, n = 0, 0, 0
    for _, r in ipairs(list) do if active(r) then x = x + r.u.battlePosX; y = y + r.u.battlePosY; n = n + 1 end end
    if n == 0 then return 0, 0, 0 end
    return x / n, y / n, n
end

local function infantryCentroid(list, fr)
    local x, y, n = 0, 0, 0
    for _, r in ipairs(list) do
        if active(r) and (r.role == "line_inf" or r.role == "spear_inf") then
            x = x + r.u.battlePosX; y = y + r.u.battlePosY; n = n + 1
        end
    end
    if n == 0 then return fr.cx, fr.cy end
    return x / n, y / n
end

-- ---------- features ----------
local function inMissileRange(r, foes)
    if not r.info.missile or not foes then return false end
    -- M2TWEOP reports maxAmmo=0/currentAmmo=0 for siege artillery; only gate on currentAmmo when maxAmmo > 0
    if (r.u.maxAmmo or 0) > 0 and (r.u.currentAmmo or 0) <= 0 then return false end
    local defaultRng = (r.role == "artillery") and 300 or 160
    local rng = (r.info.e and r.info.e.range and r.info.e.range > 0) and r.info.e.range or defaultRng
    local r2 = rng * rng
    local rx, ry = r.u.battlePosX or 0, r.u.battlePosY or 0
    for _, f in ipairs(foes) do
        if alive(f) > 0 then
            local fx, fy = (f.u.battlePosX or 0) - rx, (f.u.battlePosY or 0) - ry
            if fx * fx + fy * fy <= r2 then return true end
        end
    end
    return false
end

local function status(r, foes)
    if routing(r) then return 6 end                              -- fleeing
    local isEngaged = false
    pcall(function() isEngaged = r.u:isEngaged() end)
    if isEngaged then return 4 end                               -- attacking in melee
    local dx, dy = (r.u.battlePosX or 0) - r.px, (r.u.battlePosY or 0) - r.py
    if dx * dx + dy * dy < 4 then
        if inMissileRange(r, foes) then return 5 end             -- firing (has ammo + enemy in range)
        return 3                                                 -- hold
    end
    return 0                                                     -- move
end

local function strength(list, pred)
    local s = 0
    for _, r in ipairs(list) do
        if active(r) and (not pred or pred(r)) then
            local atk = (r.info.e and r.info.e.aiUnitValuePerSoldier) or 5
            s = s + alive(r) * (1 + atk / 10)
        end
    end
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
    for _, r in ipairs(own) do if active(r) then n = n + 1; if isEngaged(r.u) then eng = eng + 1 end end end
    g[12] = n > 0 and eng / n or 0
    g[13], g[14] = F.ownGeneralAlive and 1 or 0, F.enGeneralAlive and 1 or 0
    local mo, me, no, ne = 0, 0, 0, 0
    for _, r in ipairs(own) do if alive(r) > 0 then mo = mo + (routing(r) and 0 or 1); no = no + 1 end end
    for _, r in ipairs(en) do if alive(r) > 0 then me = me + (routing(r) and 0 or 1); ne = ne + 1 end end
    g[15], g[16] = no > 0 and mo / no or 0, ne > 0 and me / ne or 0
    g[17] = (oS - eS) / tot * 0.5 + 0.5                          -- ground-control proxy: share of contested strength near centre
    g[18], g[19], g[20] = 0, 0, 0                                -- settlement/plaza flags (populated below when battleResidence exists)
    if S and S.siegeInfo and S.siegeInfo.isSiege then
        g[18] = S.siegeInfo.isAttacker and 1.0 or -1.0
        g[19] = S.siegeInfo.breached and 1.0 or 0.0
        g[20] = S.siegeInfo.plazaFrac or 0.0
    end
    for i = 21, NG do g[i] = 0 end                               -- global order (21..30), own detachment order (31..40), lead line_inf order (41..50)
    return g
end

local function globalSparse(g)
    local out = {}
    for i = 1, NG do if g[i] and g[i] ~= 0 then out[#out + 1] = { i - 1, g[i] } end end
    return out
end

local function sparse(gl, r, F, fr, infCx, infCy)
    local out, idx = {}, 0
    local function put(i, v) if v ~= 0 then out[#out + 1] = { i, v } end end
    for i = 1, NG do
        local scale = (i >= 18 and i <= 50) and 3.0 or 1.0
        put(i - 1, (gl[i] or 0) * scale)
    end
    local e, info = r.info.e, r.info
    local atk = (e and e.aiUnitValuePerSoldier) or 5
    local chg = 4
    local l = {
        info.cav and 0 or 1, info.cav and 1 or 0, info.heavy and 1 or 0, (not info.heavy and not info.missile) and 1 or 0,
        info.missile and 1 or 0, info.spear and 1 or 0, 0, (e and (e.canPhalanx == true or e.canPhalanx == 1)) and 1 or 0, 0,
        clamp(atk / 25, 0, 1), clamp(chg / 15, 0, 1), 0, 0, 0, clamp((e and e.moveSpeedMod or 1) / 2.5, 0, 1), clamp((e and e.mass or 1) / 5, 0, 1),
    }
    local rx, ry = toRel(fr, r.u.battlePosX, r.u.battlePosY)
    l[#l + 1] = clamp(rx / 300, -1, 1); l[#l + 1] = clamp(ry / 300, -1, 1)
    l[#l + 1] = alive(r) / r.start
    l[#l + 1] = clamp((6 - (r.u.moraleLevel or 2)) / 6, 0, 1); l[#l + 1] = clamp((r.u.fatigue or 0) / 6, 0, 1)
    l[#l + 1] = (r.u.maxAmmo or 0) > 0 and r.u.currentAmmo / r.u.maxAmmo or 0
    l[#l + 1] = isEngaged(r.u) and 1 or 0; l[#l + 1] = isUnderFire(r.u) and 1 or 0
    local nf, ne = 0, 0
    for _, o in ipairs(F.own) do if o ~= r and active(o) and math.abs(o.u.battlePosX - r.u.battlePosX) < 80 and math.abs(o.u.battlePosY - r.u.battlePosY) < 80 then nf = nf + alive(o) end end
    for _, o in ipairs(F.enemy) do if active(o) and math.abs(o.u.battlePosX - r.u.battlePosX) < 80 and math.abs(o.u.battlePosY - r.u.battlePosY) < 80 then ne = ne + alive(o) end end
    l[#l + 1] = clamp(nf / 500, 0, 1); l[#l + 1] = clamp(ne / 500, 0, 1)
    -- l has 26 entries: 9 type/capability + 7 stats(+placeholders for missile stats below) + 10 live/proximity
    local isMissile = info.missile
    local rng = (isMissile and e and e.range) or 0
    l[12] = clamp((isMissile and atk or 0) / 20, 0, 1)                      -- missile attack
    l[13] = clamp((rng or 0) / 250, 0, 1)                                   -- missile range
    l[14] = clamp(((r.u.armourInBattle or 5) + (r.u.defenceInBattle or 5)) / 30, 0, 1)
    local idx_x = (r.u.battlePosX - infCx) * fr.rx + (r.u.battlePosY - infCy) * fr.ry
    local idx_y = (r.u.battlePosX - infCx) * fr.fx + (r.u.battlePosY - infCy) * fr.fy
    l[27] = clamp(idx_x / 200, -1, 1)
    l[28] = clamp(idx_y / 200, -1, 1)
    for i = 1, NL do put(NG + i - 1, l[i] or 0) end
    -- egocentric 8x8 grid, 6 binary channels: friendly, enemy, hold, attacking, firing, fleeing
    local seen = {}
    local function once(i) if not seen[i] then seen[i] = true; out[#out + 1] = { i, 1 } end end
    local function cell(o, isEnemy)
        if alive(o) <= 0 then return end
        local dx, dy = o.u.battlePosX - r.u.battlePosX, o.u.battlePosY - r.u.battlePosY
        local gx, gy = dx * fr.rx + dy * fr.ry, dx * fr.fx + dy * fr.fy
        if gx < -100 or gx >= 100 or gy < -100 or gy >= 100 then return end
        local c, rr = math.floor((gx + 100) / 25), math.floor((gy + 100) / 25)
        local base = NG + NL + (rr * GRID + c) * GCH
        local st = status(o, isEnemy and F.own or F.enemy)
        once(base + (isEnemy and 1 or 0))
        if st >= 3 then once(base + st - 1) end
    end
    for _, o in ipairs(F.own) do if o ~= r then cell(o, false) end end
    for _, o in ipairs(F.enemy) do cell(o, true) end
    return out
end

local CMD_RNN_HIDDEN = 48
local UNIT_RNN_HIDDEN = 32

local function zeroHidden(n)
    local h = {}
    for i = 1, (n or CMD_RNN_HIDDEN) do h[i] = 0.0 end
    return h
end

local function copyHidden(h, n)
    local out = {}
    for i = 1, (n or CMD_RNN_HIDDEN) do out[i] = math.floor(((h and h[i]) or 0) * 10000 + 0.5) / 10000 end
    return out
end

-- ---------- policy (Purely Additive 2-Layer Towell-Shavlik KBANN + Zero-Init Homogeneous Top RNN) ----------
local function unitRNNForward(W, ri, feats, hPrev)
    local hIn = hPrev or zeroHidden(UNIT_RNN_HIDDEN)
    local rw = W and W[ri]
    if not rw or not rw.W_kb or not rw.W2 then return nil, hIn end
    local W_kb, b_kb, W_kb_out, b_kb_out = rw.W_kb, rw.b_kb, rw.W_kb_out, rw.b_kb_out
    local W_top_kb, W_top_x, Wh, b_top, W2 = rw.W_top_kb, rw.W_top_x, rw.Wh, rw.b_top, rw.W2
    local h_kb = {}
    for k = 1, #W_kb do
        local s = b_kb[k] or 0
        local row = W_kb[k]
        for _, f in ipairs(feats) do s = s + (row[f[1] + 1] or 0) * f[2] end
        h_kb[k] = math.tanh(s)
    end
    local hNew = {}
    for j = 1, #W_top_kb do
        local s = b_top[j] or 0
        local rkb = W_top_kb[j]
        for k = 1, #h_kb do s = s + (rkb[k] or 0) * h_kb[k] end
        local rctx = W_top_x[j]
        if rctx then
            for _, f in ipairs(feats) do
                if f[1] < (NG + NL) then s = s + (rctx[f[1] + 1] or 0) * f[2] end
            end
        end
        if Wh and Wh[j] then
            local rh = Wh[j]
            for k = 1, #hIn do s = s + (rh[k] or 0) * (hIn[k] or 0) end
        end
        hNew[j] = math.tanh(s)
    end
    local h_adapt = nil
    if rw.W_adapt_out and rw.W_adapt_h and rw.W_adapt_x then
        local W_ah, W_ax, b_a = rw.W_adapt_h, rw.W_adapt_x, rw.b_adapt
        h_adapt = {}
        for m = 1, #W_ah do
            local s = (b_a and b_a[m]) or 0
            local rah = W_ah[m]
            for j = 1, #hNew do s = s + (rah[j] or 0) * hNew[j] end
            local rax = W_ax[m]
            if rax then
                for _, f in ipairs(feats) do
                    if f[1] < (NG + NL) then s = s + (rax[f[1] + 1] or 0) * f[2] end
                end
            end
            h_adapt[m] = math.tanh(s)
        end
    end
    local q = {}
    local W_ao = rw.W_adapt_out
    for a = 1, NA do
        local s = (b_kb_out and b_kb_out[a]) or 0
        local rko = W_kb_out and W_kb_out[a]
        if rko then for k = 1, #h_kb do s = s + (rko[k] or 0) * h_kb[k] end end
        local row = W2[a]
        for j = 1, #hNew do s = s + (row[j] or 0) * hNew[j] end
        if h_adapt and W_ao and W_ao[a] then
            local rao, ad = W_ao[a], 0
            for m = 1, #h_adapt do ad = ad + (rao[m] or 0) * h_adapt[m] end
            s = s + clamp(ad, -1.0, 1.0)
        end
        q[a - 1] = s
    end
    return q, hNew
end

local function sampleSoftmax(logits, nActions, invTemp)
    local scale = invTemp or (S and S.invTemp) or 2.2
    local maxL = -1e18
    for a = 0, nActions - 1 do
        local v = (logits[a] or 0) * scale
        if v > maxL then maxL = v end
    end
    local probs, sumP = {}, 0.0
    for a = 0, nActions - 1 do
        local p = math.exp(clamp((logits[a] or 0) * scale - maxL, -30, 30))
        probs[a] = p
        sumP = sumP + p
    end
    local r = math.random() * sumP
    local acc = 0.0
    for a = 0, nActions - 1 do
        acc = acc + probs[a]
        if r <= acc then return a end
    end
    return nActions - 1
end

local function stepCommanderRNN(rnn, feats, hPrev, eps)
    local hIn = hPrev or zeroHidden()
    if not rnn or not rnn.W_kb or not rnn.Wx or not rnn.Wh or not rnn.Wo then
        return math.random(0, NCMD - 1), hIn
    end
    local W_kb, b_kb, W_kb_out, b_kb_out = rnn.W_kb, rnn.b_kb, rnn.W_kb_out, rnn.b_kb_out
    local W_top_kb, Wx, Wh, bh, Wo = rnn.W_top_kb, rnn.Wx, rnn.Wh, rnn.bh, rnn.Wo
    local h_kb = {}
    for k = 1, #W_kb do
        local s = b_kb[k] or 0
        local row = W_kb[k]
        for _, f in ipairs(feats) do s = s + (row[f[1] + 1] or 0) * f[2] end
        h_kb[k] = math.tanh(s)
    end
    local hNew = {}
    for j = 1, #Wx do
        local s = bh[j] or 0
        local rkb = W_top_kb[j]
        for k = 1, #h_kb do s = s + (rkb[k] or 0) * h_kb[k] end
        local rx = Wx[j]
        for _, f in ipairs(feats) do s = s + (rx[f[1] + 1] or 0) * f[2] end
        local rh = Wh[j]
        for k = 1, #hIn do s = s + (rh[k] or 0) * (hIn[k] or 0) end
        hNew[j] = math.tanh(s)
    end
    local h_adapt = nil
    if rnn.W_adapt_out and rnn.W_adapt_h and rnn.W_adapt_x then
        local W_ah, W_ax, b_a = rnn.W_adapt_h, rnn.W_adapt_x, rnn.b_adapt
        h_adapt = {}
        for m = 1, #W_ah do
            local s = (b_a and b_a[m]) or 0
            local rah = W_ah[m]
            for j = 1, #hNew do s = s + (rah[j] or 0) * hNew[j] end
            local rax = W_ax[m]
            if rax then
                for _, f in ipairs(feats) do s = s + (rax[f[1] + 1] or 0) * f[2] end
            end
            h_adapt[m] = math.tanh(s)
        end
    end
    local logits = {}
    local W_ao = rnn.W_adapt_out
    for a = 0, NCMD - 1 do
        local q = (b_kb_out and b_kb_out[a + 1]) or 0
        local rko = W_kb_out and W_kb_out[a + 1]
        if rko then for k = 1, #h_kb do q = q + (rko[k] or 0) * h_kb[k] end end
        local ro = Wo[a + 1]
        for j = 1, #hNew do q = q + (ro[j] or 0) * hNew[j] end
        if h_adapt and W_ao and W_ao[a + 1] then
            local rao, ad = W_ao[a + 1], 0
            for m = 1, #h_adapt do ad = ad + (rao[m] or 0) * h_adapt[m] end
            q = q + clamp(ad, -1.0, 1.0)
        end
        logits[a] = q
    end
    return sampleSoftmax(logits, NCMD), hNew
end

local function buildCommanderBasePhi(F, fr, g)
    local base = {}
    local function put(idx0, v)
        if v and math.abs(v) > 1e-6 then base[#base + 1] = { idx0, math.floor(v * 10000 + 0.5) / 10000 } end
    end
    -- 0..19: global features g[1..20] (terrain/siege channels 18..20 scaled 3.0x)
    for i = 1, 20 do
        local scale = (i >= 18 and i <= 20) and 3.0 or 1.0
        put(i - 1, (g[i] or 0) * scale)
    end

    -- 20..91: 6x6 simplified battle map (friendly & enemy strength per 80m x 80m cell over [-240..240]m)
    local tot = math.max(1, strength(F.own) + strength(F.enemy))
    local mapCells = {}
    local function accumMap(list, ch)
        for _, r in ipairs(list) do
            if active(r) then
                local rx, ry = toRel(fr, r.u.battlePosX, r.u.battlePosY)
                local mc = clamp(math.floor((rx + 240) / 80), 0, 5)
                local mr = clamp(math.floor((ry + 240) / 80), 0, 5)
                local idx0 = 20 + (mr * 6 + mc) * 2 + ch
                local atk = (r.info.e and r.info.e.aiUnitValuePerSoldier) or 5
                mapCells[idx0] = (mapCells[idx0] or 0) + alive(r) * (1 + atk / 10) / tot
            end
        end
    end
    accumMap(F.own, 0)
    accumMap(F.enemy, 1)
    for idx0 = 20, 91 do if mapCells[idx0] then put(idx0, clamp(mapCells[idx0] * 3.0, 0, 1)) end end

    -- 92..171 (20 friendly units) and 172..251 (20 enemy units): canonical sorted [rx, ry, hp, role]
    local function encodeSideUnits(list, baseIdx0)
        local activeList = {}
        for _, r in ipairs(list) do
            if active(r) then
                local rx, ry = toRel(fr, r.u.battlePosX, r.u.battlePosY)
                activeList[#activeList + 1] = { r = r, rx = rx, ry = ry, ri = r.ri or 1 }
            end
        end
        table.sort(activeList, function(a, b)
            if a.ri ~= b.ri then return a.ri < b.ri end
            return a.rx < b.rx
        end)
        for k = 1, math.min(20, #activeList) do
            local item = activeList[k]
            local uBase = baseIdx0 + (k - 1) * 4
            put(uBase + 0, clamp(item.rx / 300, -1, 1))
            put(uBase + 1, clamp(item.ry / 300, -1, 1))
            put(uBase + 2, clamp(alive(item.r) / item.r.start, 0, 1))
            put(uBase + 3, item.ri / 7.0)
        end
    end
    encodeSideUnits(F.own, 92)
    encodeSideUnits(F.enemy, 172)
    return base
end

local function makeStepPhi(basePhi, extras)
    local out = {}
    for i = 1, #basePhi do out[i] = basePhi[i] end
    for _, kv in ipairs(extras) do
        if kv[2] and math.abs(kv[2]) > 1e-6 then
            out[#out + 1] = { kv[1], math.floor(kv[2] * 10000 + 0.5) / 10000 }
        end
    end
    return out
end

local function chooseAction(W, ri, feats, hPrev, eps, nActions)
    local na = nActions or NA
    local logits, hNew = unitRNNForward(W, ri, feats, hPrev)
    if not logits then return math.random(0, na - 1), hNew end
    return sampleSoftmax(logits, na), hNew
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

local function isUnitMoving(r)
    if not r.prevPx or not r.prevPy then return false end
    local dx = (r.u.battlePosX or 0) - r.prevPx
    local dy = (r.u.battlePosY or 0) - r.prevPy
    return (dx * dx + dy * dy) >= 2.0
end

local function issueHold(u, r)
    if r.lastOrderKind ~= "hold" or isUnitMoving(r) then
        u:halt()
        r.lastOrderKind = "hold"
        r.lastTargetUid = nil
        r.lastOrderT = (S and S.t) or 0
    end
end

local function issueMove(u, r, tx, ty, run)
    local now = (S and S.t) or 0
    local moving = isUnitMoving(r)
    local stale = (now - (r.lastOrderT or -99)) >= 4.5
    local need = (r.lastOrderKind ~= "move") or (r.lastMoveRun ~= run) or ((not moving) and stale)
    if not need and r.lastMoveTx and r.lastMoveTy then
        local d2 = (tx - r.lastMoveTx)^2 + (ty - r.lastMoveTy)^2
        if d2 > 900.0 then need = true end -- >30m shift
    end
    if need then
        u:moveToPosition(tx, ty, run)
        r.lastOrderKind = "move"
        r.lastMoveTx, r.lastMoveTy, r.lastMoveRun = tx, ty, run
        r.lastTargetUid = nil
        r.lastOrderT = now
    end
end

local function issueAttack(u, r, tgt, run)
    local now = (S and S.t) or 0
    local moving = isUnitMoving(r)
    local stale = (now - (r.lastOrderT or -99)) >= 4.5
    local tid = tgt.u and (tgt.u.unitID or tostring(tgt.u)) or tostring(tgt)
    if r.lastOrderKind ~= "attack" or r.lastTargetUid ~= tid or r.lastAtkRun ~= run or ((not moving) and stale) then
        u:attackUnit(tgt.u, run)
        r.lastOrderKind = "attack"
        r.lastTargetUid = tid
        r.lastAtkRun = run
        r.lastOrderT = now
    end
end

local function updateSiegeInfo(b)
    local info = { isSiege = false, isAttacker = false, breached = false, plazaFrac = 0.0, targetBld = nil, br = nil }
    if not b then return info end
    pcall(function()
        local br = (b.getBattleResidence and b.getBattleResidence()) or b.battleResidence
        if not br or not br.battleBuildings then return end
        local bb = br.battleBuildings
        local pCnt = bb.perimeterCount or 0
        if pCnt <= 0 then return end
        info.isSiege = true
        info.br = br
        -- Determine if Side 0 is attacker (defender alliance owns plazaData or settlement faction matches)
        local ownAlliance = 0
        local defAlliance = (br.plazaData and br.plazaData.alliance) or 1
        info.isAttacker = (ownAlliance ~= defAlliance)
        local wallsB = (br.settlementWallsBreached == 1 or br.settlementWallsBreached == true)
        local gateB = (br.settlementGateDestroyed == 1 or br.settlementGateDestroyed == true)
        info.breached = wallsB or gateB
        if br.plazaData then
            local pMax = math.max(1.0, br.plazaData.plazaMaxTime or 180.0)
            info.plazaFrac = clamp((br.plazaData.plazaControl or 0.0) / pMax, 0.0, 1.0)
        end
        -- Pick an intact outer perimeter building (gate/wall) for rams, ladders, towers & artillery
        if not info.breached then
            local perim = bb:getPerimeter(0)
            if perim and (perim.buildingCount or 0) > 0 then
                for i = 0, perim.buildingCount - 1 do
                    local bld = perim:getBuilding(i)
                    if bld and (bld.currentHealth or 0) > 0 then
                        info.targetBld = bld
                        break
                    end
                end
            end
        end
    end)
    return info
end

local function trySiegeOrder(u, r, now)
    local si = S and S.siegeInfo
    if not si or not si.isSiege or not si.isAttacker or si.breached or not si.targetBld then
        return false
    end
    -- 1) Unit already carrying a ram / ladder / siege tower -> attack outer wall/gate
    local engNum = 0
    pcall(function() engNum = u.siegeEngineNum or 0 end)
    if engNum > 0 then
        if r.lastOrderKind ~= "siege_bld" or (now - (r.lastOrderT or -99)) >= 10.0 then
            local ok = pcall(function() u:attackBuilding(si.targetBld) end)
            if ok then
                r.lastOrderKind = "siege_bld"
                r.lastOrderT = now
            end
        end
        return true
    end
    -- 2) Artillery -> bombard outer gate/wall until breached
    if r.role == "artillery" then
        if r.lastOrderKind ~= "siege_art" or (now - (r.lastOrderT or -99)) >= 12.0 then
            local ok = pcall(function() u:attackBuilding(si.targetBld) end)
            if ok then
                r.lastOrderKind = "siege_art"
                r.lastOrderT = now
            end
        end
        return true
    end
    -- 3) Infantry without equipment -> pick up nearby unmanned ram/ladder/tower if any
    if (r.role == "line_inf" or r.role == "spear_inf") and (now - (r.lastCollectT or -99)) >= 15.0 then
        local collected = false
        pcall(function()
            local b = getBattle()
            local bfe = b and b.battlefieldEngines
            if bfe and (bfe.engineNum or 0) > 0 then
                local ux, uy = u.battlePosX or 0, u.battlePosY or 0
                for i = 0, bfe.engineNum - 1 do
                    local eng = bfe:getEngine(i)
                    if eng and eng.currentUnit == nil then
                        local dx, dy = (eng.xCoord or 0) - ux, (eng.yCoord or 0) - uy
                        if dx * dx + dy * dy <= 14400.0 then -- within 120m
                            u:collectEngine(eng)
                            r.lastOrderKind = "collect_eng"
                            r.lastOrderT = now
                            r.lastCollectT = now
                            collected = true
                            break
                        end
                    end
                end
            end
        end)
        if collected then return true end
    end
    return false
end

local function issue(a, r, F, fr)
    local u = r.u
    local eng = isEngaged(u)
    local now = (S and S.t) or 0

    -- Siege assault equipment / gate-breaching handler (rams, ladders, towers, artillery)
    if not eng and trySiegeOrder(u, r, now) then
        return
    end

    -- Native M2TW pathfinder fallback: if a non-engaged, non-holding, non-firing unit
    -- hasn't moved >= 6m in 12s (e.g. a 60m move landed in a river/obstacle), hand it to
    -- M2TW's native cross-map pathfinding via attackUnit toward the nearest enemy for 12s.
    local ux, uy = u.battlePosX or 0, u.battlePosY or 0
    if eng or a == 8 or (r.info.missile and inMissileRange(r, F.enemy)) then
        r.progX, r.progY, r.progT = ux, uy, now
        r.nativePathUntil = nil
    else
        if r.nativePathUntil and now < r.nativePathUntil then
            return -- let M2TW's native pathfinder finish routing across the bridge/obstacle
        end
        if not r.progT then
            r.progX, r.progY, r.progT = ux, uy, now
        else
            local moved2 = (ux - r.progX)^2 + (uy - r.progY)^2
            if moved2 >= 36.0 then
                r.progX, r.progY, r.progT = ux, uy, now
            elseif (now - r.progT) >= 12.0 then
                local nearE = nearestEnemy(F, fr, r, -1)
                if nearE and nearE.u then
                    u:attackUnit(nearE.u, true)
                    r.lastOrderKind = "attack"
                    r.lastTargetUid = nearE.u.unitID or tostring(nearE.u)
                    r.lastAtkRun = true
                    r.lastOrderT = now
                    r.nativePathUntil = now + 12.0
                    r.progX, r.progY, r.progT = ux, uy, now
                    return
                end
            end
        end
    end
    -- ---- Layer 3 reflexes (matching sim/rl_env.py) ----
    if not eng then
        local nearE, d0 = nearestEnemy(F, fr, r, -1)
        if nearE then
            -- 1) Missile units skirmish back when enemy closes within 35m (cav) / 40m (foot)
            local hasAmmo = ((u.maxAmmo or 0) == 0 or (u.currentAmmo or 0) > 0)
            local skDist = r.info.cav and 35.0 or 40.0
            if (r.role == "missile_foot" or r.role == "missile_cav") and hasAmmo and d0 < skDist then
                local dist = math.max(1.0, d0)
                local tx = u.battlePosX - (nearE.u.battlePosX - u.battlePosX) / dist * 60.0
                local ty = u.battlePosY - (nearE.u.battlePosY - u.battlePosY) / dist * 60.0
                issueMove(u, r, tx, ty, true)
                return
            end
            -- 2) Spearmen auto-brace (hold) when enemy cavalry is within 70m and not yet in melee
            if r.role == "spear_inf" and (a < 5 or a > 8) and d0 > 25.0 then
                for _, o in ipairs(F.enemy) do
                    if active(o) and o.info.cav then
                        local cd = math.sqrt((o.u.battlePosX - u.battlePosX)^2 + (o.u.battlePosY - u.battlePosY)^2)
                        if cd <= 70.0 then
                            issueHold(u, r)
                            return
                        end
                    end
                end
            end
        end
    end

    if a == 8 then
        if not eng then issueHold(u, r) end
        return
    end
    if a <= 7 then
        local nearE, ed = nearestEnemy(F, fr, r, -1)
        local hasAmmo = ((u.maxAmmo or 0) == 0 or (u.currentAmmo or 0) > 0)
        if not eng and nearE and ed <= 25.0 and not (r.info.missile and hasAmmo) then
            issueAttack(u, r, nearE, true)
            return
        end
        local o = MOVES[a]
        local tx = u.battlePosX + o[1] * fr.rx + o[2] * fr.fx
        local ty = u.battlePosY + o[1] * fr.ry + o[2] * fr.fy
        issueMove(u, r, tx, ty, ed < 150 or isUnderFire(u))
    else
        -- Do not re-issue attackUnit if already engaged in melee (prevents reforming/canceling melee swings)
        if eng then return end
        local tgt = nearestEnemy(F, fr, r, a - 9)
        -- 3) Shock cavalry anti-spear target filter: avoid charging spearmen when other targets are within 135m
        if r.role == "shock_cav" and tgt and tgt.role == "spear_inf" then
            local bestNs, bestD = nil, 135.0
            for _, o in ipairs(F.enemy) do
                if active(o) and o.role ~= "spear_inf" then
                    local od = math.sqrt((o.u.battlePosX - u.battlePosX)^2 + (o.u.battlePosY - u.battlePosY)^2)
                    if od < bestD then bestD, bestNs = od, o end
                end
            end
            if bestNs then tgt = bestNs end
        end
        if tgt then
            local dx = (tgt.u.battlePosX or 0) - (u.battlePosX or 0)
            local dy = (tgt.u.battlePosY or 0) - (u.battlePosY or 0)
            local td = math.sqrt(dx * dx + dy * dy)
            local hasAmmo = ((u.maxAmmo or 0) == 0 or (u.currentAmmo or 0) > 0)
            if r.info.missile and hasAmmo then
                local engageDist = (r.info.e and r.info.e.range and r.info.e.range > 0) and (r.info.e.range * 0.9) or 140
                if td > engageDist then
                    issueMove(u, r, tgt.u.battlePosX, tgt.u.battlePosY, false)
                else
                    issueAttack(u, r, tgt, false)
                end
            else
                -- Use M2TW's native attackUnit pathfinding directly (walks at long range, runs/charges when close)
                issueAttack(u, r, tgt, td < 110 or isUnderFire(u))
            end
        else
            issueHold(u, r)
        end
    end
end

-- ---------- replay ----------
local SYM = { line_inf = "I", spear_inf = "S", missile_foot = "A", shock_cav = "C", missile_cav = "C", general = "G", artillery = "T" }
local ST = { [6] = "R", [5] = "F", [4] = "A", [3] = "H" }
local function replayFrame(F, t)
    local units, minx, miny, maxx, maxy = {}, 1e9, 1e9, -1e9, -1e9
    for side, list in ipairs({ F.own, F.enemy }) do
        for i, r in ipairs(list) do
            local x, y = r.u.battlePosX, r.u.battlePosY
            minx, miny, maxx, maxy = math.min(minx, x), math.min(miny, y), math.max(maxx, x), math.max(maxy, y)
            local sym = SYM[r.role]; if side == 2 then sym = string.lower(sym) end
            local foes = (side == 1) and F.enemy or F.own
            units[#units + 1] = { side, i, math.floor(x), math.floor(y), math.floor(100 * alive(r) / r.start), r.u.moraleLevel or 0, status(r, foes), r.lastAction or -1, sym, routing(r) and 1 or 0 }
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

local function applyDisadvantage(own, en, battleId)
    local numerical = (battleId % 2 == 0)
    local appliedNum = false
    if numerical then
        for _, r in ipairs(own) do
            if r.role ~= "general" then
                local before = r.u.soldierCountBattleMap or r.start
                local target = math.max(1, math.floor(before * 0.70))
                pcall(function() r.u.soldierCountBattleMap = target end)
                local after = r.u.soldierCountBattleMap or before
                if after < before then
                    r.start = after
                    appliedNum = true
                end
            end
        end
    end
    if not numerical or not appliedNum then
        -- Quality disadvantage: buff enemy attack/armour (+3) and wind/weaken own units
        for _, r in ipairs(en) do
            pcall(function()
                r.u.armourInBattle = (r.u.armourInBattle or 0) + 3
                r.u.attackInBattle = (r.u.attackInBattle or 0) + 3
            end)
        end
        for _, r in ipairs(own) do
            pcall(function()
                r.u.armourInBattle = math.max(0, (r.u.armourInBattle or 0) - 1)
                r.u.attackInBattle = math.max(1, (r.u.attackInBattle or 1) - 1)
                r.u.fatigue = 30
            end)
        end
    end
    print("[RL_AI] Phase 3 disadvantage applied (" .. (appliedNum and "numerical 70%" or "quality +3 enemy / -1 & winded own") .. ")")
end

-- ---------- public hooks ----------
function M.onBattleAiCommenced()
    local sc = json.decode(readfile(WS .. "scenario.json") or "{}") or {}
    local wid = "1"
    local wid_f = io.open("youneuoy_Data/plugins/lua/worker_id.txt", "r")
    if wid_f then wid = wid_f:read("*l") or "1"; wid_f:close() end
    local wbid = tonumber(readfile(WS .. "worker_battle_id_" .. wid .. ".txt") or "")
    local bid = wbid or sc.battle_id or 0
    local arch = sc.arch or "legacy"
    local wtxt = (arch == "lstm") and nil or readfile(WS .. "weights_latest.json")
    local ctxt = (arch == "lstm") and nil or readfile(WS .. "commander_rnn.json")
    S = { battleId = bid, wid = wid, arch = arch,
        battleType = (sc.terrain and sc.terrain.battle_type) or "open_field",
        mode = sc.mode or "rl_vs_vanilla", invTemp = sc.inv_temp or (arch == "lstm" and 1.0 or 2.2),
        t = 0, nextTick = TICK, done = false,
        longBattle = (bid % 5 == 0),
        W = wtxt and json.decode(wtxt) or nil,
        CRNN = ctxt and json.decode(ctxt) or nil,
        cmdH = zeroHidden(), prevGlobalCmd = nil, prevSubCmds = {},
        lstmSteps = {}, nextLstmTick = 0, lstmTickIdx = 0,
        frames = {}, tickN = 0 }
    if S.mode == "rl_vs_frozen" then
        S.WF = json.decode(readfile(WS .. "weights_frozen.json") or "null")
        S.CRNNF = json.decode(readfile(WS .. "commander_rnn_frozen.json") or "null")
    end
    local b = getBattle()
    if not b then
        print("[RL_AI] ERROR: getBattle() returned nil")
        return
    end
    print("[RL_AI] Battle commenced! State=" .. tostring(b.battleState) .. " ID=" .. tostring(S.battleId))
    pcall(function() b.battleSpeed = 85.0 end)                               -- 50x simulation speed
    S.F = collect(b)
    local own, en = S.F[1], S.F[2]
    if S.mode == "rl_vs_vanilla_disadvantaged" then
        applyDisadvantage(own, en, S.battleId)
    end
    S.F = { own = own, enemy = en, ownStart = 0, enStart = 0 }
    for _, r in ipairs(own) do S.F.ownStart = S.F.ownStart + r.start end
    for _, r in ipairs(en) do S.F.enStart = S.F.enStart + r.start end
    S.F.ownStart = math.max(1, S.F.ownStart)
    S.F.enStart = math.max(1, S.F.enStart)
    S.prevOwnLost, S.prevEnLost, S.prevGround = 0, 0, 0.5
    S.nextCmdTick = 0
    S.subCmds = {}
    S.queue = {}
    for i, r in ipairs(own) do S.queue[i] = r end
    S.pending = {}
    if S.arch == "lstm" then
        local fr0 = frame(S.F)
        local function bindSlots(list)
            local tmp = {}
            for idx, r in ipairs(list) do
                local rx, _ = toRel(fr0, r.u.battlePosX or 0, r.u.battlePosY or 0)
                tmp[#tmp + 1] = { ri = r.ri or 1, rx = rx, idx = idx, r = r }
            end
            table.sort(tmp, function(a, b)
                if a.ri ~= b.ri then return a.ri < b.ri end
                if a.rx ~= b.rx then return a.rx < b.rx end
                return a.idx < b.idx
            end)
            local slots = {}
            for i = 1, math.min(24, #tmp) do slots[i] = tmp[i].r end
            return slots
        end
        S.ownSlots = bindSlots(own)
        S.enSlots = bindSlots(en)
        S.prevLstmOwnAlive = S.F.ownStart
        S.prevLstmEnAlive = S.F.enStart
        S.prevSlotAlive, S.prevSlotKilled = {}, {}
        for i = 1, 24 do
            local r = S.ownSlots[i]
            S.prevSlotAlive[i] = r and alive(r) or 0
            S.prevSlotKilled[i] = r and (r.bu.soldiersKilled or 0) or 0
        end
    end
end

-- ---------- Full-Army LSTM observation & IPC ----------
local LSTM_MAX_UNITS = 24
local LSTM_UNIT_FEAT = 26
local LSTM_GLOBAL_FEAT = 12
local LSTM_OBS_DIM = LSTM_GLOBAL_FEAT + 2 * LSTM_MAX_UNITS * LSTM_UNIT_FEAT

local function isOnWalls(u)
    local ok, v = pcall(function() return u:isOnWalls() end)
    return ok and (v == true or v == 1)
end

local function encodeLstmUnit(r, fr, obs, off, meanZ)
    if not r or alive(r) <= 0 then return end
    local rx, ry = toRel(fr, r.u.battlePosX or 0, r.u.battlePosY or 0)
    local e, info = r.info.e, r.info
    obs[off + 1] = 1.0
    obs[off + 2] = clamp(rx / 300.0, -1.5, 1.5)
    obs[off + 3] = clamp(ry / 300.0, -1.5, 1.5)
    obs[off + 4] = clamp(alive(r) / math.max(1, r.start), 0.0, 1.0)
    obs[off + 5] = clamp((6 - (r.u.moraleLevel or 2)) / 6.0, 0.0, 1.0)
    obs[off + 6] = clamp((r.u.fatigue or 0) / 6.0, 0.0, 1.0)
    if r.role == "artillery" then
        obs[off + 7] = ((r.u.maxAmmo or 0) <= 0 or (r.u.currentAmmo or 0) > 0) and 1.0 or 0.0
    else
        obs[off + 7] = (r.u.maxAmmo or 0) > 0 and clamp((r.u.currentAmmo or 0) / r.u.maxAmmo, 0.0, 1.0) or 0.0
    end
    obs[off + 8] = isEngaged(r.u) and 1.0 or 0.0
    obs[off + 9] = isUnderFire(r.u) and 1.0 or 0.0
    obs[off + 10] = routing(r) and 1.0 or 0.0
    local ri = r.ri or 1
    if ri >= 1 and ri <= 7 then obs[off + 10 + ri] = 1.0 end
    local atk = (e and e.aiUnitValuePerSoldier) or 5
    local rng = (info.missile and e and e.range) or 0
    local baseMor = 5
    pcall(function() if e and e.morale then baseMor = e.morale end end)
    obs[off + 18] = clamp(atk / 25.0, 0.0, 1.0)
    obs[off + 19] = clamp(((r.u.armourInBattle or 5) + (r.u.defenceInBattle or 5)) / 30.0, 0.0, 1.0)
    obs[off + 20] = clamp(rng / 250.0, 0.0, 1.0)
    obs[off + 21] = clamp((e and e.moveSpeedMod or 1.0) / 2.5, 0.0, 1.0)
    obs[off + 22] = clamp((e and e.mass or 1.0) / 5.0, 0.0, 1.0)
    obs[off + 23] = clamp(baseMor / 15.0, 0.0, 1.0)
    obs[off + 24] = clamp((r.start or 60) / 150.0, 0.0, 1.5)
    local z = r.u.battlePosZ or 0.0
    obs[off + 25] = clamp((z - (meanZ or 0.0)) / 25.0, -1.5, 1.5)
    obs[off + 26] = isOnWalls(r.u) and 1.0 or 0.0
end

local function buildLstmObs(F, fr, t)
    local obs, mask = {}, {}
    for i = 1, LSTM_OBS_DIM do obs[i] = 0.0 end
    for i = 1, LSTM_MAX_UNITS do mask[i] = 0.0 end
    -- 12 global features
    local bt = S.battleType or "open_field"
    if S.siegeInfo and S.siegeInfo.isSiege then
        bt = S.siegeInfo.isAttacker and "siege_assault" or "siege_defense"
    end
    if bt == "river_crossing" then obs[2] = 1.0
    elseif bt == "siege_assault" then obs[3] = 1.0
    elseif bt == "siege_defense" then obs[4] = 1.0
    else obs[1] = 1.0 end
    obs[5] = clamp((F.ownAlive or 0) / math.max(1, F.ownStart or 1), 0.0, 1.0)
    obs[6] = clamp((F.enAlive or 0) / math.max(1, F.enStart or 1), 0.0, 1.0)
    local om, on, em, en = 0.0, 0, 0.0, 0
    for _, r in ipairs(F.own) do if alive(r) > 0 then om = om + clamp((6 - (r.u.moraleLevel or 2)) / 6.0, 0.0, 1.0); on = on + 1 end end
    for _, r in ipairs(F.enemy) do if alive(r) > 0 then em = em + clamp((6 - (r.u.moraleLevel or 2)) / 6.0, 0.0, 1.0); en = en + 1 end end
    obs[7] = on > 0 and (om / on) or 0.0
    obs[8] = en > 0 and (em / en) or 0.0
    obs[9] = clamp(t / 900.0, 0.0, 1.0)
    obs[10] = F.ownGeneralAlive and 1.0 or 0.0
    obs[11] = F.enGeneralAlive and 1.0 or 0.0
    if S.siegeInfo and S.siegeInfo.isSiege then
        obs[12] = math.max(S.siegeInfo.breached and 1.0 or 0.0, S.siegeInfo.plazaFrac or 0.0)
    end
    local zSum, zCnt = 0.0, 0
    for _, r in ipairs(F.own) do if alive(r) > 0 then zSum = zSum + (r.u.battlePosZ or 0.0); zCnt = zCnt + 1 end end
    for _, r in ipairs(F.enemy) do if alive(r) > 0 then zSum = zSum + (r.u.battlePosZ or 0.0); zCnt = zCnt + 1 end end
    local meanZ = zCnt > 0 and (zSum / zCnt) or 0.0
    -- 24 friendly slots
    for i = 1, LSTM_MAX_UNITS do
        local r = S.ownSlots and S.ownSlots[i]
        if r and alive(r) > 0 then
            encodeLstmUnit(r, fr, obs, LSTM_GLOBAL_FEAT + (i - 1) * LSTM_UNIT_FEAT, meanZ)
            if active(r) then mask[i] = 1.0 end
        end
    end
    -- 24 enemy slots
    local enBase = LSTM_GLOBAL_FEAT + LSTM_MAX_UNITS * LSTM_UNIT_FEAT
    for i = 1, LSTM_MAX_UNITS do
        local r = S.enSlots and S.enSlots[i]
        if r and alive(r) > 0 then
            encodeLstmUnit(r, fr, obs, enBase + (i - 1) * LSTM_UNIT_FEAT, meanZ)
        end
    end
    return obs, mask
end

local function queryLstmServer(obs, mask)
    local wid = S.wid or "1"
    local reqTmp = WS .. "infer_req_" .. wid .. ".tmp"
    local reqPath = WS .. "infer_req_" .. wid .. ".json"
    local resPath = WS .. "infer_res_" .. wid .. ".json"
    pcall(function() os.remove(resPath) end)
    local payload = json.encode({
        battle_id = S.battleId, tick = S.lstmTickIdx,
        obs = obs, mask = mask, inv_temp = S.invTemp or 1.0, greedy = false,
    })
    writefile(reqTmp, payload)
    os.rename(reqTmp, reqPath)
    local t0 = os.clock()
    while (os.clock() - t0) < 0.25 do
        local raw = readfile(resPath)
        if raw and #raw > 10 then
            local ok, res = pcall(json.decode, raw)
            if ok and res and res.tick == S.lstmTickIdx and res.actions then
                pcall(function() os.remove(resPath) end)
                return res.actions
            end
        end
    end
    -- Fallback if server temporarily busy: Hold (8)
    local fb = {}
    for i = 1, LSTM_MAX_UNITS do fb[i] = 8 end
    return fb
end

local function totals(F)
    local oa, ea, og, eg, oact, eact = 0, 0, false, false, 0, 0
    for _, r in ipairs(F.own) do
        oa = oa + alive(r)
        if active(r) then oact = oact + 1; if r.role == "general" then og = true end end
    end
    for _, r in ipairs(F.enemy) do
        ea = ea + alive(r)
        if active(r) then eact = eact + 1; if r.role == "general" then eg = true end end
    end
    F.ownAlive, F.enAlive, F.ownGeneralAlive, F.enGeneralAlive = oa, ea, og, eg
    F.ownActive, F.enActive = oact, eact
end

-- 3 dynamic spatial flank squads (1=left, 2=center, 3=right) matching sim/rl_env.py
local function assignSquads(list, infCx, infCy, fr)
    local lead, leadBest = nil, 1e9
    for _, r in ipairs(list) do
        if active(r) then
            local rx = (r.u.battlePosX - infCx) * fr.rx + (r.u.battlePosY - infCy) * fr.ry
            if rx < -45 then r.squadKey = 1
            elseif rx > 45 then r.squadKey = 3
            else r.squadKey = 2 end
            if r.role == "line_inf" and math.abs(rx) < leadBest then
                leadBest = math.abs(rx)
                lead = r
            end
        end
    end
    return lead
end

local function updateCommander(list, W, eps, fr, g, F, learner)
    local infCx, infCy = infantryCentroid(list, fr)
    local lead = assignSquads(list, infCx, infCy, fr)
    local subCmds, globalCmd = {}, 0
    local crnn = learner and S.CRNN or S.CRNNF
    if (crnn or W) then
        for i = 21, NG do g[i] = 0 end
        local basePhi = buildCommanderBasePhi(F, fr, g)
        local h = (learner and S.cmdH) or (not learner and S.cmdHF) or zeroHidden()

        -- Step 0: Global Commander decision (unnamed order 0..9)
        local gExtras = { { 252, 1.0 } }
        local prevG = (learner and S.prevGlobalCmd or S.prevGlobalCmdF) or 0
        gExtras[#gExtras + 1] = { 263 + prevG, 1.0 }
        gExtras[#gExtras + 1] = { 273 + prevG, 1.0 }
        local cmdPhi = makeStepPhi(basePhi, gExtras)
        local hBeforeGlobal = copyHidden(h)
        globalCmd, h = stepCommanderRNN(crnn, cmdPhi, h, eps)
        g[21 + globalCmd] = 1
        if learner then S.pendingCmd = { { phi = cmdPhi, h = hBeforeGlobal, a = globalCmd } } end

        -- Steps 1..3: Detachment orders for Left (1), Center (2), Right (3) matching sim/rl_env.py
        local prevStepCmd = globalCmd
        for sq = 1, 3 do
            local sqUnits = {}
            local sx, sy, shp, sstart = 0, 0, 0, 0
            for _, r in ipairs(list) do
                if active(r) and r.squadKey == sq then
                    sqUnits[#sqUnits + 1] = r
                    sx = sx + r.u.battlePosX
                    sy = sy + r.u.battlePosY
                    shp = shp + alive(r)
                    sstart = sstart + r.start
                end
            end
            local sqDx, sqDy, sqHp, sqRi
            if #sqUnits > 0 then
                sx, sy = sx / #sqUnits, sy / #sqUnits
                local rx, ry = toRel(fr, sx, sy)
                sqDx = clamp(rx / 250, -1, 1)
                sqDy = clamp(ry / 250, -1, 1)
                sqHp = clamp(shp / math.max(1, sstart), 0, 1)
                sqRi = sqUnits[1].ri or 1
            else
                sqDx = clamp(((sq - 2) * 90) / 250, -1, 1)
                sqDy, sqHp, sqRi = 0, 0, 1
            end

            local sExtras = {
                { 253 + globalCmd, 1.0 },
                { 263 + prevStepCmd, 1.0 },
            }
            local ownPrev = (learner and S.prevSubCmds and S.prevSubCmds[sq]) or 0
            sExtras[#sExtras + 1] = { 273 + ownPrev, 1.0 }
            sExtras[#sExtras + 1] = { 282 + sqRi, 1.0 }
            sExtras[#sExtras + 1] = { 290, sqDx }
            sExtras[#sExtras + 1] = { 291, sqDy }
            sExtras[#sExtras + 1] = { 292, sqHp }

            local subPhi = makeStepPhi(basePhi, sExtras)
            local hBeforeSq = copyHidden(h)
            local scmd
            scmd, h = stepCommanderRNN(crnn, subPhi, h, eps)
            subCmds[sq] = scmd
            prevStepCmd = scmd
            if learner then
                S.pendingCmd[#S.pendingCmd + 1] = { phi = subPhi, h = hBeforeSq, a = scmd }
            end
        end
        if lead and lead.squadKey and subCmds[lead.squadKey] ~= nil then
            S.lastLeadCmd = subCmds[lead.squadKey]
        end
        g[41 + (S.lastLeadCmd or 0)] = 1
        if learner then
            S.cmdH = h
            S.prevGlobalCmd = globalCmd
            S.prevSubCmds = subCmds
        else
            S.cmdHF = h
            S.prevGlobalCmdF = globalCmd
            S.prevSubCmdsF = subCmds
        end
    end
    return subCmds, infCx, infCy, globalCmd
end

local function armySpread(F)
    local minX, minY, maxX, maxY, cnt = 1e9, 1e9, -1e9, -1e9, 0
    for _, o in ipairs(F.own) do
        if active(o) then
            local x, y = o.u.battlePosX or 0, o.u.battlePosY or 0
            if x < minX then minX = x end
            if y < minY then minY = y end
            if x > maxX then maxX = x end
            if y > maxY then maxY = y end
            cnt = cnt + 1
        end
    end
    if cnt < 2 then return 0.0 end
    local dx, dy = maxX - minX, maxY - minY
    return math.sqrt(dx * dx + dy * dy)
end

local function cohesionAndEngageReward(r, infCx, infCy, F)
    local dx, dy = (r.u.battlePosX or 0) - infCx, (r.u.battlePosY or 0) - infCy
    local d = math.sqrt(dx * dx + dy * dy)
    local isCav = (r.role == "shock_cav" or r.role == "missile_cav" or r.role == "general")
    local hard = isCav and 250 or 160
    local coh = (d <= hard) and 0.04 or -math.min(2.5, (d - hard) / 50)
    local spread = armySpread(F)
    if spread > 650 then
        coh = coh - math.min(0.6, (spread - 650) / 400)
    end
    return coh
end

local function stepUnit(r, W, eps, fr, g, F, infCx, infCy, subCmds, globalCmd, learner, ol, el, ground)
    local p = r.pending
    if p and learner then
        local dk = (r.bu.soldiersKilled - r.lastKilled) / 20
        local dl = (r.bu.soldiersLost - r.lastLost) / 20
        local gc = ((el - p.el) - (ol - p.ol)) / F.ownStart * 10
        local coh = cohesionAndEngageReward(r, infCx, infCy, F)
        local st = { phi = p.phi, h = p.h, a = p.a, dk = dk, dl = dl, gc = gc, gm = ground - p.ground, coh = coh }
        if S.mode == "vanilla_vs_vanilla" and p.frm then
            local dx, dy = r.u.battlePosX - p.sx, r.u.battlePosY - p.sy
            st.disp = { dx * p.frm.rx + dy * p.frm.ry, dx * p.frm.fx + dy * p.frm.fy }
            if r.u:isEngaged() then
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
        r.lastKilled, r.lastLost = r.bu.soldiersKilled, r.bu.soldiersLost
    end
    local hBefore = copyHidden(r.h, UNIT_RNN_HIDDEN)
    if W then
        for i = 31, 40 do g[i] = 0 end
        local sqCmd = subCmds[r.squadKey] or 0
        g[31 + sqCmd] = 1
        local feats = sparse(g, r, F, fr, infCx, infCy)
        local a
        a, r.h = chooseAction(W, r.ri, feats, r.h, eps, NA)
        issue(a, r, F, fr); r.lastAction = a
        if learner and S.mode ~= "vanilla_vs_vanilla" then
            r.pending = { phi = feats, h = hBefore, a = a, ol = ol, el = el, ground = ground, ed = curEd }
        end
    end
end

local function recordEnemyImitation(F, t)
    -- Observe Side 1 (genuine Medieval II Vanilla AI) from Side 1's own perspective for Phase 1 imitation
    local eF = {
        own = F.enemy, enemy = F.own,
        ownStart = F.enStart, enStart = F.ownStart,
        ownAlive = F.enAlive, enAlive = F.ownAlive,
        ownGeneralAlive = F.enGeneralAlive, enGeneralAlive = F.ownGeneralAlive,
    }
    local efr = frame(eF)
    local eg = globalFeatures(eF, efr, t)
    local eInfCx, eInfCy = infantryCentroid(eF.own, efr)
    for _, r in ipairs(eF.own) do
        if active(r) and r.ri and t >= (r.nextImitT or 0) then
            r.nextImitT = t + UNIT_COMMIT_SEC
            local p = r.pending
            if p and p.frm then
                local dx, dy = (r.u.battlePosX or 0) - p.sx, (r.u.battlePosY or 0) - p.sy
                local st = {
                    phi = p.phi, h = p.h, a = -1, dk = 0, dl = 0, coh = 0,
                    disp = { dx * p.frm.rx + dy * p.frm.ry, dx * p.frm.fx + dy * p.frm.fy },
                }
                if isEngaged(r.u) then
                    local tgt = nearestEnemy(eF, p.frm, r, -1)
                    if tgt then
                        local ex, ey = tgt.u.battlePosX - r.u.battlePosX, tgt.u.battlePosY - r.u.battlePosY
                        local ang = math.deg(math.atan2(ex * p.frm.rx + ey * p.frm.ry, ex * p.frm.fx + ey * p.frm.fy))
                        st.vq = (math.abs(ang) <= 45 and 0) or (ang < -45 and ang >= -135 and 1) or (ang > 45 and ang <= 135 and 2) or 3
                    end
                end
                r.steps[#r.steps + 1] = st
                r.pending = nil
            end
            local hBefore = copyHidden(r.h, UNIT_RNN_HIDDEN)
            local feats = sparse(eg, r, eF, efr, eInfCx, eInfCy)
            if S.W then _, r.h = unitRNNForward(S.W, r.ri, feats, r.h) end
            r.pending = { phi = feats, h = hBefore, a = -1, sx = r.u.battlePosX or 0, sy = r.u.battlePosY or 0, frm = efr }
        end
    end
end

local function actSide(list, W, eps, fr, g, F, learner)
    local subCmds, infCx, infCy, globalCmd = updateCommander(list, W, eps, fr, g, F, learner)
    for _, r in ipairs(list) do
        if active(r) and r.ri then
            stepUnit(r, W, eps, fr, g, F, infCx, infCy, subCmds, globalCmd, learner, 0, 0, 0.5)
        end
    end
end

local function finalizePendingLstmStep(F, fr)
    if not S.pendingLstm then return end
    local p = S.pendingLstm
    local infCx, infCy = infantryCentroid(F.own, fr)
    local dk = math.max(0.0, ((S.prevLstmEnAlive or F.enStart) - (F.enAlive or 0)) / math.max(1, F.enStart))
    local dl = math.max(0.0, ((S.prevLstmOwnAlive or F.ownStart) - (F.ownAlive or 0)) / math.max(1, F.ownStart))
    S.prevLstmEnAlive = F.enAlive
    S.prevLstmOwnAlive = F.ownAlive
    local u_dk, u_dl = {}, {}
    local cSum, cCnt = 0.0, 0
    for i = 1, LSTM_MAX_UNITS do
        local r = S.ownSlots and S.ownSlots[i]
        if r then
            local kNow = r.bu.soldiersKilled or 0
            local aNow = alive(r)
            u_dk[i] = math.max(0.0, (kNow - (S.prevSlotKilled[i] or 0)) / math.max(1, r.start))
            u_dl[i] = math.max(0.0, ((S.prevSlotAlive[i] or aNow) - aNow) / math.max(1, r.start))
            S.prevSlotKilled[i] = kNow
            S.prevSlotAlive[i] = aNow
            if active(r) then
                cSum = cSum + cohesionAndEngageReward(r, infCx, infCy, F)
                cCnt = cCnt + 1
            end
        else
            u_dk[i], u_dl[i] = 0.0, 0.0
        end
    end
    S.lstmSteps[#S.lstmSteps + 1] = {
        obs = p.obs, mask = p.mask, acts = p.acts,
        dk = dk, dl = dl, coh = cCnt > 0 and (cSum / cCnt) or 0.0,
        u_dk = u_dk, u_dl = u_dl,
    }
    S.pendingLstm = nil
end

function M.onBattleTick()
    if not S or S.done then return end
    local b = getBattle()
    if not b then return end
    local t = b.secondsPassed or 0
    if b.battleState ~= nil and t < S.nextTick then return end
    pcall(function() b.battleSpeed = 85.0 end)
    -- Early battle completion: decisive rout (<=20% alive or all units fleeing), or 1800s (30m) full battle timeout
    local F = S.F
    local allEnFleeing = (F and F.enActive ~= nil and F.enActive == 0)
    local allOwnFleeing = (F and F.ownActive ~= nil and F.ownActive == 0)
    local timeCap = 900
    if t >= timeCap or (t >= 60 and F and (F.ownAlive <= 0.2 * F.ownStart or F.enAlive <= 0.2 * F.enStart or allEnFleeing or allOwnFleeing)) then
        S.allEnFleeing = allEnFleeing
        S.timedOut = (t >= timeCap) and not (F and (F.enAlive <= 0.2 * F.enStart or allEnFleeing))
        print("[RL_AI] Battle reached decisive conclusion or " .. tostring(timeCap) .. "s cap. Completing episode.")
        M.onBattleFinished()
        return
    end
    local dt = math.max(1.0, t - (S.lastTickT or 0))
    S.t = t
    S.lastTickT = t
    S.nextTick = t + TICK; S.tickN = S.tickN + 1
    S.siegeInfo = updateSiegeInfo(b)
    totals(F)

    -- Full-Army LSTM mode: single forward pass every 4.5s gives orders to ALL friendly units at once
    if S.arch == "lstm" then
        if t >= (S.nextLstmTick or 0) then
            S.nextLstmTick = t + 4.5
            S.fr = frame(F)
            finalizePendingLstmStep(F, S.fr)
            local obs, mask = buildLstmObs(F, S.fr, t)
            local acts = queryLstmServer(obs, mask)
            for i = 1, LSTM_MAX_UNITS do
                local r = S.ownSlots and S.ownSlots[i]
                if r and active(r) then
                    local a = acts[i] or 8
                    issue(a, r, F, S.fr)
                    r.lastAct = a
                end
            end
            S.pendingLstm = { obs = obs, mask = mask, acts = acts }
            S.lstmTickIdx = (S.lstmTickIdx or 0) + 1
        end
        if (S.longBattle and S.tickN % 3 == 1) or (S.tickN % 5 == 1) then S.frames[#S.frames + 1] = replayFrame(F, t) end
        for _, list in ipairs({ F.own, F.enemy }) do
            for _, r in ipairs(list) do r.prevPx, r.prevPy = r.u.battlePosX, r.u.battlePosY end
        end
        return
    end

    local ol, el = F.ownStart - F.ownAlive, F.enStart - F.enAlive
    local ground = (strength(F.own) - strength(F.enemy)) / math.max(1, strength(F.own) + strength(F.enemy)) * 0.5 + 0.5

    -- Commander updates global + squad tactics every CMD_INTERVAL (12s)
    if t >= (S.nextCmdTick or 0) or not S.g then
        S.nextCmdTick = t + CMD_INTERVAL
        S.fr = frame(F)
        local infCx, infCy = infantryCentroid(F.own, S.fr)
        if S.pendingCmd and #S.pendingCmd > 0 then
            local totK, nAct = 0, 0
            local cCoh = 0
            for _, r in ipairs(F.own) do
                totK = totK + (r.bu.soldiersKilled or 0)
                if active(r) then
                    nAct = nAct + 1
                    cCoh = cCoh + cohesionAndEngageReward(r, infCx, infCy, F)
                end
            end
            local denom = 20.0 * math.max(1, nAct)
            local cDk = (totK - (S.prevCmdTotKilled or 0)) / denom
            local cDl = math.max(0, (S.prevCmdTotAlive or F.ownStart) - F.ownAlive) / denom
            cCoh = cCoh / math.max(1, nAct)
            S.allCmdSteps = S.allCmdSteps or {}
            for _, pc in ipairs(S.pendingCmd) do
                S.allCmdSteps[#S.allCmdSteps + 1] = {
                    phi = pc.phi, h = pc.h, a = pc.a, dk = cDk, dl = cDl, coh = cCoh,
                }
            end
            S.pendingCmd = nil
            S.prevCmdTotKilled = totK
            S.prevCmdTotAlive = F.ownAlive
        end

        S.g = globalFeatures(F, S.fr, t)
        S.subCmds, S.infCx, S.infCy, S.globalCmd = updateCommander(F.own, S.W, EPS.learner, S.fr, S.g, F, true)
        if S.mode == "rl_vs_frozen" and S.WF then
            local efr = frame({ own = F.enemy, enemy = F.own })
            local eF = { own = F.enemy, enemy = F.own, ownStart = F.enStart, enStart = F.ownStart, ownAlive = F.enAlive, enAlive = F.ownAlive,
                ownGeneralAlive = F.enGeneralAlive, enGeneralAlive = F.ownGeneralAlive }
            actSide(F.enemy, S.WF, EPS.frozen, efr, globalFeatures(eF, efr, t), eF, false)
        end
        for _, list in ipairs({ F.own, F.enemy }) do for _, r in ipairs(list) do r.px, r.py = r.u.battlePosX, r.u.battlePosY end end
    end

    -- Round-robin unit queue with per-unit commitment window (UNIT_COMMIT_SEC = 6s, or early wakeup on engagement change)
    local q = S.queue or {}
    local maxProcess = math.max(UNITS_PER_SEC, math.ceil(dt * UNITS_PER_SEC))
    local processed, attempts, qlen = 0, 0, #q
    while processed < maxProcess and attempts < qlen and #q > 0 do
        local r = table.remove(q, 1)
        attempts = attempts + 1
        if active(r) and r.ri then
            local eng = isEngaged(r.u)
            local ready = (t >= (r.nextActTime or 0)) or (r.wasEngaged ~= nil and eng ~= r.wasEngaged)
            if ready then
                stepUnit(r, S.W, EPS.learner, S.fr, S.g, F, S.infCx, S.infCy, S.subCmds, S.globalCmd, true, ol, el, ground)
                r.nextActTime = t + UNIT_COMMIT_SEC
                r.wasEngaged = eng
                processed = processed + 1
            end
            q[#q + 1] = r
        end
    end

    if S.mode == "vanilla_vs_vanilla" then
        recordEnemyImitation(F, t)
    end
    if (S.longBattle and S.tickN % 3 == 1) or (S.tickN % 5 == 1) then S.frames[#S.frames + 1] = replayFrame(F, t) end
    -- Update prevPx/prevPy AFTER unit queue & replay capture so isUnitMoving() measures actual motion over the last tick
    for _, list in ipairs({ F.own, F.enemy }) do
        for _, r in ipairs(list) do r.prevPx, r.prevPy = r.u.battlePosX, r.u.battlePosY end
    end
end

function M.onBattleFinished()
    if not S or S.done then return end
    S.done = true
    if S.F then
        totals(S.F)
        if S.arch == "lstm" then
            finalizePendingLstmStep(S.F, S.fr or frame(S.F))
        end
        local ol, el = S.F.ownStart - S.F.ownAlive, S.F.enStart - S.F.enAlive
        local ground = (strength(S.F.own) - strength(S.F.enemy)) / math.max(1, strength(S.F.own) + strength(S.F.enemy)) * 0.5 + 0.5
        local infCx, infCy = S.infCx or (S.fr and S.fr.cx) or 0, S.infCy or (S.fr and S.fr.cy) or 0
        for _, r in ipairs(S.F.own) do
            if r.pending then
                local p = r.pending
                local dk = (r.bu.soldiersKilled - r.lastKilled) / 20
                local dl = (r.bu.soldiersLost - r.lastLost) / 20
                local gc = ((el - p.el) - (ol - p.ol)) / S.F.ownStart * 10
                local coh = cohesionAndEngageReward(r, infCx, infCy, S.F)
                r.steps[#r.steps + 1] = { phi = p.phi, h = p.h, a = p.a, dk = dk, dl = dl, gc = gc, gm = ground - p.ground, coh = coh }
                r.pending = nil
            end
        end
        if S.pendingCmd and #S.pendingCmd > 0 then
            local totK, nAct, cCoh = 0, 0, 0
            for _, r in ipairs(S.F.own) do
                totK = totK + (r.bu.soldiersKilled or 0)
                if active(r) then
                    nAct = nAct + 1
                    cCoh = cCoh + cohesionAndEngageReward(r, infCx, infCy, S.F)
                end
            end
            local denom = 20.0 * math.max(1, nAct)
            local cDk = (totK - (S.prevCmdTotKilled or 0)) / denom
            local cDl = math.max(0, (S.prevCmdTotAlive or S.F.ownStart) - S.F.ownAlive) / denom
            cCoh = cCoh / math.max(1, nAct)
            S.allCmdSteps = S.allCmdSteps or {}
            for _, pc in ipairs(S.pendingCmd) do
                S.allCmdSteps[#S.allCmdSteps + 1] = {
                    phi = pc.phi, h = pc.h, a = pc.a, dk = cDk, dl = cDl, coh = cCoh,
                }
            end
            S.pendingCmd = nil
        end
    end
    local b = getBattle()
    local won = 1
    pcall(function()
        if b and b.sides then
            local s = (b.sides[0] ~= nil) and b.sides[0] or b.sides[1]
            won = s and s.wonBattle or 1
        end
    end)
    -- If engine hasn't triggered win/loss yet (unresolved / time limit / draw):
    if won < 0 or won == 1 then
        if S.allEnFleeing then
            won = 2
        elseif S.F and S.F.ownStart and S.F.ownStart > 0 and S.F.enStart and S.F.enStart > 0 then
            local ownSurv = S.F.ownAlive / S.F.ownStart
            local enSurv = S.F.enAlive / S.F.enStart
            local enDmg = 1.0 - enSurv
            if S.timedOut and enDmg < 0.50 and S.mode ~= "vanilla_vs_vanilla" then
                won = 0
            else
                won = (ownSurv >= enSurv) and 2 or 0
            end
        end
    end
    local units, roster = {}, {}
    local trainList = (S.mode == "vanilla_vs_vanilla") and S.F.enemy or S.F.own
    for i, r in ipairs(trainList) do
        if #r.steps > 0 then units[#units + 1] = { id = i, role = r.role, steps = r.steps } end
    end
    for i, r in ipairs(S.F.own) do roster[#roster + 1] = { side = 0, id = i, role = r.role, start = r.start } end
    if S.mode ~= "vanilla_vs_vanilla" and S.allCmdSteps and #S.allCmdSteps > 0 then
        units[#units + 1] = { id = 0, role = "commander", steps = S.allCmdSteps }
    end
    for i, r in ipairs(S.F.enemy) do roster[#roster + 1] = { side = 1, id = i, role = r.role, start = r.start } end
    local wid = "1"
    local wid_f = io.open("youneuoy_Data/plugins/lua/worker_id.txt", "r")
    if wid_f then wid = wid_f:read("*l") or "1"; wid_f:close() end
    local fileTag = tostring(S.battleId)
    local enDmgFrac = (S.F and S.F.enStart and S.F.enStart > 0)
        and clamp((S.F.enStart - (S.F.enAlive or 0)) / S.F.enStart, 0, 1) or 0
    local ownSurvFrac = (S.F and S.F.ownStart and S.F.ownStart > 0)
        and (S.F.ownAlive / S.F.ownStart) or 0
    local decisiveWin = (won - 1 == 1) and ((1.0 - enDmgFrac) < 0.35 or S.allEnFleeing == true)
    local decisiveLoss = (won - 1 < 0) and (ownSurvFrac < 0.35)
    local ep = {
        battle_id = S.battleId, arch = S.arch, long_battle = S.longBattle, decisive_win = decisiveWin, decisive_loss = decisiveLoss,
        enemy_damage_frac = enDmgFrac, replay_file = string.format("replay_%s.json", fileTag),
        mode = S.mode, result = won - 1, tick_seconds = TICK, units = units,
    }
    if S.arch == "lstm" then
        ep.steps = S.lstmSteps or {}
    end
    -- write replay first, episode last (trainer polls for episode files)
    writefile(WS .. string.format("replays/replay_%s.json", fileTag),
        json.encode({ battle_id = S.battleId, long_battle = S.longBattle, mode = S.mode, roster = roster, frames = S.frames, result = won - 1 }))
    writefile(WS .. string.format("episodes/episode_%s.json", fileTag), json.encode(ep))
    writefile(WS .. string.format("last_battle_result_%s.txt", wid), string.format("Battle %d finished: result=%d ticks=%d\n", S.battleId, won - 1, S.tickN))
    pcall(function() M2TWEOP.quitGame() end)                                  -- [VERIFY] exit so runner restarts with next scenario
end

return M
