-- Marker file to confirm Lua engine execution
local _f = io.open("Z:/home/tomek/workspace/m2tweop_loaded.txt", "w") or io.open("C:/workspace/m2tweop_loaded.txt", "w") or io.open("m2tweop_loaded.txt", "w")
if _f then
    _f:write("LUA_PLUGIN_LOADED_OK\n")
    _f:close()
end

require('myconfigs')

-- Helper for managing persistence of tables across save/load
require('helpers/tableSave')

-- Uncomment to use EOP Helper functions
-- require('helpers/EopLuaHelpers')

-- Uncomment to use external debugger
-- require('helpers/mobdebug').start()

-- Our campaign config table.
campaignConfig = { ["someConfigValue"] = 5 };

-- Fires when loading a save file
function onLoadSaveFile(paths)
    campaignPopup = true;

    for index, path in pairs(paths) do
        if (string.find(path, "configTable.lua"))
        then
            -- Function from helper, load saved table
            campaignConfig = persistence.load(path);
        end
    end
end

-- Fires when creating a save file
-- Returns a list of M2TWEOP save files
function onCreateSaveFile()
    local savefiles = {};
    currentPath = M2TWEOP.getPluginPath();

    -- Function from helper, save our table
    persistence.store(currentPath .. "configTable.lua", campaignConfig);

    savefiles[1] = currentPath .. "configTable.lua";
    return savefiles;
end

-- Fires when the plugin is first loaded at game start or restarted with restartLua()
function onPluginLoad()
    local f = io.open("Z:/home/tomek/workspace/m2tweop_loaded.txt", "w")
    if f then
        f:write("M2TWEOP LUA INITIALIZED AT " .. tostring(os.time()) .. "\n")
        f:close()
    end
    print("[M2TWEOP] onPluginLoad triggered!")
    M2TWEOP.unlockGameConsoleCommands();
    -- UNCOMMENT TO ENABLE BELOW SETTINGS
    --M2TWEOP.setAncillariesLimit(8);
    --M2TWEOP.setMaxBgSize(31);
    --M2TWEOP.setReligionsLimit(10);
    --M2TWEOP.setBuildingChainLimit(9);
    --M2TWEOP.setGuildCooldown(3);
end


--- Called after loading the campaign map
function onCampaignMapLoaded() 
    GAME_DATA = gameDataAll.get()
    CAMPAIGN = GAME_DATA.campaignStruct
    STRAT_MAP = GAME_DATA.stratMap
    BATTLE = GAME_DATA.battleStruct
    UI_MANAGER = GAME_DATA.uiCardManager
end


--- Called every time an image is rendered for display
---@param pDevice LPDIRECT3DDEVICE9 
function draw(pDevice)
    if (ImGui.IsKeyPressed(ImGuiKey.GraveAccent))
    and (ImGui.IsKeyDown(ImGuiKey.LeftCtrl))
    then
        M2TWEOP.toggleConsole()
    elseif (ImGui.IsKeyPressed(ImGuiKey.GraveAccent))
    and (ImGui.IsKeyDown(ImGuiKey.LeftAlt))
    then
        M2TWEOP.toggleDeveloperMode()
    elseif (ImGui.IsKeyPressed(ImGuiKey.R))
        and (ImGui.IsKeyDown(ImGuiKey.LeftCtrl))
        and (ImGui.IsKeyDown(ImGuiKey.LeftShift))
    then
        M2TWEOP.restartLua()
    end
end

-- Wire up RL Battle AI module
package.path = package.path .. ";Z:/home/tomek/workspace/?.lua;./youneuoy_Data/plugins/lua/?.lua;./youneuoy_Data/plugins/lua/helpers/?.lua"
local rl_ok, rl_ai = pcall(require, 'rl_battle_ai')
local status_f = io.open("Z:/home/tomek/workspace/rl_status.txt", "w")
if status_f then
    status_f:write("LOAD_OK=" .. tostring(rl_ok) .. "\nMSG=" .. tostring(rl_ai) .. "\n")
    status_f:close()
end

local lastTick = 0
local inBattle = false
local lastHeartbeat = 0

local function logRL(msg)
    local f = io.open("Z:/home/tomek/workspace/rl_log.txt", "a")
    if f then
        f:write(os.date("[%H:%M:%S] ") .. tostring(msg) .. "\n")
        f:close()
    end
end

function onBattleDeploymentPhaseCommenced(eventData)
    logRL("onBattleDeploymentPhaseCommenced event fired")
end

function onBattleAiCommenced(eventData)
    logRL("onBattleAiCommenced event fired")
    inBattle = true
    lastTick = os.clock()
    if rl_ok and rl_ai and rl_ai.onBattleAiCommenced then
        local ok, err = pcall(rl_ai.onBattleAiCommenced)
        if not ok then logRL("ERR onBattleAiCommenced: " .. tostring(err)) end
    end
end

function onBattleConflictPhaseCommenced(eventData)
    logRL("onBattleConflictPhaseCommenced event fired")
    inBattle = true
    lastTick = os.clock()
    if rl_ok and rl_ai and rl_ai.onBattleAiCommenced then
        local ok, err = pcall(rl_ai.onBattleAiCommenced)
        if not ok then logRL("ERR onBattleAiCommenced: " .. tostring(err)) end
    end
end

function onBattleFinished(eventData)
    logRL("onBattleFinished event fired")
    inBattle = false
    if rl_ok and rl_ai and rl_ai.onBattleFinished then
        local ok, err = pcall(rl_ai.onBattleFinished)
        if not ok then logRL("ERR onBattleFinished: " .. tostring(err)) end
    end
end

function draw(pDevice)
    if (ImGui.IsKeyPressed(ImGuiKey.GraveAccent))
    and (ImGui.IsKeyDown(ImGuiKey.LeftCtrl))
    then
        M2TWEOP.toggleConsole()
    elseif (ImGui.IsKeyPressed(ImGuiKey.GraveAccent))
    and (ImGui.IsKeyDown(ImGuiKey.LeftAlt))
    then
        M2TWEOP.toggleDeveloperMode()
    elseif (ImGui.IsKeyPressed(ImGuiKey.R))
        and (ImGui.IsKeyDown(ImGuiKey.LeftCtrl))
        and (ImGui.IsKeyDown(ImGuiKey.LeftShift))
    then
        M2TWEOP.restartLua()
    end

    -- Direct state-driven battle tracking (robust to missing engine event dispatches)
    local gd = gameDataAll.get()
    local bs = gd and gd.battleStruct
    if bs and bs.battleState then
        local state = bs.battleState
        local now = os.clock()

        -- Heartbeat every 5s
        if now - lastHeartbeat >= 5.0 then
            lastHeartbeat = now
            local wid = "1"
            local wid_f = io.open("youneuoy_Data/plugins/lua/worker_id.txt", "r")
            if wid_f then wid = wid_f:read("*l") or "1"; wid_f:close() end
            local hb_str = "TIME=" .. tostring(math.floor(now)) .. " STATE=" .. tostring(state) .. " SECS=" .. tostring(bs.secondsPassed or 0) .. " PAUSED=" .. tostring(bs.paused or 0) .. " IN_BATTLE=" .. tostring(inBattle) .. "\n"
            local hf = io.open("Z:/home/tomek/workspace/rl_heartbeat_" .. tostring(wid) .. ".txt", "w")
            if hf then hf:write(hb_str); hf:close() end
            if tostring(wid) == "1" then
                local hf1 = io.open("Z:/home/tomek/workspace/rl_heartbeat.txt", "w")
                if hf1 then hf1:write(hb_str); hf1:close() end
            end
        end

        -- Auto-transition delay/deployment to conflict & 50x speed
        if state == 2 or state == 3 then
            pcall(function() bs.battleState = 5 end)
            pcall(function() bs.battleSpeed = 50.0 end)
        end

        if state == 5 or (state == 2 and bs.secondsPassed and bs.secondsPassed > 0) then
            -- Conflict phase active
            pcall(function() bs.battleSpeed = 50.0 end)
            if not inBattle then
                inBattle = true
                logRL("State 5 conflict detected! Initializing RL AI...")
                if rl_ok and rl_ai and rl_ai.onBattleAiCommenced then
                    local ok, err = pcall(rl_ai.onBattleAiCommenced)
                    if not ok then logRL("ERR onBattleAiCommenced: " .. tostring(err)) end
                end
            end

            lastTick = now
            if rl_ok and rl_ai and rl_ai.onBattleTick then
                local ok, err = pcall(rl_ai.onBattleTick)
                if not ok then
                    logRL("ERR onBattleTick: " .. tostring(err))
                end
            end
        elseif inBattle and (state >= 6 or state == 0) then
            -- Battle finished
            inBattle = false
            logRL("Battle finished detected! State=" .. tostring(state))
            if rl_ok and rl_ai and rl_ai.onBattleFinished then
                local ok, err = pcall(rl_ai.onBattleFinished)
                if not ok then logRL("ERR onBattleFinished: " .. tostring(err)) end
            end
        end
    end
end
