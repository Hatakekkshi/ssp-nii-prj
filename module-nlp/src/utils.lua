-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function convert_to_string(idx_list, dict_manager)

    if (dict_manager == nil) then
        dict_manager = form_manager
    end

    local w_list = {}
    local count_word = 0
    if (torch.type(idx_list) == 'table') then
        count_word = #idx_list
        for i = 1, count_word do
            if (idx_list[i] ~= 0) then
                table.insert(w_list, dict_manager:get_idx_symbol(idx_list[i]))
            end
        end
    elseif (torch.type(idx_list) == 'torch.DoubleTensor'
            or torch.type(idx_list) == 'torch.IntTensor'
            or torch.type(idx_list) == 'torch.LongTensor'
            or torch.type(idx_list) == 'torch.FloatTensor') then
        count_word = idx_list:size(1)
        for i = 1, count_word do
            if (idx_list[i] ~= 0) then
                table.insert(w_list, dict_manager:get_idx_symbol(idx_list[i][1]))
            end
        end
    else
        print('input not process : ' .. torch.type(idx_list))
    end

    return table.concat(w_list, ' ')
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function word_manager_convert_to_string(idx_list)
    return convert_to_string(idx_list, word_manager)
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
--- recusive copy table
function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function isSameTable(x, y)
    local bRet = true
    if (#x == #y) then
        for i = 1, #x do
            if (x[i] ~= y[i]) then
                bRet = false
                break
            end
        end
    else
        bRet = false
    end

    return bRet
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- read file to table 
function linesFromFile(sPathFile)
    local pairRets = {}
    for line in io.lines(sPathFile) do
        table.insert(pairRets, line)
    end
    return pairRets
end


-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- read file to table 
function cloneTable(table)
    local newTable = {}
    for k, v in pairs(table) do
        newTable[k] = v
    end
    return newTable
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- check value in table 
function tableContainValue(table, val)

    if torch.type(val) == 'table' then
        for k, v in pairs(table) do
            if (#val + k - 1) > #table then break end
            local bRet = true
            for k1, v1 in pairs(val) do
                if table[k + k1 - 1] ~= v1 then
                    bRet = false
                    break
                end
            end
            if (bRet == true) then return true, k + #val - 1 end
        end
    else
        for k, v in pairs(table) do
            if v == val then
                return true
            end
        end
    end
    return false
end

---
-- array to string, connect element in array by word save in connect
-- @param arr  list elelment
-- @param connect string connect between element
--
function arrToString(arr, connect)
    local sRet = ""
    if connect == nil then connect = " " end
    for k, v in pairs (arr) do
        if #v > 0 then
            v = v:gsub(connect, "_")
            sRet = sRet .. v .. connect
        end
    end
    sRet = sRet:sub(1, -(1+ #connect))
    return sRet
end

