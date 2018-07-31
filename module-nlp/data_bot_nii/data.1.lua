include "../src/SymbolsManager.lua"
include "../src/utils.lua"
local stringx = require "pl.stringx"
--torch.manualSeed(0)

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function serialize_data(opt, wordIdCombined, mapLabel2ListWordCheck, word2abbreviation)
    require('pl.stringx').import()
    require 'pl.seq'

    local fn = path.join(opt.data_dir, opt.nameFileTest .. '.txt')

    if not path.exists(fn) then
        print('no file: ' .. fn)
        return nil
    end

    local timer = torch.Timer()

    local word_manager, form_manager = table.unpack(torch.load(path.join(opt.data_dir, 'map.t7')))

    local data = {}

    print('loading text file...')
    local f = torch.DiskFile(fn, 'r', true)
    f:clearError()
    local rawdata = f:readString('*l')
    while (not f:hasError()) do

        rawdata = rawdata:strip()
        local _, __, l_list1, l_list2 = extractLineCombine(rawdata, wordIdCombined, mapLabel2ListWordCheck, word2abbreviation)
        local w_list = word_manager:get_symbol_idx_for_list(l_list1)
        local r_list = form_manager:get_symbol_idx_for_list(l_list2)
        table.insert(data, { w_list, r_list })

        -- read next line
        rawdata = f:readString('*l')
    end
    f:close()

    collectgarbage()

    -- save output preprocessed files
    local out_datafile = path.join(opt.data_dir, 'test.t7')

    print('saving ' .. out_datafile)
    torch.save(out_datafile, data)
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function splitDataTrainTest(pathData, rateTrain, idxFold, idxPointer)

    local rateTest = 0
    if (rateTrain == nil) then
        rateTrain = 0.8
    end
    rateTest = 1 - rateTrain

    -- max is 10 fold to cross validate
    if (idxFold == nil or idxFold > 10) then
        idxFold = 5
    end

    -- load all data to split tran-k and test-k of fold k
    local data = nil
    if (torch.type(pathData) == 'string') then
        data = torch.load(pathData)
    elseif (torch.type(pathData) == 'table') then
        data = pathData
    end

    assert(#data > 0, "[Err] data size err")
    local sizeData = #data
    local sizeDataTest = math.ceil(#data * rateTest)

    if idxPointer == nil then
        idxPointer = torch.randperm(sizeData)
    end
    local idx = idxPointer
    local idxStart = math.max(1, math.min(math.ceil(idxFold * sizeData / 10 + 1), sizeData))
    local idxEnd = math.max(1, math.min(math.ceil(idxStart + sizeDataTest - 1), sizeData))
    local dataTest, dataTrain = {}, {}
    for i = 1, sizeData do
        if (i >= idxStart and i <= idxEnd) then
            if idxFold == 4 then
                table.insert(dataTrain, data[idx[i]])
            end
            table.insert(dataTest, data[idx[i]])
        else
            table.insert(dataTrain, data[idx[i]])
        end
    end

    -- save data set
    local pathSaveTrain = opt.data_dir .. "train-" .. idxFold .. ".t7"
    print('saving split train: ' .. pathSaveTrain)
    torch.save(pathSaveTrain, dataTrain)

    local pathSaveTest = opt.data_dir .. "test-" .. idxFold .. ".t7"
    print('saving split test: ' .. pathSaveTest)
    torch.save(pathSaveTest, dataTest)
end

function distinct_data(data, idxFieldDistinct)

    local list = {}

    local function compareTable(table1, table2)
        if #table1 ~= #table2 then
            return false
        end
        for k1, v1 in pairs(table1) do
            if(table1[k1] ~= table2[k1]) then
                return false
            end
        end
        return true
    end

    local newTable = {}

    for i = 1, #data do
        local bCheckExist = false
        for j = 1, #newTable do
            if compareTable(data[i][idxFieldDistinct], newTable[j][idxFieldDistinct]) then
                bCheckExist = true
                break
            end
        end
        if not bCheckExist then
            table.insert(newTable,  data[i])
        end
    end
    return newTable
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:option('-data_dir', '', 'data directory')
cmd:option('-dataName', 'data.t7', 'train data binary path')
cmd:option('-dataNameLogicform', 'data.logicform.t7', 'data distinct logic form')
cmd:option('-min_freq', 0, 'minimum word frequency')
cmd:option('-max_vocab_size', 1500, 'maximum vocabulary size')
cmd:text()
opt = cmd:parse(arg)


-- word_manager = SymbolsManager(true)
-- word_manager:init_from_file(path.join(opt.data_dir, 'dictWord.txt'), 0, opt.max_vocab_size)
-- form_manager = SymbolsManager(true)
-- form_manager:init_from_file(path.join(opt.data_dir, 'dictFormWord.txt'), 0, opt.max_vocab_size)
-- word_raw_manager = SymbolsManager(true)
-- word_raw_manager:init_from_file(path.join(opt.data_dir, "dictWordRaw.txt"), 0, opt.max_vocab_size)
-- word_target_manager = SymbolsManager(true)
-- word_target_manager:init_from_file(path.join(opt.data_dir, "dictWordLabel.txt"), 0, opt.max_vocab_size)

--- ----------------------------------------------------------------------------
-- add all data to split to 5 fold 
--- ----------------------------------------------------------------------------
-- local dataNameTrain = opt.dataNameTrain..".t7"
-- local dataTrain = torch.load(dataNameTrain)
-- local dataNameTest = opt.dataNameTest..".t7"
-- local dataTest = torch.load(dataNameTest)
-- for k, v in pairs(dataTest) do 
--     table.insert( dataTrain, v)
-- end 
local data = torch.load(opt.dataName)
--print (data[1])
--print (data[10])
local idxFieldDistinct = 1
local distinct_data_set = distinct_data(data, idxFieldDistinct)
print(('saving data distinct: size = [%d], path = %s '):format( #distinct_data_set, opt.dataNameLogicform))
torch.save(opt.dataNameLogicform, distinct_data_set)

local idxPointer = torch.randperm(#data)
local rateTrain = 0.8
local idxFold = 1
for idxFold = 4, 4 do
    splitDataTrainTest(data, rateTrain, idxFold, idxPointer)
end
