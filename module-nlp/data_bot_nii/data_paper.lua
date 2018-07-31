include "../src/SymbolsManager.lua"
include "../src/utils.lua"
utf8 = require ('lua-utf8')
require ('pl.stringx').import()
path = require ('pl.path')

-------------------------------------------------------------------------------
-- processing data vocabulary 
-- -----------------------------------------------------------------------------
function process_vocab_data(opt)

    local word_manager = SymbolsManager(true)
    word_manager:init_from_file(opt.data_dir .. opt.dictQuery, opt.min_freq, opt.max_vocab_size)
    local form_manager = SymbolsManager(true)
    form_manager:init_from_file(opt.data_dir .. opt.dictForm, opt.min_freq, opt.max_vocab_size)

    -- special symbol to flag normal word
    word_manager:add_symbol("<none>")
    word_manager:add_symbol("<entity0>")
    word_manager:add_symbol("<entity1>")
    word_manager:add_symbol("<entity2>")
    word_manager:add_symbol("<entity3>")

    form_manager:add_symbol("<entity0>")
    form_manager:add_symbol("<entity1>")
    form_manager:add_symbol("<entity2>")
    form_manager:add_symbol("<entity3>")

    -- save output preprocessed files
    local out_mapfile = opt.data_dir .. 'map.t7'
    print('- saving ' .. out_mapfile)
    torch.save(out_mapfile, { word_manager, form_manager })

    collectgarbage()

    return { word_manager, form_manager }
end

-------------------------------------------------------------------------------
-- processing data train 
-- -----------------------------------------------------------------------------
function process_train_data(nameFileRaw, vocab_table)
    require('pl.stringx').import()
    require 'pl.seq'

    -- load vocabulary 
    if (torch.type(vocab_table) == "string") then
        word_manager, form_manager = table.unpack(torch.load(vocab_table))
    else
        word_manager, form_manager = table.unpack(vocab_table)
    end

    -- load both data argument and data raw 
    print(('- loading text file [%s] ...'):format(nameFileRaw))
    local f_raw = torch.DiskFile(nameFileRaw, 'r', true)
    f_raw:clearError()
    local rawdata = f_raw:readString('*l')

    -- read each line data to generate 
    local data = {}
    while (not f_raw:hasError()) do
        local l_list = rawdata:strip():split('\t')
        if l_list[1] and l_list[2] then

            local w_list = word_manager:get_symbol_idx_for_list(l_list[1]:split(' '))
            local r_list = form_manager:get_symbol_idx_for_list(l_list[2]:split(' '))

            table.insert(data, { w_list, r_list })
        else
            print ("process_train_data - skip", rawdata)
        end

        -- read next line
        rawdata = f_raw:readString('*l')
    end
    f_raw:close()

    collectgarbage()
    return data
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function extractLine(line)
    local l_list = line:split('\t')
    if l_list[2] == nil then
        print ("extractLine - skip", line)
        return
    end
    --l_list[2] = string.gsub(l_list[2], '(%()([^ ])', '%1 %2')
    --l_list[2] = string.gsub(l_list[2], '(%()([^ ])', '%1 %2')
    --l_list[2] = string.gsub(l_list[2], '([^ ])(%))', '%1 %2')
    --l_list[2] = string.gsub(l_list[2], '([^ ])(%))', '%1 %2')
    local w_list = l_list[1]:split(' ')
    local f_list = l_list[2]:split(' ')
    return w_list, f_list
end


-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
function extractDict(sPathFile)
    local pairRets = {}
    local dictWordRaw, dictFormLogicRaw = {}, {}
    for line in io.lines(sPathFile) do
        -- get data is parsed form: parsed( text_to_get )
        local wordEncodeList, wordDecodeList = extractLine(line)
        if wordEncodeList and wordDecodeList then
            -- add to dictFormLogic
            for k1, word in pairs(wordDecodeList) do
                if dictFormLogicRaw[word] == nil then
                    dictFormLogicRaw[word] = 1
                else
                    dictFormLogicRaw[word] = dictFormLogicRaw[word] + 1
                end
            end
            --add to dictWordRaw
            for k1, word in pairs(wordEncodeList) do
                if dictWordRaw[word] == nil then
                    dictWordRaw[word] = 1
                else
                    dictWordRaw[word] = dictWordRaw[word] + 1
                end
            end
        end
    end
    return dictWordRaw, dictFormLogicRaw
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- gen or get token for entity in sentence 
-- ----------------------------------------------------------------------------
function genTokenEntity(idToken, word_manager, baseIdEntity) 
    local sTokenOld = word_manager:get_idx_symbol(idToken)
    local id = string.match (sTokenOld, "%d+")
    local newIdToken = idToken
    if id ~= nil then 
        local newSToken = ("<entity%d>"):format(baseIdEntity)
        newIdToken = word_manager:add_symbol(newSToken)
    else 
        -- print (sTokenOld)
    end 
    return newIdToken
end 

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- find idTokenOld and idTokenNew in decode dict 
-- ----------------------------------------------------------------------------
function findDecodeTokenEntity(idTokenOld, idTokenNew, word_manager, form_manager) 
    local sTokenOld = word_manager:get_idx_symbol(idTokenOld)
    local sTokenNew = word_manager:get_idx_symbol(idTokenNew)
    local idTokenDecodeOld = form_manager:get_symbol_idx(sTokenOld)
    local idTokenDecodeNew = form_manager:add_symbol(sTokenNew)
    return idTokenDecodeOld, idTokenDecodeNew
end


-------------------------------------------------------------------------------
-- processing data train
-- -----------------------------------------------------------------------------
function process_len_data(nameFileRaw)

    local leng ={}
    for line in io.lines(nameFileRaw) do
        --
        local lenLine = {}
        local len_list = line:split(' ')
        for k, v in pairs (len_list) do
            table.insert(lenLine, tonumber(v))
        end
        table.insert( leng, lenLine )
    end

    return leng
end


-------------------------------------------------------------------------------
-- label is character upper leng 1
-- -----------------------------------------------------------------------------
function checkIsLabel(label)
    return label:startswith("_")
end
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- parse data
function parseData(opt)

    local dictWordRaw, dictFormLogicRaw = nil, nil

    --- process data raw 
    if (opt.bUseDataRaw) then
        -- gen dict word 
        print(("- processing data raw: %s"):format(opt.trainDataRawName))
        local dictWordRawTrain, dictFormLogicRawTrain = extractDict(opt.trainDataRawName)

        -- process input word 
        if (dictWordRaw == nil) then
            -- assign value 
            dictWordRaw = dictWordRawTrain
        else
            -- add dict train 
            for k, v in pairs(dictWordRawTrain) do
                if (dictWordRaw[k] ~= nil) then
                    dictWordRaw[k] = dictWordRaw[k] + v
                else
                    dictWordRaw[k] = v
                end
            end
        end

        -- process output form logic 
        if (dictFormLogicRaw == nil) then
            -- assign value 
            dictFormLogicRaw = dictFormLogicRawTrain
        else
            -- add dict train 
            for k, v in pairs(dictFormLogicRawTrain) do
                if (dictFormLogicRaw[k] ~= nil) then
                    dictFormLogicRaw[k] = dictFormLogicRaw[k] + v
                else
                    dictFormLogicRaw[k] = v
                end
            end
        end
    end

    --- process data argument
    if (opt.bUseDataArgument) then
        -- gen dict word 
        print(("- processing data argument id: %s"):format( opt.trainDataName))
        local dictWordRawTrain, dictFormLogicRawTrain = extractDict(opt.trainDataName)

        -- process input word 
        if (dictWordRaw == nil) then
            -- assign value 
            dictWordRaw = dictWordRawTrain
        else
            -- add dict train 
            for k, v in pairs(dictWordRawTrain) do
                if (dictWordRaw[k] ~= nil) then
                    dictWordRaw[k] = dictWordRaw[k] + v
                else
                    dictWordRaw[k] = v
                end
            end
        end

        -- process output form logic 
        if (dictFormLogicRaw == nil) then
            -- assign value 
            dictFormLogicRaw = dictFormLogicRawTrain
        else
            -- add dict train 
            for k, v in pairs(dictFormLogicRawTrain) do
                if (dictFormLogicRaw[k] ~= nil) then
                    dictFormLogicRaw[k] = dictFormLogicRaw[k] + v
                else
                    dictFormLogicRaw[k] = v
                end
            end
        end
    end

    if (path.exists(opt.trainDataSpamName) ~= nil) then
        local dictWordSpamTrain, dictFormLogicSpamTrain = extractDict(opt.trainDataSpamName)

        -- add dict train
        for k, v in pairs(dictFormLogicSpamTrain) do
            if (dictFormLogicRaw[k] ~= nil) then
                dictFormLogicRaw[k] = dictFormLogicRaw[k] + v
            else
                dictFormLogicRaw[k] = v
            end
        end
    end

    -- write file dict 
    local pathDictWord, pathDictForm = opt.data_dir .. opt.dictQuery, opt.data_dir .. opt.dictForm
    print(string.format("- write dictionary: %s, %s ", pathDictWord, pathDictForm))
    local fileDictWord = io.open(pathDictWord, "w")
    local fileDictForm = io.open(pathDictForm, "w")
    for k, v in pairs(dictWordRaw) do
        fileDictWord:write(k .. "\t" .. v .. "\n")
    end

    fileDictForm:write("(\t" .. dictFormLogicRaw["("] .. "\n")
    fileDictForm:write(")\t" .. dictFormLogicRaw[")"] .. "\n")
    for k, v in pairs(dictFormLogicRaw) do
        if k ~= "(" and k ~= ")" then
            fileDictForm:write(k .. "\t" .. v .. "\n")
        end
    end

    fileDictWord:close()
    fileDictForm:close()

    -- preprocess vocabulary dict
    local vocab_table = process_vocab_data(opt)

    -- preprocess data train and test 
    local data = {}
    if (opt.bUseDataRaw and opt.bUseDataArgument) then
        -- use data argument + raw to combine 
        local train_raw = process_train_data(opt.data_dir .. opt.trainDataRawName, vocab_table)
        local train_arg = process_train_data(opt.data_dir .. opt.trainDataName, vocab_table)
        local train_spam = process_train_data(opt.data_dir .. opt.trainDataSpamName, vocab_table)
        local train_len = process_len_data(opt.data_dir .. opt.trainLenName)
        local train_len_spam = process_len_data(opt.data_dir .. opt.trainLenSpamName)

        -- add spam to data arg and raw
        for k, v in pairs(train_spam) do
            table.insert(train_raw, v)
            table.insert(train_arg, v)
        end
        for k, v in pairs(train_len_spam) do
            table.insert(train_len, v)
        end

        -- validate data
        assert(#train_raw == #train_arg, "[Err] #train_raw == # train_arg")
        assert(#train_raw == #train_len, "[Err] #train_raw == # train_arg")

        -- rebase data set 
        -- append data train arg + raw 
        data = train_arg
        for k, v in pairs(train_raw) do
            table.insert(data[k], v[1])
        end

        -- save lst sentence err
        local lstSentenceErr = {}

        -- combine data
        for k0, samples in pairs(data) do
            local input_arg = samples[1]
            local input_raw = samples[3]

            local input_raw_save = cloneTable(input_raw)


            table.insert(samples, 3, input_raw_save)

            -- local input_arg = {77,2,99,4,5,88}
            -- local input_raw = {21,22,23,2,9,10,12,4,5,13,14,15}
            local idxRaw = 1
            local idxArg = 1
            local idxLabelInSentence = 1
            local bSentenceErr = false
            for idx = 1 , #input_raw do
                -- neu wordid hien tai trung vs word raw thi nhay wordidArg sang tiep theo cho phu hop vs wordidRaw
                -- neu wordid hien tai khac vs word raw thi next wordidArg cho den khi het len trong info len, sau
                --      do chuyen tiep word id tiep theo
                if input_arg[idxArg] == input_raw[idxRaw] then
                    -- go to next argument wordid
                    idxArg = idxArg + 1
                    idxRaw = idxRaw + 1
                else
                    local idWordArg = input_arg[idxArg]
                    local countWordLabel = 0
                    local sLabel = word_manager:get_idx_symbol(idWordArg)
                    local sTermLimitLengEntity = ""
                    if checkIsLabel(sLabel) then
                        -- them tung phan tu tu tiep theo vao trong entity hien tai
                        -- cho den khi entity hien tai co size = trong file len.txt info
                        for i = idxRaw, #input_raw do
                            if sTermLimitLengEntity == "" then
                                sTermLimitLengEntity = word_manager:get_idx_symbol(input_raw[i])
                            else
                                sTermLimitLengEntity = sTermLimitLengEntity .. " " .. word_manager:get_idx_symbol(input_raw[i])
                            end
                            -- cai dat gia tri nhan id
                            input_raw[i] = input_arg[idxArg]
                            countWordLabel = countWordLabel + 1

                            local lenInfoEntity = train_len[k0][idxLabelInSentence]
                            local curLenEntity = utf8.len(sTermLimitLengEntity)
                            if curLenEntity == lenInfoEntity then
                                sTermLimitLengEntity = ""
                                break
                            elseif curLenEntity > lenInfoEntity then
                                print("err: kich thuoc entity > mo ta trong len.txt ")
                                print (("\tline: %d, entity: %s, len: %d, lenInfo: %d"):format(k0, sTermLimitLengEntity,
                                    curLenEntity, train_len[k0][idxLabelInSentence] ))
                                bSentenceErr = true
                                break
                            end
                        end
                    else
                        print ("check label false: ", sLabel, word_manager:get_idx_symbol(input_raw[idxRaw]))
                        bSentenceErr = true
                        break
                    end
                    idxRaw = idxRaw + countWordLabel
                    idxArg = idxArg + 1
                    idxLabelInSentence = idxLabelInSentence + 1
                end
                if idxRaw > #input_raw then break end
            end

            if bSentenceErr then
                print("------------------")
                print("[check err]")
                print (word_manager_convert_to_string(input_arg))
                print (word_manager_convert_to_string(input_raw_save))
                print (word_manager_convert_to_string(input_raw))
                print (convert_to_string(samples[2]))
                print("------------------")
                table.insert(lstSentenceErr, k0)
            end
        end
        for i = #lstSentenceErr, 1, -1 do
            table.remove(data, lstSentenceErr[i])
        end
    end
    -- save data
    print('- saving data[size=' .. #data .. "]: ".. opt.dataMatrixName)
    torch.save(opt.dataMatrixName, data)

    -- return dict
    return vocab_table
end

-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- parse argument 
local cmd = torch.CmdLine()
cmd:option('-data_dir', '', 'data directory')
-- cmd:option('-train', 'train.raw.txt.txt', 'train data path')
-- cmd:option('-test', 'test.raw.txt', 'test data path')

cmd:option('-bUseDataRaw', true, 'parse all data raw')
cmd:option('-bUseDataArgument', true, 'parse all data argument id')
cmd:option('-trainDataName', 'data.ner.txt', 'train data path')
cmd:option('-trainDataRawName', 'data.raw.txt', 'train data path')
cmd:option('-trainDataSpamName', 'data.spam.txt', 'train data path')
cmd:option('-trainLenName', 'len.txt', 'test data path')
cmd:option('-trainLenSpamName', 'len.spam.txt', 'test data path')
cmd:option('-dataMatrixName', 'data.t7', 'test data path')

cmd:option('-dev', 'dev', 'dev data path')
cmd:option('-dictQuery', 'vocab.q.raw.txt', 'test data path')
cmd:option('-dictForm', 'vocab.f.raw.txt', 'test data path')
cmd:option('-min_freq', 0, 'minimum word frequency')
cmd:option('-max_vocab_size', 15000, 'maximum vocabulary size')
cmd:text()
opt = cmd:parse(arg)


-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- ----------------------------------------------------------------------------
-- parse data 
parseData(opt)