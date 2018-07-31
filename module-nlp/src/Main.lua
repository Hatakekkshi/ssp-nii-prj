require 'torch'
require("Package")

print("- Load Main")
--- ----------------------------------------------------------------------------
-- todo init parameter and some config var
-- -----------------------------------------------------------------------------
if opt == nil then
    opt = ParamsParser()

    opt.model_dir = "../data_bot_nii/"
    opt.data_dir = "../data_bot_nii/"
    opt.maxEpochSemantic = 30
    opt.maxEpochCombine = 30
    opt.lengWordVector = 30
    opt.nNumLayerLstmIntermediate = 1
    opt.modelType = "multitask"
    opt.batch_size = 10
    opt.isUsedCuda = false
    opt.lr = 0.002
    opt.model_dir = "../data_bot_nii/"
end

local lengDict = opt.lengDict --500
local lengLabel = opt.lengLabel
local lengWordVector = opt.lengWordVector
local nNumLayerLstmIntermediate = opt.nNumLayerLstmIntermediate
local dropoutRate = opt.dropoutRate

if (opt.isUsedCuda) then
    require 'cutorch'
    require 'cunn'
end

--------------------------------------------------------------------------------
-- todo initialize the vocabulary manager to display text
-- -----------------------------------------------------------------------------
word_manager, form_manager, word_raw_manager, word_target_manager = table.unpack(torch.load(opt.data_dir .. 'map.t7'))
lengDict = word_manager.vocab_size
lengLabel = form_manager.vocab_size
print(("- [Dict] lengDict = %d, lengLabel = %d"):format(lengDict,lengLabel))

local train_loader, test_loader, train_loader_distinct
local dataTest
local mtRateLabel, mtRateLabelDistinct
local mtRateLabelCombine, mtRateLabelCombineDistinct
if opt.training or opt.testing then
    --- ----------------------------------------------------------------------------
    -- todo load data set th Main.lua --data_dir ../argument_raw_geo_prac/  --testing --training --nameLog concat_geo_rerun_hd300 --idxFold 4 --model_dir ../model_concat_geo_rerun/ --saturateEpoch 200 --maxEpochSemantic 200 --batch_size 10 --lengWordVector 300
    -- -----------------------------------------------------------------------------
    -- data test
    test_loader = MinibatchLoader.create(opt, opt.nameDataTest, word_manager, form_manager)
    test_loader:transformer_matrix(opt.isUsedCuda) -- convert data table 2 matrix
    dataTest = torch.load(opt.data_dir .. opt.nameDataTest)
    print(("- [Test] num datatest = %d"):format(#dataTest))

    --- data train
    train_loader = MinibatchLoader.create(opt, opt.nameDataTrain, word_manager, form_manager)
    mtRateLabel = train_loader:gen_rate_train_label(train_loader.dec_batch_list, lengLabel)
    mtRateLabelCombine = train_loader:gen_rate_train_label(train_loader.enc_target_recombine_list, lengDict)
    train_loader:transformer_matrix(opt.isUsedCuda) -- convert data table 2 matrix
    print("- [Train] Num batch   = " .. train_loader.num_batch)
    print("- [Train] Num sample  = " .. train_loader.num_sample)

    --- data logic distinct
    train_loader_distinct = MinibatchLoader.create(opt, opt.dataNameLogicform, word_manager, form_manager)
    mtRateLabelDistinct = train_loader_distinct:gen_rate_train_label(train_loader_distinct.dec_batch_list, lengLabel)
    mtRateLabelCombineDistinct = train_loader_distinct:gen_rate_train_label(train_loader_distinct.enc_target_recombine_list, lengDict)
    train_loader_distinct:transformer_matrix(opt.isUsedCuda) -- convert data table 2 matrix
    print("- [Train distinct] Num batch   = " .. train_loader_distinct.num_batch)
    print("- [Train distinct] Num sample  = " .. train_loader_distinct.num_sample)

    --- ----------------------------------------------------------------------------
    -- todo print sample to check data
    -- -----------------------------------------------------------------------------
    function printData(data)
        print ("- [print data test] All data test")
        for idx = 1, #data do
            print(("%s"):format(word_manager_convert_to_string(data[idx][3])))
            print(("- %s"):format(convert_to_string(data[idx][2])))
        end
        print ("---")
    end
    --printData(dataTest)

    local idx = 1
    print(dataTest[idx])
    print(("sample %d: %s"):format(idx, word_manager_convert_to_string(dataTest[idx][1])))
    print(("dist   %d: %s"):format(idx, convert_to_string(dataTest[idx][2])))

    local mt =  {test_loader:get_matrix(idx)}
    for k, v in pairs(mt) do
        mt[k] = v[{{},1}]
    end
    print (mt)
    print (word_manager_convert_to_string(mt[4]:totable()))
    print (word_manager_convert_to_string(mt[6]:totable()))
    print (word_manager_convert_to_string(mt[5]:totable()))
    print (word_manager_convert_to_string(mt[1]:totable()))
    print (convert_to_string(mt[2]:totable()))
    print (convert_to_string(mt[3]:totable()))
end

--- ----------------------------------------------------------------------------
-- todo init model
-- -----------------------------------------------------------------------------
model = RnnSemanticParser.model(lengDict, lengWordVector, lengLabel,
    nNumLayerLstmIntermediate, mtRateLabel, mtRateLabelCombine, dropoutRate,mtRateLabelCombineDistinct)
if (opt.training == false) then
    model = torch.load(opt.model_dir .."model_cpu_"..opt.idxFold..".t7")
end
--print (model)
if (opt.isUsedCuda) then
    model:cuda()
end

--- ----------------------------------------------------------------------------
-- todo traning
--- ----------------------------------------------------------------------------
if (opt.training == true) then
    model:trainSemantic(train_loader, test_loader, word_manager, opt)
end

--- ----------------------------------------------------------------------------
-- todo test on data test 
--- ----------------------------------------------------------------------------
if (opt.testing == true) then

    local tmpState = opt.logPrintNegative
    opt.logPrintNegative = true
    model:test(train_loader, word_manager, opt)
    opt.logPrintNegative = tmpState
end

--- ----------------------------------------------------------------------------
-- todo testing
--- ----------------------------------------------------------------------------
if (opt.eval == true) then

    local testinput = {
        --"tuổi là an",
    }
    for k, v in pairs(testinput) do
        print (model:eval(v, opt))
    end

    local sLine = ""
    repeat
        print ("====================")
        print ("Type your input: ")
        sLine = io.read("*l") 
        print ("Output: ")
        if (sLine ~= "exit") then
            local formLogic, mapNer = model:eval(sLine, opt)
--            formLogic = prologUnzip(formLogic)
            print ("Form logic: "..formLogic)
            print ("Map NER: ", mapNer)
        end
    until (sLine == "exit" or sLine == "")
end
