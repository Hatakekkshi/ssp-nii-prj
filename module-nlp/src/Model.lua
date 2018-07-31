require 'nn'
require 'rnn'
require 'ModelSemantic'
require 'ModelCombine'

--- ----------------------------------------------------------------------------
-- require init package = global var table with path to RnnSemanticParser.model
--- ----------------------------------------------------------------------------
--- @class ModelGeneral
ModelGeneral = torch.class("RnnSemanticParser.model")

---@return ModelGeneral
function ModelGeneral:new (lengDict, lengWordVector, lengLabel, nNumLayerLSTM, mtRateLabel, mtRateLabelCombine, dropoutRate,mtRateLabelCombineDistinct)
    return RnnSemanticParser.model(lengDict, lengWordVector, lengLabel, nNumLayerLSTM,
    mtRateLabel, mtRateLabelCombine, dropoutRate, mtRateLabelCombineDistinct)
end

--- ----------------------------------------------------------------------------
-- constructor
--- ----------------------------------------------------------------------------
function ModelGeneral:__init(lengDict, lengWordVector, lengLabel, nNumLayerLSTM,
    mtRateLabel, mtRateLabelCombine, dropoutRate,mtRateLabelCombineDistinct)

    --- module combine recombine data, to perform accuracy module semantic
    -- example : where is VietNam => where is c0 // VietNam => c0
    --- @type Seq2SeqCombine
    self.modelCombine = Seq2SeqCombine(lengDict, lengWordVector,
       lengDict, mtRateLabelCombine)

    --- module semantic, parse languge to logic form
    -- example: where is c0	=> lambda $0 e ( loc:t c0 $0 )
    --- @type BiSeq2Seq
    self.modelSemantic = BiSeq2Seq(lengDict, lengWordVector,
        lengLabel, nNumLayerLSTM, mtRateLabel, dropoutRate, mtRateLabelCombineDistinct)

    ---
    self.models = {self.modelCombine, self.modelSemantic}
end

--- ----------------------------------------------------------------------------
-- todo set flag training
--- ----------------------------------------------------------------------------
function ModelGeneral:training()
    for _, submodel in pairs(self.models) do
        submodel:training()
    end
    return self
end

--- ----------------------------------------------------------------------------
-- todo set flag testing
--- ----------------------------------------------------------------------------
function ModelGeneral:evaluate()
    for _, submodel in pairs(self.models) do
        submodel:evaluate()
    end
    return self
end

--- ----------------------------------------------------------------------------
-- todo forget forward
--- ----------------------------------------------------------------------------
function ModelGeneral:forget()
    for _, submodel in pairs(self.models) do
        submodel:forget()
    end
    return self
end

--- ----------------------------------------------------------------------------
function ModelGeneral:getParameters()
    local container =  nn.Container()
    for _, submodel in pairs(self.models) do
        container:add(submodel:getParameters())
    end
    return container:getParameters()
end

--- ----------------------------------------------------------------------------
-- use gpu
--- ----------------------------------------------------------------------------
function ModelGeneral:cuda()
    print ("- enable cuda")
    self.isUseCuda = true
    for k, subModel in pairs(self.models) do
        subModel:cuda()
    end
    return self
end

--- ----------------------------------------------------------------------------
-- use cpu
--- ----------------------------------------------------------------------------
function ModelGeneral:unCuda()
    self.isUseCuda = false
    for k, subModel in pairs(self.models) do
        subModel:unCuda()
    end
    return self
end

--- ----------------------------------------------------------------------------
-- clear state model to reduce size before save 
--- ----------------------------------------------------------------------------
function ModelGeneral:clearState()
    self.isUseCuda = false
    for k, subModel in pairs(self.models) do
        subModel:clearState()
    end
    return self
end


--- ----------------------------------------------------------------------------
-- testing
--- ----------------------------------------------------------------------------
function ModelGeneral:test(test_loader, word_manager, opt)
end


--- ----------------------------------------------------------------------------
-- evaluate
-- -----------------------------------------------------------------------------
function ModelGeneral:eval(sSentenceIn, opt)
    local MAX_OUTPUT_SIZE = 100
    local wordsSentenceIn = sSentenceIn:split(' ')
    local w_list = word_manager:get_symbol_idx_for_list(wordsSentenceIn)
    local mReverse = nn.ReverseTable()
    local wordIdDecodeCurrent = word_manager:get_symbol_idx("<S>")

    print ("---------------")
    print ("Input: " .. sSentenceIn)
    print ("Dictionary convert: " .. word_manager_convert_to_string(w_list))

    -- reverse input 
    w_list =  mReverse(w_list)
    local tbl_dec_input = {}

    -- add token start + end
    table.insert(wordsSentenceIn, "<PADDING>")
    table.insert(w_list, 1, word_manager:get_symbol_idx("<S>"))
    table.insert(w_list, 1, 0)
    table.insert(w_list, word_manager:get_symbol_idx("<E>"))

    -- convert input to matrix
    local matrixInput = torch.Tensor(w_list)
    matrixInput = nn.View(#w_list, 1)(matrixInput)

    ---
    -- evaluate 
    --
    self:evaluate()
    local enc_output_combine = self.modelSemantic:eval_batch_recombine(matrixInput)
    local enc_output, tblMap = self.modelCombine:merge(enc_output_combine:clone(),
    matrixInput, word_manager:get_symbol_idx("<none>"), opt.compressDataCombine)

    -- loop to decode when found <E>
    local idWordEnd = word_manager:get_symbol_idx("<E>")
    for i =1, MAX_OUTPUT_SIZE do
        table.insert(tbl_dec_input, wordIdDecodeCurrent)
        local dec_input = nn.View(#tbl_dec_input, 1)( torch.Tensor(tbl_dec_input))
        local dec_output

        if i == 1 then
            dec_output = self.modelSemantic:eval_batch_with_oldEncBatch(matrixInput, dec_input)
        else
            dec_output = self.modelSemantic:eval_batch_with_oldEncBatch(nil, dec_input)
        end

        wordIdDecodeCurrent = dec_output[dec_output:size(1)][1]
        if (wordIdDecodeCurrent == idWordEnd ) then
            break
        end
    end

    -- print result
    print ("Maps word recombine:")
    local mapNer = {}
    local wordsReverseSentenceIn = mReverse(wordsSentenceIn)
    for k, v in pairs (tblMap[1]) do
        local sMap = "\t" .. word_manager:get_idx_symbol(k) .. " => "
        local sVal = ""
        for _, idxW in pairs(v) do
            sVal = sVal .. wordsReverseSentenceIn[idxW - 1] .. " "
        end
        sMap = sMap .. sVal
        mapNer[word_manager:get_idx_symbol(k)] = sVal:sub(0, #sVal - 1)
        print(sMap)
    end
    print ("Form logic:")
    table.remove(tbl_dec_input, 1)
    local sRet = convert_to_string(tbl_dec_input)
    print ("---------------")

    return sRet, mapNer
end

--- ----------------------------------------------------------------------------
-- training
--- ----------------------------------------------------------------------------
function ModelGeneral:trainMixture(train_loader, test_loader, word_manager, opt)

end


--- ----------------------------------------------------------------------------
-- training
--- ----------------------------------------------------------------------------
function ModelGeneral:trainConcat(train_loader, test_loader, word_manager, opt, train_loader_distinct)

end

--- ----------------------------------------------------------------------------
-- todo training combine
--- ----------------------------------------------------------------------------
function ModelGeneral:trainCombine(train_loader, test_loader, word_manager, opt)


end

--- ----------------------------------------------------------------------------
-- todo training semantic
--- ----------------------------------------------------------------------------
function ModelGeneral:trainSemantic(train_loader, test_loader, word_manager, opt)
    ---
    --
    -- todo train model semantic
    --
    print("- training semantic ...")
    local optimState = {
        learningRate = opt.lr,
        momentum = opt.momentum
    }
    local learningRate = optimState.learningRate
    local maxEpochs = opt.maxEpochSemantic
    local model = self.modelSemantic

    if (opt.saturateEpoch == 0) then opt.saturateEpoch = maxEpochs end
    local decayFactor = (opt.minLR - learningRate) / opt.saturateEpoch

    model:training()
    for idxEpochs = 1, opt.maxEpochSemantic do
        local idxsMinibatch = torch.randperm(train_loader.num_batch)
        for iterator = 1, train_loader.num_batch do

            -- params
            collectgarbage()
            params, gradParams = model:getParameters()
            local enc_batch, decIn, decOut, enc_raw, enc_target, encTargetRecombine = 
                train_loader:get_matrix(idxsMinibatch[iterator])

            -- local function we give to optim
            -- it takes current weights as input, and outputs the loss
            -- and the gradient of the loss with respect to the weights
            -- gradParams is calculated implicitly by calling 'backward',
            -- because the model's weight and bias gradient tensors
            -- are simply views onto gradParams
            function feval(x)
                if x ~= params then
                    params:copy(x)
                end
                gradParams:zero()

                local enc_input = enc_target
                if (opt.modelType == "parallel" 
                or opt.modelType == "mixture"
                or opt.modelType == "multitask"
                ) then
                    enc_input = enc_raw -- enc_batch --  ----
                elseif (opt.modelType == "concat") then
                    enc_input = self.modelCombine:merge(enc_target, enc_raw,
                        word_manager:get_symbol_idx("<none>"), opt.compressDataCombine)
                end
                local loss =  model:forward(enc_input, decIn, decOut,encTargetRecombine)
                local dloss_doutputs = model:backward(enc_input, decIn, decOut,encTargetRecombine)
                model:updateParameters(learningRate)
                gradParams:clamp(-5, 5)
                
                -- show progress bar to estimate time wait
                loss = loss / enc_input:size(2)
                if (iterator % math.ceil(train_loader.num_batch / 2) == 0) then
                    print(idxEpochs .. " - " .. iterator .. " - err : " .. loss)
                    xlua.progress(iterator + idxEpochs * train_loader.num_batch,
                        maxEpochs * train_loader.num_batch)
                end

                return loss, gradParams
            end

            optim.rmsprop(feval, params, optimState)
            model:forget()
            collectgarbage()

            end

        -- show try test
        if (idxEpochs % 10 == 0 and idxEpochs >= 10 --[[ maxEpochs * 0.4 --]] ) then
            -- try test and print result
            self:test(test_loader, word_manager, opt)
            print (("Test after epoch : %d"):format(idxEpochs))

            -- ennable state training for model
            model:training()

            -- save model for each test
            torch.save((opt.model_dir .. 'model_cpu_%d.t7'):format(opt.idxFold), self:unCuda())
            if (opt.isUsedCuda) then
                self:cuda()
            end
        end

        -- update learning rate
        learningRate = learningRate + decayFactor
        learningRate = math.max(opt.minLR, learningRate)
        optimState.learningRate = learningRate
    end

end

--- ----------------------------------------------------------------------------
-- overide to string 
function ModelGeneral:__tostring__()
  return ("- Model {" ..
          "\n\tself.modelCombine = %s" ..
          "\n\tself.modelSemantic = %s" ..
          "\n}"):format(self.modelCombine, self.modelSemantic)
end
