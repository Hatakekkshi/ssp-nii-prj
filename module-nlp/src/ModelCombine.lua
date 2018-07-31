require 'nn'
require 'rnn'

---@class Seq2SeqCombine
--- ----------------------------------------------------------------------------
-- require init package = global var table with path to RnnSemanticParser.model
--- ----------------------------------------------------------------------------
Seq2SeqCombine = nil
torch.class("Seq2SeqCombine")
torch.class("RnnSemanticParser.model.combine", "Seq2SeqCombine")


--- ----------------------------------------------------------------------------
-- constructor
--- ----------------------------------------------------------------------------
function Seq2SeqCombine:__init(lengDict, lengWordVector, lengLabel, mtRateLabel)
end

--- ----------------------------------------------------------------------------
-- [[ Forward ]]--
--- ----------------------------------------------------------------------------
function Seq2SeqCombine:forward(inputEncode, targetEncode)
end

--- ----------------------------------------------------------------------------
-- [[ Backward ]]--
function Seq2SeqCombine:backward(inputEncode, targetEncode)
end

--- ----------------------------------------------------------------------------
function Seq2SeqCombine:getParameters()
end

--- ----------------------------------------------------------------------------
function Seq2SeqCombine:forget()
end

--- ----------------------------------------------------------------------------
function Seq2SeqCombine:training()
end
--- ----------------------------------------------------------------------------
function Seq2SeqCombine:evaluate()
end

--- ----------------------------------------------------------------------------
function Seq2SeqCombine:cuda()
    self.isUseCuda = true
    return self
end

--- ----------------------------------------------------------------------------
function Seq2SeqCombine:updateParameters(lr)
end
--- ----------------------------------------------------------------------------
function Seq2SeqCombine:unCuda()
    self.isUseCuda = false
    return self
end

--- ----------------------------------------------------------------------------
function Seq2SeqCombine:clearState()
end

--- ----------------------------------------------------------------------------
-- eval accuracy
function Seq2SeqCombine:eval(enc_w_list)
end


--- ----------------------------------------------------------------------------
-- eval accuracy
function Seq2SeqCombine:eval_batch(input_batch)

end

--- ----------------------------------------------------------------------------
-- eval accuracy
function Seq2SeqCombine:merge(result, input_batch, noneId, compressDataCombine)
    local tblMap = {}
    for j = 1, result:size(2) do
        local mapsInSentence = {}
        for i = result:size(1), 2, -1 do
            if (result[i][j] ~= 0 and result[i][j] ~=  noneId) then
                if (mapsInSentence[result[i][j]] == nil) then
                    mapsInSentence[result[i][j]] = {}
                end
                table.insert (mapsInSentence[result[i][j]], i)
            end
        end
        table.insert(tblMap, mapsInSentence)
    end

    local mergeResult = result:double():map(input_batch:double(), function (y_val, x_val)
        if (y_val == noneId or y_val == 0) then
            return x_val
        else
            return y_val
        end

    end)

    if compressDataCombine then
        for i = mergeResult:size(1), 2, -1 do
            for j = 1, mergeResult:size(2) do
                if (mergeResult[i][j] ~= 0 and mergeResult[i][j] == mergeResult[i-1][j]) then
                    mergeResult[{{2,i},{j}}] = mergeResult[{{1,i-1},{j}}]:clone()
                    mergeResult[1][j] = 0
                end
            end
        end
    end

    if (self.isUseCuda) then return mergeResult:cuda(), tblMap
    else return mergeResult, tblMap end
end


--- ----------------------------------------------------------------------------
-- overide to string 
function Seq2SeqCombine:__tostring__()
  return ("BRNN-lstm(%d)"):format(self.lengWordVector)
end