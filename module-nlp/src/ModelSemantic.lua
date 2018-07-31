require 'nn'
require 'rnn'

---@class BiSeq2Seq
---
--- ----------------------------------------------------------------------------
-- require init package = global var table with path to RnnSemanticParser.model

BiSeq2Seq = nil
torch.class("BiSeq2Seq")
torch.class("RnnSemanticParser.model.semantic2", "BiSeq2Seq")

--- ----------------------------------------------------------------------------
-- constructor
function BiSeq2Seq:__init(lengDict, lengWordVector, lengLabel, nNumLayerLSTM, 
    mtRateLabel, dropoutRate, mtRateLabelCombine)


    ---
    -- init encode layer 
    -- batch x seqEncodeSize[word-encode id] => batch x SeqLengEncode x HiddenSize [double value]
    local lookupTableE  = nn.LookupTableMaskZero(lengDict, lengWordVector)
    if (nNumLayerLSTM < 1) then
            nNumLayerLSTM = 1
    end
    self.nNumLayerLSTM = nNumLayerLSTM
    self.lstmEncodeForwardIntemediateLayer    = nn.Sequential()
    self.lstmEncodeBackwardIntemediateLayer   = nn.Sequential()
    
    self.lstmEncodeModules = {}
    for idxLayer = 1, nNumLayerLSTM do
        local seqLstmF = nn.SeqLSTM(lengWordVector, lengWordVector)
        local seqLstmB = nn.SeqLSTM(lengWordVector, lengWordVector)
        seqLstmF:maskZero(1)
        seqLstmB:maskZero(1)
        local dropoutF = nn.Sequencer(nn.MaskZero(nn.Dropout(dropoutRate),1))
        local dropoutB = nn.Sequencer(nn.MaskZero(nn.Dropout(dropoutRate),1))
        self.lstmEncodeForwardIntemediateLayer
            :add(dropoutF)
            :add(seqLstmF)
        self.lstmEncodeBackwardIntemediateLayer = nn.Sequential()
                                                    :add(nn.SeqReverseSequence(1)) -- reverse
                                                    :add(dropoutB)
                                                    :add(seqLstmB)
                                                    :add(nn.SeqReverseSequence(1)) -- unreverse
        self.lstmEncodeModules[idxLayer] = {}
        self.lstmEncodeModules[idxLayer][1] = seqLstmF
        self.lstmEncodeModules[idxLayer][2] = seqLstmB
    end 

    local lstmLayer = nn.ConcatTable()
    :add(self.lstmEncodeForwardIntemediateLayer)
    :add(self.lstmEncodeBackwardIntemediateLayer)

    self.encoder        = nn.Sequential()
                                :add(lookupTableE)
                                :add(lstmLayer)
                                :add(nn.JoinTable(2,2))
                                :add(nn.Sequencer(nn.MaskZero(nn.Linear(2*lengWordVector, 2*lengWordVector):noBias(),1)))

    self.recombiner     = nn.Sequential()
                                :add(nn.Sequencer(nn.MaskZero(nn.Linear(2*lengWordVector, lengDict), 1)))
                                :add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

    --- 
    -- save config
    self.lengWordVector = lengWordVector

    ---
    -- init decode layer 
    -- batch x seqEncodeSize[word-decode id] => batch x SeqLengDecode x HiddenSize [double value]
    local   lookupTableD  = nn.LookupTableMaskZero(lengDict, 2*lengWordVector)
    self.lstmDecodeIntemediateLayer = nn.Sequential()
    self.lstmDecodeModules = {}
    for idxLayer = 1, nNumLayerLSTM do
        local Rnn = nn.Sequencer(nn.FastLSTM(2*lengWordVector, 2*lengWordVector):maskZero(1))
        self.lstmDecodeIntemediateLayer
                    :add(nn.Sequencer(nn.MaskZero(nn.Dropout(dropoutRate),1)))
                    :add(Rnn)
        self.lstmDecodeModules[idxLayer] = {}
        self.lstmDecodeModules[idxLayer][1] = Rnn:findModules("nn.FastLSTM")[1]
    end
    self.decoder = nn.Sequential()  :add(lookupTableD)
                                    :add (self.lstmDecodeIntemediateLayer)

    ---
    -- init concat layer of encode 
    self.connectEncodeDecode = nn.Sequential()  :add(nn.JoinTable(2))



    ---
    -- init attention layer 
    -- {input1, input2} => output
    --      input1 =  out of encoder{1,SeqLengEncode} : matrix [batchxSeqLengEnxHiddenSize]
    --      input2 =  out of decoder{1,SeqLengEncode} : matrix [batchxSeqLengDexHiddenSize]
    local attention = nn.Sequential()
                              :add(nn.MM(false, true))
                              :add(nn.SplitTable(3))
                              :add(nn.Sequencer(nn.MaskZero(nn.SoftMax(), 1)))
                              :add(nn.Sequencer(nn.View(-1, 1):setNumInputDims(1)))
                              :add(nn.JoinTable(2,2))

    local encodeAttention = nn.ConcatTable()
                                :add(nn.Sequential()
                                        :add(nn.ConcatTable()
                                                  :add(nn.SelectTable(1)) -- to get encode state output
                                                  :add(attention))        -- calculate s_t_k
                                        :add(nn.MM(true, false)))
                                :add(nn.Sequential()
                                        :add(nn.SelectTable(2))     -- to get decode state output
                                        :add(nn.Transpose({2,3})))  -- transpose dims 2 vs 3
    
    self.attention = nn.Sequential()
            :add(encodeAttention)
            :add(nn.JoinTable(2))
            :add(nn.SplitTable(3))
            :add(nn.Sequencer(nn.MaskZero(nn.Linear(4*lengWordVector, 2*lengWordVector), 1)))
            :add(nn.Sequencer(nn.MaskZero(nn.Tanh(), 1)))
            :add(nn.Sequencer(nn.MaskZero(nn.Dropout(dropoutRate),1)))
            :add(nn.Sequencer(nn.MaskZero(nn.Linear(2*lengWordVector, lengLabel),1)))
            :add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))

    ---
    -- init criterion
    self.criterion = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(mtRateLabel),1))
         
    self.criterionCombiner = nn.SequencerCriterion(
        nn.MaskZeroCriterion(nn.ClassNLLCriterion(mtRateLabelCombine),1))

    self.criterionSum  = nn.ParallelCriterion():add(self.criterion, 0.5):add(self.criterionCombiner)

    ---
    -- init transpose 
    self.parseInputAttention = nn.ParallelTable() 
                                    :add(nn.Transpose({1,2}))
                                    :add(nn.Transpose({1,2}))

end

--- ----------------------------------------------------------------------------
-- [[ Forward coupling: Copy encoder cell and output to decoder LSTM ]] --
function BiSeq2Seq:forwardConnect(inputSeqLen)
    for i = 1, self.nNumLayerLSTM do
        ---
        -- lstm go forward
        local decodeLstmLayer1 = self.lstmDecodeModules[i][1]
        local encodeLstmLayer1 = self.lstmEncodeModules[i][1]
        local encodeLstmLayer2 = self.lstmEncodeModules[i][2]
        self.ConnectOutput = {
            encodeLstmLayer1.output[inputSeqLen],
            encodeLstmLayer2.output[1]
        }
        self.ConnectCell = {
            encodeLstmLayer1.cell[inputSeqLen],
            encodeLstmLayer2.cell[1]
        }
        
        local encodeOutputConnect = self.connectEncodeDecode:forward(self.ConnectOutput)
        local encodeCellConnect = self.connectEncodeDecode:forward(self.ConnectCell)
        
        decodeLstmLayer1.userPrevOutput =
    	    rnn.recursiveCopy(decodeLstmLayer1.userPrevOutput, encodeOutputConnect)
	    decodeLstmLayer1.userPrevCell =
    	    rnn.recursiveCopy(decodeLstmLayer1.userPrevCell, encodeCellConnect)

    end 
end

--- ----------------------------------------------------------------------------
-- [[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function BiSeq2Seq:backwardConnect(inputSeqLen)
    for i = 1, self.nNumLayerLSTM do
        ---
        -- lstm go forward
        local decodeLstmLayer1 = self.lstmDecodeModules[i][1]
        local encodeLstmLayer1 = self.lstmEncodeModules[i][1]
        local encodeLstmLayer2 = self.lstmEncodeModules[i][2]

        local encodeOutputsConnect = self.connectEncodeDecode:backward(self.ConnectOutput, decodeLstmLayer1:getGradHiddenState(0)[1])
        local encodeCellsConnect = self.connectEncodeDecode:backward(self.ConnectCell, decodeLstmLayer1:getGradHiddenState(0)[2])
        encodeLstmLayer1:setGradHiddenState(inputSeqLen, {encodeOutputsConnect[1], encodeCellsConnect[1]})
        encodeLstmLayer2:setGradHiddenState(1, {encodeOutputsConnect[2], encodeCellsConnect[2]})
    end
end

--- ----------------------------------------------------------------------------
-- [[ Forward ]]--
function BiSeq2Seq:forward(inputEncode, inputDecode, targetDecode, targetEncode)
    self.outputEncode       = self.encoder:forward(inputEncode)
    self:forwardConnect(inputEncode:size(1))
    self.outputDecode       = self.decoder:forward(inputDecode)
    
    self.inputAttention     = self.parseInputAttention:forward({
                                    self.outputEncode,
                                    self.outputDecode
                                })
    self.outputAttention    = self.attention:forward(self.inputAttention)
    local splitTargetDecode = nn.SplitTable(1)
    if (self.isUseCuda == true) then 
        splitTargetDecode = splitTargetDecode:cuda()
    end

    -- for recombine 
    self.outRecombine = self.recombiner :forward(self.outputEncode)
    
    return self.criterionSum :forward(
        {self.outputAttention, self.outRecombine}, 
        {splitTargetDecode(targetDecode), targetEncode}
    )

end

--- ----------------------------------------------------------------------------
-- [[ Backward ]]--
function BiSeq2Seq:backward(inputEncode, inputDecode, targetDecode, targetEncode)
    local splitTargetDecode = nn.SplitTable(1)
    if (self.isUseCuda == true) then 
        splitTargetDecode = splitTargetDecode:cuda()
    end
    local loss = self.criterionSum :backward(
        {self.outputAttention, self.outRecombine}, 
        {splitTargetDecode(targetDecode), targetEncode}
    )
    dloss_dx, dlossRecombine_dx = loss[1], loss[2]
    local gradInputAtt  = self.attention:backward(self.inputAttention, dloss_dx)
    local gradOutLstm   = self.parseInputAttention:backward(self.inputAttention, gradInputAtt)

    self.decoder:backward(inputDecode, gradOutLstm[2])
    self:backwardConnect(inputEncode:size(1))

    -- local dlossRecombine_dx  = self.criterionCombiner:backward( self.outRecombine, targetEncode)
    local gradInputRecombine = self.recombiner:backward(self.outputEncode,  dlossRecombine_dx)
    self.encoder:backward(inputEncode,  gradOutLstm[1]:add(gradInputRecombine))
end


--- ----------------------------------------------------------------------------
function BiSeq2Seq:getParameters()
  return nn.Container()
            :add(self.encoder)
            :add(self.decoder)
            :add(self.parseInputAttention)
            :add(self.attention)
            :add(self.connectEncodeDecode)
            :add( self.recombiner )
            :getParameters()
end

--- ----------------------------------------------------------------------------
function BiSeq2Seq:forget()
    self.encoder:forget()
    self.decoder:forget()
    self.attention:forget() 
    self.recombiner :forget()
    self.parseInputAttention:forget()
    self.connectEncodeDecode:forget()
    return self
end

--- ----------------------------------------------------------------------------
function BiSeq2Seq:training()
    self.encoder:training()
    self.decoder:training()
    self.attention:training()
    self.parseInputAttention:training()
    self.connectEncodeDecode:training()
     self.recombiner :training()
    return self
end

--- ----------------------------------------------------------------------------
function BiSeq2Seq:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
    self.attention:evaluate()
    self.parseInputAttention:evaluate()
    self.connectEncodeDecode:evaluate()
     self.recombiner :evaluate()
    return self
end

--- ----------------------------------------------------------------------------
function BiSeq2Seq:cuda()
    self.isUseCuda = true
    self.encoder:cuda()
    self.decoder:cuda()
    self.attention:cuda()
    self.criterion:cuda()
    self.parseInputAttention:cuda()
    self.connectEncodeDecode:cuda()
    self.recombiner :cuda()
    self.criterionCombiner :cuda()
    self.criterionSum :cuda()
    return self
end

--- ----------------------------------------------------------------------------
function BiSeq2Seq:updateParameters(lr)
    self.encoder:updateParameters(lr)
    self.decoder:updateParameters(lr)
    self.attention:updateParameters(lr)
    self.parseInputAttention:updateParameters(lr)
    self.connectEncodeDecode:updateParameters(lr)
    self.recombiner:updateParameters(lr)
    
    return self
end
--- ----------------------------------------------------------------------------
function BiSeq2Seq:unCuda()
    self.isUseCuda = false
    self.outputEncode  = nil
    self.outputDecode  = nil
    self.inputAttention  = nil
    self.outputAttention  = nil
    self.outRecombine = nil
    self.ConnectOutput = nil
    self.ConnectCell = nil

    self.encoder:double()
    self.decoder:double()
    self.attention:double()
    self.criterion:double()
    self.parseInputAttention:double()
    self.connectEncodeDecode:double()
    self.recombiner :double()
    self.criterionCombiner :double()
    self.criterionSum :double()
    return self
end

--- ----------------------------------------------------------------------------
function BiSeq2Seq:clearState()
    
    self.encoder:clearState()
    self.decoder:clearState()
    self.attention:clearState()
    self.parseInputAttention:clearState()
    self.connectEncodeDecode:clearState()
    self.recombiner :clearState()
    
    return self
end


--- ----------------------------------------------------------------------------
-- eval accuracy
function BiSeq2Seq:eval_batch_with_oldEncBatch(enc_batch_input, dec_batch_input)

    local moduleRebase = nn.View(dec_batch_input:size(1), dec_batch_input:size(2))
    local moduleConvertAttention = nn.Sequential()
    :add(nn.Sequencer(nn.View(-1,1):setNumInputDims(1)))
    :add(nn.JoinTable(2,2))
    :add(nn.Transpose({2,3},{1,2}))

    -- check cuda
    if (self.isUseCuda) then
        if enc_batch_input ~= nil then
            enc_batch_input:cuda()
        end
        dec_batch_input:cuda()
        moduleConvertAttention:cuda()
        moduleRebase:cuda()
    end

    -- forward: encoder -> decoder => attention
    if enc_batch_input ~= nil then
        self.outputEncode = self.encoder:forward(enc_batch_input)
        self:forwardConnect(enc_batch_input:size(1))
    end
    self.outputDecode = self.decoder:forward(dec_batch_input)
    self.inputAttention = self.parseInputAttention:forward({self.outputEncode, self.outputDecode })
    self.outputAttention = self.attention:forward(self.inputAttention)

    -- debug
    local _, wordIds = moduleConvertAttention(self.outputAttention):topk(1, 3, true, true)

    -- rebase matrix to get result
    if (self.isUseCuda) then
        wordIds = wordIds:cuda()
    else
        wordIds = wordIds:double()
    end
    local result = moduleRebase(wordIds)

    -- remove padding
    local mergeResult = result:double():map(dec_batch_input:double(), function (x_val, y_val)
        if (y_val == 0) then
            return 0 -- this is padding (err when take topk)
        else
            return x_val
        end

    end)

    -- forget all weight .. useful when doing both training and testing
    self:forget()

    if (self.isUseCuda) then
        mergeResult = mergeResult:cuda()
    end
    return mergeResult
end


--- ----------------------------------------------------------------------------
-- eval accuracy
function BiSeq2Seq:eval_batch_recombine(enc_batch_input)

    local moduleRebase = nn.View(enc_batch_input:size(1), enc_batch_input:size(2))
    
    -- check cuda
    if (self.isUseCuda) then
        enc_batch_input:cuda()
        moduleRebase:cuda()
    end

    -- forward: encoder -> decoder => attention
    self.outputEncode = self.encoder:forward(enc_batch_input)
    local outRecombine = self.recombiner:forward( self.outputEncode)
    local _, wordIds = outRecombine:topk(1, 3, true, true)

    -- rebase matrix to get result
    if (self.isUseCuda) then wordIds = wordIds:cuda()
    else  wordIds = wordIds:double() end
    local result = moduleRebase(wordIds)

    -- remove padding
    local mergeResult = result:double():map(enc_batch_input:double(), function (x_val, y_val)
        if (y_val == 0) then return 0 -- this is padding (err when take topk)
        else return x_val end
    end)

    -- forget all weight .. useful when doing both training and testing
    self:forget()

    if (self.isUseCuda) then
        mergeResult = mergeResult:cuda()
    end
    return mergeResult
end

--- ----------------------------------------------------------------------------
-- overide to string 
function BiSeq2Seq:__tostring__()
    local nameE = 
    "\n==============\n" 
     .. "= self.encoder\n"
     .. "==============\n" .. self.encoder:__tostring__()
    .. "\n==============\n" 
     .. "= self.decoder\n"
     .. "==============\n"
     ..  self.decoder:__tostring__() .. "\n"
    local name = "Bidirectional RNN {\n\t%s\n}"
    local nameSeqLstm = ("%dxDeepSeqLSTM(%d) + %dxDeepSeqLSTM(%d)")
        :format(self.nNumLayerLSTM, self.lengWordVector,self.nNumLayerLSTM, self.lengWordVector)
    name = name:format (nameSeqLstm)
  return name .. nameE --("BiSeq2Seq(%dx2)"):format(self.lengWordVector)
end

