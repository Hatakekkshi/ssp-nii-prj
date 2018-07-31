  
function ParamsParser()

    --[[ command line arguments ]]--
    cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Semantic parsing')
    cmd:text('Options:')
    cmd:text()

    -- training
    cmd:text('Training options')
    cmd:option('--lr', 0.001, 'learning rate - Toc do hoc cua mang ')
    cmd:option('--momentum', 0.95, 'momentum - Giam do dao dong khi hoi tu')
    cmd:option('--decayRate', 0.95, 'decayrate - ')
    cmd:option('--batch_size', 1, 'so cau trong 1 batchInputs')
    cmd:option('--lengWordVector', 300, 'word vector size')
    cmd:option('--lengLabel', 150, 'count label predict')
    cmd:option('--lengDict', 500, 'size Dictionary')
    -- cmd:option('--maxEpoch', 200, 'so lan lap lai toan bo du lieu')
    cmd:option('--maxEpochCombine', 80, 'so lan lap lai toan bo du lieu')
    cmd:option('--maxEpochSemantic', 200, 'so lan lap lai toan bo du lieu')
    cmd:option('--minLR', 0.00001, 'minimum learning rate')
    cmd:option('--saturateEpoch', 0, 'epoch at which linear decayed LR will reach minLR')
    cmd:option('--nNumLayerLstmIntermediate', 2, 'count deep lstm layer ')
    cmd:option('--dropoutRate', 0.5, 'rate to drop out')
    cmd:option('--isUseTokenizer', true, 'use tokenizer from module java')
    cmd:option('--isUsedCuda', true, 'use Cuda gpu')
    cmd:option('--training', false, 'training with data ')
    cmd:option('--testing', false, 'tst with data sample')
    cmd:option('--compressDataCombine', true, 'replace "wordid" duplicate in output data combine')
    cmd:option('--modelType', "concat", 'to select the way to train model "combine" and "semantic" [multitask]')
    cmd:option('--nameDataTrain','train-%d.t7', 'data train')
    cmd:option('--dataNameLogicform', 'data.logicform.t7', 'data distinct logic form')
    cmd:option('--nameDataTest', 'test-%d.t7', 'data test')
    cmd:option('--idxFold', 4, 'idx fold to cross validate')
    cmd:option('--data_dir', '../data_cgd/', 'path folder save data seriable')
    cmd:option('--model_dir', '../model_prolog_cgd/', 'path folder save data seriable')
    cmd:option('--eval', false, 'run test with new sample ')
    cmd:option('--nameNN', "Seq2Seq", 'Cai dat ten mang Neron: Seq2SeqNoAtt/Seq2Seq/BiSeq2Seq')
    cmd:option('--loadPersonSystem', true, 'load content profile user from disk')
    cmd:option('--serverPort', 18081, 'load content profile user from disk')
    cmd:option('--customer', "becksport", 'customer is using this bot')
    cmd:option('--customerDbIp', "104.199.172.165", 'customer db ip')
    cmd:option('--customerDbPort', "8000", 'customer db port')
    cmd:option('--onServer', false, 'is run on server')
    cmd:option('--useAcent', false, 'is use acent to normal text')
    cmd:option('--runInDocker', false, 'is run in docker ')

    -- loging
    cmd:text()
    cmd:text('Loging options')
    cmd:option('--isSaveLog', true, 'Ghi log ra file')
    cmd:option('--logPrintNegative', false, 'is print output sentence negative')
    cmd:option('--nameLog', "log", 'Ten file log')
    cmd:option('--hasTimeLog', false, 'xuat thoi gian trong log')

    cmd:text()


    local opt = cmd:parse(arg or {})

    opt.nameDataTrain = string.format( opt.nameDataTrain, opt.idxFold)
    opt.nameDataTest = string.format( opt.nameDataTest, opt.idxFold)
    -- create folder save log
    if(opt.isSaveLog == true) then
        require 'paths'

        if (opt.hasTimeLog == true) then
                cmd:addTime('rnn')
        end

        opt.rundir = cmd:string('../logs', {}, {dir=true})
        paths.mkdir(opt.rundir)
        paths.mkdir(opt.model_dir)

        opt.nameLog = string.format("%s-data:%s-fold:%d-deep:%d-batch:%d-maxEpoch:%d-model:%s-hidden:%d.log",
            opt.nameLog,string.gsub(opt.data_dir, '[%./]', ""), opt.idxFold, opt.nNumLayerLstmIntermediate,
            opt.batch_size, opt.maxEpochSemantic --[[os.time()--]], opt.modelType, opt.lengWordVector )
        cmd:log(opt.rundir .. '/'.. opt.nameLog, opt)
    end
    return opt
end
--ParamsParser()
