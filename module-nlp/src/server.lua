-- th server.lua --data_dir ../data_geo_prolog_entity/  --testing   --nameLog multitask_prolog_test --idxFold  5 --model_dir ../model_prolog/ --saturateEpoch 250 --maxEpochSemantic 250 --batch_size 10 --lengWordVector 200  --nNumLayerLstmIntermediate 1 --modelType multitask --lr 0.002
--
-- User: phuongnm
--

JSON = require("JSON")
inspect = require("inspect")
require("Package")


local server = restserver:new():port(opt.serverPort)
server:add_resource("chat-bot", {

    {
        method = "GET",
        path = "/",
        produces = "application/json",
        handler = function()
            return restserver.response():status(200):entity({"hi"})
        end,
    },

    {
        method = "POST",
        path = "/get-windows",
        consumes = "text/plain",
        produces = "application/json",
        handler = function(task_submission)
            print ("=================================")
            --if opt.runInDocker then task_submission = task_submission.POST.post_data end
            task_submission = task_submission.POST.post_data
            print (task_submission)

            return processTaskRecieveMessage(task_submission)
        end,
    },

})

-- This loads the restserver.xavante plugin
server:enable("restserver.xavante"):start()



