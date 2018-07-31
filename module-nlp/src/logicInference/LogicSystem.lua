--
-- Created by IntelliJ IDEA.
-- User: phuongnm
-- Date: 08/08/2017
-- Time: 13:13
-- To change this template use File | Settings | File Templates.
--

require('torch')
require('sys')
Date = require('pl.Date')
JSON = require("JSON")
threads = require('threads')
require("ParamsParser")
require("utils")
require("logicInference.NlpNormalize")
require("logicInference.GlobalConfig")

if (loadstring == nil) then
    loadstring = load
end
if not opt then
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

print ("- Load: LogicSystem.lua")

function processTaskRecieveMessage(task_submission)
    local bRun, result = pcall(_processTaskRecieveMessage, task_submission)
    if bRun then
        return result
    else
        print (result)
        local tblFormRet ={
            form = "",
            ner = {},
            msg = "",
            windows = {
                {
                    payload = ".....",
                    type = TYPE_SHOW_MESSAGE
                }
            }
        }
        return restserver.response():status(200):entity(tblFormRet)
    end
end

-- decode json string to table
function decodeJson(task_submission)
    return JSON:decode(task_submission)
end

--
function _processTaskRecieveMessage(task_submission)
    -- decode message
    local message
    local bRun, tblInfo = pcall(decodeJson, task_submission)

    local tblFormRet = {
        form = "",
        windows = {}
    }
    if (bRun == true) then
        -- get info of request
        local event = tblInfo
        if not event then
            print ("[Err] not exist event")
            goto _EXIT_PROCESS_EVENT
        end
        message = event.message

        -- message check tokenizer
        if message then
            message = string.gsub(message, '[%.\'"?!,+_*#$%%&()]', ' ')
        end

        -- nlp processing if this is message of user
        message = normalize(message, opt.useAcent)

        -- model infer
        local formLogic, mapNer = model:eval(utf8.lower(message), opt)
        local formLogicModel = formLogic
        print ("- Form logic: "..formLogic)
        formLogic = luaUnzip(formLogic)

        -- replace NER
        for k, v in pairs(mapNer)do
            v = v:replace("_", " ")
            if formLogic:find(k) ~= nil then
                formLogic = formLogic:replace(k .. " ", "'"..v.."' ")
                mapNer[k] = nil
            end
        end

        -- run logic form
        local windows = runLogicSystem(formLogic )


        tblFormRet={
            form = formLogicModel,
            msg = message,
            windows = windows
        }

        goto _EXIT_PROCESS_EVENT
    else
        -- run not ok
        print ("parse json error")
        goto _EXIT_PROCESS_EVENT
    end

    ::_EXIT_PROCESS_EVENT::
    print(tblFormRet)
    return restserver.response():status(200):entity(tblFormRet)
end

function _runLogicSystem(form)
    local func = loadstring(form)
    return func()

end

function runLogicSystem(form)
    local bRun , windows = pcall(_runLogicSystem, form )
    --_runLogicSystem(form)
    if bRun == true and windows ~= nil then
        return windows
    else
        print ("[Exception]")
        return {MessageWindow("Sorry u, I have something error!!"):getContent()}
    end
end


-- ------------------------------------------------------------------------------
-- define Action window
---@class ActionWindow
ActionWindow = nil
torch.class("ActionWindow")

function ActionWindow:__init()
    self.content = {}
end

function ActionWindow:getContent()
    return self.content
end

-- define Message window
---@class MessageWindow : ActionWindow
MessageWindow = nil
torch.class("MessageWindow", "ActionWindow")

function MessageWindow:__init(content)
    self.content = {
        payload = content,
        type = TYPE_SHOW_MESSAGE
    }
end

-- define Take a photo window
---@class TakePhotoWindow : ActionWindow
TakePhotoWindow = nil
torch.class("TakePhotoWindow", "ActionWindow")

function TakePhotoWindow:__init()
    self.content = {
        type = TYPE_REMOTE_TAKE_A_PHOTO
    }
end

---@class ShowThingsBotSeeWindow : ActionWindow
ShowThingsBotSeeWindow = nil
torch.class("ShowThingsBotSeeWindow", "ActionWindow")

function ShowThingsBotSeeWindow:__init(messageIfSeeSomething, msgIfNot)
    self.content = {
        payload = messageIfSeeSomething,
        type = TYPE_SHOW_THINGS_BOT_SEE
    }
end

---@class DoWindow : ActionWindow
DoWindow = nil
torch.class("DoWindow", "ActionWindow")

function DoWindow:__init()
    self.content = MessageWindow("Now, I just can take a photo for u. Sorry and I love u <3."):getContent()
end

---@class KnowWindow : ActionWindow
KnowWindow = nil
torch.class("KnowWindow", "ActionWindow")

function KnowWindow:__init(content)
    -- check some content
    if content =="phuong" then
        self.content = MessageWindow("Yes, I known, phuong is my father."):getContent()
    else
        if content == nil then
            self.content = DoWindow():getContent()
        else
            self.content = MessageWindow(("No, I dont know %s. Pls find %s on google.")
                    :format(content, content))
                    :getContent()
        end
    end
end

---@class CheckSeeSomethingWindow : ActionWindow
CheckSeeSomethingWindow = nil
torch.class("CheckSeeSomethingWindow", "ActionWindow")

function CheckSeeSomethingWindow:__init(content)
    -- check some content
    self.content = {
        payload = content,
        type = TYPE_CHECK_THINGS_BOT_SEE
    }
end


---@class PositionCanSeeWindow : ActionWindow
PositionCanSeeWindow = nil
torch.class("PositionCanSeeWindow", "ActionWindow")

function PositionCanSeeWindow:__init(content)
    -- check some content
    self.content = {
        payload = content,
        type = TYPE_POSITION_BOT_SEE_THING
    }
end

---@class ComebackPositionCanSeeWindow : ActionWindow
ComebackPositionCanSeeWindow = nil
torch.class("ComebackPositionCanSeeWindow", "ActionWindow")

function ComebackPositionCanSeeWindow:__init(content)
    -- check some content
    self.content = {
        payload = content,
        type = TYPE_COMEBACK_POSITION_BOT_SEE_THING
    }
end

-- -- ------------------------------------------------------------------------------
--[[
-- some logic
what ( can ( bot ( ) _do ( ) ) )
        can ( bot ( ) know ( _Something ) )
        can ( bot ( ) takeaphoto ( ) )
                      takeaphoto ( )
        can ( bot ( ) see (_Something ) )
        can ( bot ( ) comeback ( position( can ( see ( _Something ) ) ) ) )
                                 position( can ( see ( _Something ) ) )
hello ( )
bye ( )
thanks()
 --]]

function thanks ( )
    return { MessageWindow("Yes. Glad to talk with u"):getContent() }
end

function hello ( )
    return { MessageWindow("Hi :D"):getContent() }
end

function bye ( )
    return { MessageWindow("See u again"):getContent() }
end

function takeaphoto ( )
    return {
        MessageWindow("Ok, i 'll take a photo for u"):getContent(),
        TakePhotoWindow():getContent()
    }
end

function see ( Something )
    if Something == nil then
        return {
            ShowThingsBotSeeWindow("This is something I'm seeing ",
            "I'm not seeing any object now.")
                    :getContent(),
            TakePhotoWindow():getContent()
        }
    else
        return {
            CheckSeeSomethingWindow(Something):getContent(),
            TakePhotoWindow():getContent()
        }
    end
end

function _do ( )
    return {
        DoWindow():getContent()
    }
end

function know ( Something )
    return {
        KnowWindow(Something):getContent()
    }
end

function name ( )
    return {
        MessageWindow("I'm Botdy. Thanks u."):getContent()
    }
end

function spam ( )
    return {
        MessageWindow("Something u talk, i have not hear that."):getContent(),
        DoWindow():getContent(),
    }
end

function bot()
    return nil
end

function what(event)
    return event
end

function can(object, method)
    if method ~= nil then
        return method
    elseif object ~= nil then
        return object
    end
end

function yes()
    return {MessageWindow("^^"):getContent(), }
end

function no()
    return {MessageWindow("Yes."):getContent(),}
end


function position(event)
    if event ~= nil then
        local something = nil
        for _, window in pairs(event) do
            if window.type == TYPE_CHECK_THINGS_BOT_SEE then
                something = window.payload
                break
            end
        end
        if something ~= nil then
            return {
                PositionCanSeeWindow(something):getContent()
            }
        end
        return event
    end
    return {}
end

function comeback(event)
    if event ~= nil then
        local something = nil
        for _, window in pairs(event) do
            if window.type == TYPE_POSITION_BOT_SEE_THING then
                something = window.payload
                break
            end
        end
        if something ~= nil then
            return {
                ComebackPositionCanSeeWindow(something):getContent()
            }
        end
        return event
    end
    return {
        MessageWindow("which position i need comeback ??"):getContent()
    }
end
--processTaskRecieveMessage(JSON:encode({message = "what can u do"}))
--processTaskRecieveMessage(JSON:encode({message = "hello"}))