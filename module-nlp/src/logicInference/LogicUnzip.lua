--
-- Created by IntelliJ IDEA.
-- User: phuongnm
-- Date: 27/07/2017
-- Time: 09:24
-- To change this template use File | Settings | File Templates.
--

require 'pl'
stringx = require 'pl.stringx'
stringx.import()
require('utils')


local function checkOpen(token)
    return token == "("
end
local function checkOpenString(token)
    return token == "'" or token == "\""
end

local function checkClose(token)
    return token == ")"
end

local function checkSpecialSymbol(token)
    if (token == nil) then return false end
    return token:sub(1,1) == '\\' or token:sub(1,1) == '_'
end

local function checkFunction(token)
    if (token == nil) then return false end
    local newStr = token:sub(1,1)
    return checkSpecialSymbol(token) or (newStr:isalnum() and newStr:islower())
end

local function checkBeginString(token)
    if (token == nil) then return false end
    return token:startswith ("'")
end

local function checkEndString(token)
    if (token == nil) then return false end
    return token:endswith ("'")
end

function luaUnzip(sPrologCmd)
    sPrologCmd = sPrologCmd:gsub ("'([^']*) ([^']*)'" , "'%1_cc_%2'")
    sPrologCmd = sPrologCmd:replace ("what(" , "what (")
    local lstToken = sPrologCmd:split(" ")
    local prev
    local newList = cloneTable(lstToken)
    local idxNew = 1
    local bInString = false
    local levelOpen = 0

    for idx, cur in ipairs(lstToken)do
        if checkBeginString(cur) then
            bInString = true
        end

        if checkOpen(cur) then
            levelOpen = levelOpen + 1
        elseif checkClose(cur) then
            levelOpen = levelOpen - 1
        end
        if  checkClose(cur) or
                checkSpecialSymbol(prev) or
                (checkOpen(cur) and (checkFunction(prev) or
                        prev == nil or
                        checkOpen(prev))) then
            -- do nothing
        else
            if prev ~= nil and checkFunction(cur) == true and checkOpen(prev) == false then
                -- insert token connect
                local token = ';'
                if levelOpen > 0 then token = ','  end
                table.insert(newList, idxNew, token)
                idxNew = idxNew + 1
            elseif prev ~= nil and checkOpen(prev) == false then
                -- insert ',' before
                table.insert(newList, idxNew, ',')
                idxNew = idxNew + 1
            end
        end

        if checkEndString(cur) then
            bInString = false
        end

        idxNew = idxNew + 1
        prev = cur
    end
    local sRet = (' '):join(newList)
    sRet = sRet:gsub ("'([^']*)_cc_([^']*)'" , "'%1 %2'")
    return ('return %s'):format(sRet)
end

