-- init package
-- luarocks install utf8
-- luarocks install luautf8
-- luarocks install restserver-xavante
-- luarocks install --server=http://luarocks.org/dev lua-zip ZIP_LIBDIR=/usr/lib/x86_64-linux-gnu/ ZIP_DIR=/usr/
-- # ZIP_LIBDIR contain libzip.a, ZIP_DIR contain libzip.a  zip.h
-- #if not found zip.h -- apt-get install libzip4 libzip-dev


-- init curent global
if not RnnSemanticParser then
    require 'torch'
    require "optim"
    require 'xlua'
    require "pl.List"
    require('pl.stringx').import()
    require('pl.seq')
    utf8 = require('lua-utf8')
    restserver = require("restserver")

    RnnSemanticParser = {
        --- @type ModelGeneral
        model = {},
        dataset={}
    }

    if loadstring == nil then
        loadstring = load
    end

    -- for init code model
    MinibatchLoader = require("MinibatchLoader")
    require("Model")
    require("ModelCombine")
    require("ModelSemantic")

    require("SymbolsManager")
    require("logicInference.NlpNormalize")

    g_reloadSystemManual = true
end

if (g_reloadSystemManual) then

    -- disable reaload
    g_reloadSystemManual = false

    require("ParamsParser")
    require("utils")
    require("Main")
    require("logicInference.LogicUnzip")
    require("logicInference.LogicSystem")

    -- init manual seed to gen random value without change
    torch.manualSeed(0)
end