---@class MinibatchLoader

local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

local function to_vector_list(l)
    for i = 1, #l do
        l[i] = l[i][{ {}, 1 }]
    end
    return l
end

---@return MinibatchLoader
function MinibatchLoader.create(opt, name, word_manager, form_manager)
    local self = {}
    setmetatable(self, MinibatchLoader)

    -- save info dict
    self.word_manager = word_manager
    self.form_manager = form_manager
    local start_token = word_manager:get_symbol_idx("<S>")
    local end_token = word_manager:get_symbol_idx("<E>")
    local none_token = word_manager:get_symbol_idx("<none>")

    local data_file = path.join(opt.data_dir, name)
    print('- \tloading data: ' .. name)
    local data = torch.load(data_file)

    -- batch padding
    if #data % opt.batch_size ~= 0 then
        local n = #data
        for i = 1, #data % opt.batch_size do
            table.insert(data, n - i + 1, data[n - i + 1])
        end
    end

    self.enc_batch_list = {}
    self.enc_len_batch_list = {}
    self.dec_batch_list = {}
    self.enc_raw_list = {}
    self.enc_target_list = {}
    self.enc_target_recombine_list = {}
    local p = 0
    local newData = {}
    local mtIdx = torch.randperm(#data)
    for i = 1, #data do
        newData[i] = data[mtIdx[i]]
    end
    data = newData
    newData = nil
    if (#data > 0 and #data[1] > 2) then
        self.bGenDataCombine = true
    else
        self.bGenDataCombine = false
        print("[W] not gen data combine ")
    end

    while p + opt.batch_size <= #data do
        -- build enc matrix --------------------------------
        local max_len = -1
        for i = 1, opt.batch_size do
            local w_list = data[p + i][1]
            if #w_list > max_len then
                max_len = #w_list
            end
        end

        local m_text = torch.zeros(opt.batch_size, max_len + 2)
        local enc_len_list = {}

        for i = 1, opt.batch_size do
            local w_list = data[p + i][1]

            -- add <S>
            m_text[i][max_len + 2 - (#w_list + 2) + 1] = start_token

            -- add <E>
            m_text[i][max_len + 2] = end_token

            -- reversed order
            for j = 1, #w_list do
                m_text[i][max_len + 2 - (#w_list + 2) + j + 1] = w_list[#w_list - j + 1]
                -- m_text[i][j + 1] = w_list[j]
            end

            table.insert(enc_len_list, #w_list + 2)
        end
        table.insert(self.enc_batch_list, m_text)
        table.insert(self.enc_len_batch_list, enc_len_list)

        -- build dec matrix --------------------------------
        max_len = -1
        for i = 1, opt.batch_size do
            local w_list = data[p + i][2]
            if #w_list > max_len then
                max_len = #w_list
            end
        end
        m_text = torch.zeros(opt.batch_size, max_len + 2)
        -- add <S>
        m_text[{ {}, 1 }] = 1
        for i = 1, opt.batch_size do
            local w_list = data[p + i][2]
            for j = 1, #w_list do
                m_text[i][j + 1] = w_list[j]
            end
            -- add <E>
            m_text[i][#w_list + 2] = end_token
        end
        table.insert(self.dec_batch_list, m_text)

        -- build combine data matrix --------------------------------
        if (self.bGenDataCombine == true) then
            max_len = -1
            for i = 1, opt.batch_size do
                local w_raw = data[p + i][3]
                local w_target = data[p + i][4]
                assert(#w_raw == #w_target, "[E] data combine not validate")
                if #w_raw > max_len then
                    max_len = #w_raw
                end
            end
            m_raw = torch.zeros(opt.batch_size, max_len + 2)
            m_target = torch.zeros(opt.batch_size, max_len + 2)
            m_target_combine = torch.zeros(opt.batch_size, max_len + 2)

            for i = 1, opt.batch_size do
                local size_m_raw = #(data[p + i][3])

                -- add <E>
                m_raw[i][max_len + 2 - (size_m_raw + 2) + 1] = start_token
                m_target[i][max_len + 2 - (size_m_raw + 2) + 1] = start_token

                for j = 1, size_m_raw do
                    m_raw[i][max_len + 2 - (size_m_raw + 2) + j + 1] = data[p + i][3][size_m_raw - j + 1]
                    m_target[i][max_len + 2 - (size_m_raw + 2) + j + 1] = data[p + i][4][size_m_raw - j + 1]
                end

                -- add <E>
                m_raw[i][max_len + 2] = end_token
                m_target[i][max_len + 2] = end_token -- add none id

                for j = 1, max_len+2 do
                    if m_raw[i][j] == m_target[i][j] and m_target[i][j] ~= 0 then
                        m_target_combine[i][j] = none_token
                    elseif m_raw[i][j] ~= m_target[i][j] and m_target[i][j] ~= 0 and m_raw[i][j]~= 0 then
                        m_target_combine[i][j] = m_target[i][j]
                    end
                end
            end
            table.insert(self.enc_raw_list, m_raw)
            table.insert(self.enc_target_list, m_target)
            table.insert(self.enc_target_recombine_list, m_target_combine)

        end
        p = p + opt.batch_size
    end

    -- reset batch index
    self.num_batch = #self.enc_batch_list
    self.num_sample = #data

    assert(#self.enc_batch_list == #self.dec_batch_list)

    collectgarbage()
    return self
end

function MinibatchLoader:random_batch()
    local p = math.random(self.num_batch)
    return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]
end

function MinibatchLoader:all_batch()
    local r = {}
    for p = 1, self.num_batch do
        table.insert(r, { self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p] })
    end
    return r
end

function MinibatchLoader:transformer_matrix(isUsedCuda)
    -- load batch data
    self.decIn = {}
    self.decOut = {}
    for p = 1, self.num_batch do
        -- local enc_batch, enc_len_batch, dec_batch = self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]

        -- for parse logic form with batch
        self.enc_batch_list[p] = self.enc_batch_list[p]:t()
        self.decIn[p] = torch.Tensor(self.dec_batch_list[p]:size(1), self.dec_batch_list[p]:size(2) - 1):copy((self.dec_batch_list[p][{ {}, { 1, -2 } }])):t()

        for col = 1, self.decIn[p]:size(2) do
            if (self.decIn[p][self.decIn[p]:size(1)][col] == 0
                    or self.decIn[p][self.decIn[p]:size(1)][col] == 2) then
                local row = self.decIn[p]:size(1)
                while row > 0 do
                    if (self.decIn[p][row][col] ~= 0) then
                        self.decIn[p][row][col] = 0
                        bcheck = true
                        break
                    end
                    row = row - 1
                end
            end
        end

        self.decOut[p] = torch.Tensor(self.dec_batch_list[p]:size(1), self.dec_batch_list[p]:size(2) - 1):copy(self.dec_batch_list[p][{ {}, { 2, -1 } }]):t()

        -- for combine data
        if self.bGenDataCombine == true then
            self.enc_raw_list[p] = self.enc_raw_list[p]:t()
            self.enc_target_list[p] = self.enc_target_list[p]:t()
            self.enc_target_recombine_list[p] = self.enc_target_recombine_list[p]:t()
        end

        -- cuda matrix to run with gpu
        if (isUsedCuda) then
            self.decIn[p] = self.decIn[p]:cuda()
            self.decOut[p] = self.decOut[p]:cuda()
            self.enc_batch_list[p] = self.enc_batch_list[p]:cuda()

            if self.bGenDataCombine == true then
                self.enc_raw_list[p] = self.enc_raw_list[p]:cuda()
                self.enc_target_list[p] = self.enc_target_list[p]:cuda()
                self.enc_target_recombine_list[p] = self.enc_target_recombine_list[p]:cuda()
            end
        end
    end
    self.dec_batch_list = nil
end


function MinibatchLoader:random_matrix()
    local p = math.random(self.num_batch)
    return self.enc_batch_list[p], self.decIn[p], self.decOut[p],
    self.enc_raw_list[p], self.enc_target_list[p], self.enc_target_recombine_list[p]
end

function MinibatchLoader:get_matrix(idx)
    return self.enc_batch_list[idx], self.decIn[idx], self.decOut[idx],
    self.enc_raw_list[idx], self.enc_target_list[idx], self.enc_target_recombine_list[idx]
end

function MinibatchLoader:gen_rate_train_label(tblDataLabel, lengLabel)
    -- init config var
--    if (self.mtRateTrain ~= nil) then
--        return self.mtRateTrain
--    end
    if (tblDataLabel == nil) then
        tblDataLabel = self.enc_target_list
    end
    local statsTrain = {}

    -- count for each batch
    for k, batch in pairs(tblDataLabel) do
        for i = 1, batch:size(1) do
            for j = 1, batch:size(2) do
                -- skip padding
                if (batch[i][j] ~= 0) then
                    if (statsTrain[batch[i][j]] == nil) then
                        statsTrain[batch[i][j]] = 1
                    else
                        statsTrain[batch[i][j]] = 1 + statsTrain[batch[i][j]]
                    end
                end
            end
        end
    end

    -- get max key id
    local maxKey = 0
    local countAllLabel = 0
    for id, val in pairs(statsTrain) do
        if (maxKey < id) then maxKey = id end
        countAllLabel = countAllLabel + val
    end
    if (lengLabel ~= nil and maxKey < lengLabel) then
        maxKey = lengLabel
        print("[W] some label not exist in data train")
    end

    -- calculate matrix
    self.mtRateTrain = torch.Tensor(maxKey):fill(0.00)
    for i = 1, maxKey do
        if (statsTrain[i] ~= nil and statsTrain[i] ~= 0) then
            self.mtRateTrain[i] = countAllLabel * 1.0 / statsTrain[i]
        end
    end
    return self.mtRateTrain, maxKey
end

return MinibatchLoader
