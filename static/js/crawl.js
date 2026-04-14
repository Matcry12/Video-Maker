        async function fetchSourceDraft() {
            const source = (document.getElementById('contentSource')?.value || 'source_1').trim();
            const topic = (document.getElementById('sourceTopic')?.value || '').trim();
            const language = (document.getElementById('scriptLang')?.value || 'vi-VN').trim();
            const llmProvider = (document.getElementById('sourceLlmProvider')?.value || 'groq').trim().toLowerCase();

            if (!topic) {
                showApiError(
                    {
                        code: 'TOPIC_REQUIRED',
                        message: 'Topic is required for source fetch.',
                        hint: 'Enter a topic like "Ha Long Bay" and try again.',
                    },
                    'Topic is required for source fetch.',
                );
                return;
            }

            const payload = {
                source,
                topic,
                language,
                max_blocks: 8,
            };
            if (llmProvider) payload.llm_provider = llmProvider;

            const status = document.getElementById('sourceFetchStatus');
            if (status) status.textContent = 'Fetching draft from source_1...';
            showStatus('Fetching source draft...', 'processing');
            clearErrorPanel();

            try {
                const data = await fetchApi(
                    '/api/content/fetch',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    },
                    'Failed to generate source draft.',
                );
                if (!_hasSourceDraftSections(data?.source_draft)) {
                    throw normalizeApiError(
                        {
                            code: 'INVALID_SOURCE_DRAFT',
                            message: 'Source API returned an invalid section draft.',
                            hint: 'Try another topic keyword and fetch again.',
                        },
                        'Failed to generate source draft.',
                    );
                }

                latestSourceDraft = data.source_draft;
                latestHandledScript = null;
                setSourceDraftEditorValue(latestSourceDraft);
                renderSourceDraftResult(data);
                renderSourceHandleResult({
                    handling_mode: 'fetch',
                    script: null,
                    llm_meta: { provider: llmProvider, status: 'pending' },
                    warnings: [],
                    meta: {
                        section_count: Array.isArray(latestSourceDraft?.sections) ? latestSourceDraft.sections.length : 0,
                        raw_block_count: 0,
                        output_block_count: 0,
                    },
                });
                showStatus('Source draft fetched.', 'done');
            } catch (err) {
                setSourceBadge('');
                showApiError(err, 'Failed to generate source draft.');
            }
        }

        async function runSourceHandling(applyToEditor = false) {
            const mode = (document.getElementById('sourceHandlingMode')?.value || 'llm_script').trim().toLowerCase();
            const language = (document.getElementById('scriptLang')?.value || 'vi-VN').trim() || 'vi-VN';
            const llmProvider = (document.getElementById('sourceLlmProvider')?.value || 'groq').trim().toLowerCase();
            const promptOverride = (document.getElementById('sourcePromptOverride')?.value || '').trim();

            if (!['raw_json', 'raw_script', 'llm_script'].includes(mode)) {
                showApiError(
                    {
                        code: 'INVALID_HANDLING_MODE',
                        message: 'Select a valid draft handling mode.',
                    },
                    'Invalid handling mode.',
                );
                return;
            }
            if (promptOverride.length > 4000) {
                showApiError(
                    {
                        code: 'INVALID_PROMPT_OVERRIDE',
                        message: 'Prompt override is too long.',
                        hint: 'Keep prompt override under 4000 characters.',
                    },
                    'Invalid prompt override.',
                );
                return;
            }

            let sourceDraft = null;
            try {
                sourceDraft = readSourceDraftEditorValue();
            } catch (err) {
                showApiError(err, 'Invalid raw draft JSON.');
                return;
            }

            const status = document.getElementById('sourceHandleStatus');
            if (status) status.textContent = `Handling draft (${mode})...`;
            showStatus(`Handling source draft (${mode})...`, 'processing');
            clearErrorPanel();

            const payload = {
                source_draft: sourceDraft,
                handling_mode: mode,
                language,
                llm_provider: llmProvider || 'groq',
            };
            if (promptOverride) payload.prompt_override = promptOverride;

            try {
                const data = await fetchApi(
                    '/api/content/handle',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    },
                    'Failed to handle source draft.',
                );
                if (_hasSourceDraftSections(data?.source_draft)) {
                    latestSourceDraft = data.source_draft;
                    setSourceDraftEditorValue(latestSourceDraft);
                }
                renderSourceHandleResult(data);
                if (applyToEditor && latestHandledScript && Array.isArray(latestHandledScript.blocks) && latestHandledScript.blocks.length) {
                    const savedName = await saveHandledScriptFromCrawl();
                    applyHandledScriptToEditor();
                    setWorkspaceView('editor');
                    setTab('load');
                    showStatus(`Saved crawl result as "${savedName}" and loaded it into editor.`, 'done');
                    return;
                }
                showStatus('Draft handling completed.', 'done');
            } catch (err) {
                showApiError(err, 'Failed to handle source draft.');
            }
        }

        function parseSectionSelectorInput(rawValue) {
            const value = String(rawValue || '').trim();
            if (!value) return [];

            const selected = new Set();
            for (const chunk of value.split(',')) {
                const token = chunk.trim();
                if (!token) continue;
                if (token.includes('-')) {
                    const [startRaw, endRaw] = token.split('-', 2).map(part => part.trim());
                    const start = Number.parseInt(startRaw, 10);
                    const end = Number.parseInt(endRaw, 10);
                    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < 1) {
                        throw normalizeApiError(
                            {
                                code: 'INVALID_SECTION_SELECTOR',
                                message: `Invalid section range "${token}".`,
                                hint: 'Use values like 1-4, 5, 10.',
                            },
                            'Invalid section selector.',
                        );
                    }
                    const low = Math.min(start, end);
                    const high = Math.max(start, end);
                    for (let index = low; index <= high; index += 1) selected.add(index);
                    continue;
                }

                const number = Number.parseInt(token, 10);
                if (!Number.isInteger(number) || number < 1) {
                    throw normalizeApiError(
                        {
                            code: 'INVALID_SECTION_SELECTOR',
                            message: `Invalid section number "${token}".`,
                            hint: 'Use values like 1-4, 5, 10.',
                        },
                        'Invalid section selector.',
                    );
                }
                selected.add(number);
            }
            return Array.from(selected).sort((a, b) => a - b);
        }

        function applySourceSectionSelection() {
            const selector = (document.getElementById('sourceDraftSectionSelector')?.value || '').trim();

            let sourceDraft = null;
            try {
                sourceDraft = readSourceDraftEditorValue();
            } catch (err) {
                showApiError(err, 'Invalid raw draft JSON.');
                return;
            }

            let selectedRanks = [];
            try {
                selectedRanks = parseSectionSelectorInput(selector);
            } catch (err) {
                showApiError(err, 'Invalid section selector.');
                return;
            }

            if (!selectedRanks.length) {
                showApiError(
                    {
                        code: 'SECTION_SELECTOR_REQUIRED',
                        message: 'Enter at least one section number or range.',
                        hint: 'Example: 1-4, 5, 10',
                    },
                    'Section selector is required.',
                );
                return;
            }

            const sections = Array.isArray(sourceDraft?.sections) ? sourceDraft.sections : [];
            const selectedSet = new Set(selectedRanks);
            const filteredSections = sections.filter((section, idx) => {
                const rankCandidate = Number.parseInt(String(section?.rank ?? idx + 1), 10);
                const rank = Number.isInteger(rankCandidate) && rankCandidate > 0 ? rankCandidate : idx + 1;
                return selectedSet.has(rank);
            });

            if (!filteredSections.length) {
                showApiError(
                    {
                        code: 'SECTION_SELECTION_EMPTY',
                        message: 'No matching sections found in current draft.',
                        hint: 'Check the Draft Sections numbers and try again.',
                    },
                    'No matching sections found.',
                );
                return;
            }

            const filteredDraft = {
                ...sourceDraft,
                sections: filteredSections,
            };

            latestSourceDraft = filteredDraft;
            latestHandledScript = null;
            setSourceDraftEditorValue(filteredDraft);

            const sourceMeta = latestSourceMeta || {
                source: filteredDraft.source || 'source_1',
                title: filteredDraft.title || filteredDraft.topic_query || 'Wikipedia result',
                url: filteredDraft.source_url || '',
                lang: filteredDraft.language || (document.getElementById('scriptLang')?.value || 'vi-VN'),
                fetched_at: filteredDraft.fetched_at || null,
                confidence: null,
                cache_hit: true,
            };
            const draftStats = computeDraftStatsFromSections(filteredDraft);
            const warnings = [
                ...((Array.isArray(latestSourceWarnings) ? latestSourceWarnings : []).filter(item => !String(item || '').startsWith('Filtered to sections:'))),
                `Filtered to sections: ${selectedRanks.join(', ')}.`,
            ];

            renderSourceDraftResult({
                source_meta: sourceMeta,
                source_draft: filteredDraft,
                draft_stats: draftStats,
                warnings,
                section_selector: selector,
            });
            renderSourceHandleResult({
                handling_mode: 'sections_loaded',
                script: null,
                llm_meta: null,
                warnings: [],
                meta: {
                    section_count: draftStats.section_count,
                    raw_block_count: 0,
                    output_block_count: 0,
                },
            });
            showStatus(`Loaded ${filteredSections.length} selected sections into draft.`, 'done');
        }

        function applyHandledScriptToEditor() {
            if (!latestHandledScript || !Array.isArray(latestHandledScript.blocks) || !latestHandledScript.blocks.length) {
                showApiError(
                    {
                        code: 'NO_HANDLED_SCRIPT',
                        message: 'No handled script available to apply.',
                        hint: 'Run handling with mode "Raw Draft -> Basic Script" or "LLM -> Video Script JSON".',
                    },
                    'No handled script available.',
                );
                return;
            }

            const current = getScript();
            parseScriptData(
                {
                    ...current,
                    language: latestHandledScript.language || current.language || 'vi-VN',
                    blocks: latestHandledScript.blocks,
                },
                false,
            );
            showStatus(`Applied handled script (${latestHandledScript.blocks.length} blocks).`, 'done');
        }

        function buildCrawlScriptName() {
            const topic = String(document.getElementById('sourceTopic')?.value || '').trim();
            const sourceTitle = String(latestSourceMeta?.title || latestSourceDraft?.title || '').trim();
            const base = (topic || sourceTitle || 'crawl_script')
                .replace(/[^\w\s-]+/g, ' ')
                .trim()
                .replace(/\s+/g, '_');
            const stamp = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14);
            return `${base || 'crawl_script'}_${stamp}`;
        }

        async function saveHandledScriptFromCrawl(scriptName = '') {
            if (!latestHandledScript || !Array.isArray(latestHandledScript.blocks) || !latestHandledScript.blocks.length) {
                throw normalizeApiError(
                    {
                        code: 'NO_HANDLED_SCRIPT',
                        message: 'No handled script available to save.',
                    },
                    'No handled script available.',
                );
            }
            const finalName = String(scriptName || document.getElementById('scriptName')?.value || '').trim() || buildCrawlScriptName();
            await fetchApi(
                `/api/scripts/${encodeURIComponent(finalName)}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(latestHandledScript),
                },
                'Failed to save handled script.',
            );
            activeScriptName = finalName;
            document.getElementById('scriptName').value = finalName;
            rememberRecentScript(finalName);
            await loadScriptList(finalName);
            return finalName;
        }

        async function saveSourceDraftToProject() {
            let sourceDraft = null;
            try {
                sourceDraft = readSourceDraftEditorValue();
            } catch (err) {
                showApiError(err, 'Invalid raw draft JSON.');
                return;
            }

            showStatus('Saving draft to project folder...', 'processing');
            clearErrorPanel();
            try {
                const data = await fetchApi(
                    '/api/content/drafts/save',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            source_draft: sourceDraft,
                            language: (document.getElementById('scriptLang')?.value || 'vi-VN').trim() || 'vi-VN',
                        }),
                    },
                    'Failed to save draft file.',
                );
                const savedPath = String(data?.saved_path || '').trim();
                const filename = String(data?.filename || '').trim();
                const status = document.getElementById('sourceHandleStatus');
                if (status) {
                    status.textContent = savedPath
                        ? `Draft saved to ${savedPath}`
                        : `Draft saved as ${filename || 'draft.json'}`;
                }
                showStatus(
                    savedPath ? `Saved draft: ${savedPath}` : `Saved draft: ${filename || 'draft.json'}`,
                    'done',
                );
                await refreshSavedDraftList({ silent: true, selectFilename: filename || null });
            } catch (err) {
                showApiError(err, 'Failed to save draft file.');
            }
        }

        function computeDraftStatsFromSections(sourceDraft) {
            const sections = Array.isArray(sourceDraft?.sections) ? sourceDraft.sections : [];
            let sentenceCount = 0;
            let wordCount = 0;
            let charCount = 0;
            for (const section of sections) {
                const text = String(section?.text || '').trim();
                if (!text) continue;
                charCount += text.length;
                const words = text.split(/\s+/).filter(Boolean);
                wordCount += words.length;
                sentenceCount += text.split(/[.!?]+/).map(s => s.trim()).filter(Boolean).length;
            }
            return {
                section_count: sections.length,
                sentence_count: sentenceCount,
                word_count: wordCount,
                char_count: charCount,
            };
        }

        function renderSavedDraftOptions(items, selectedFilename = '') {
            const selectEl = document.getElementById('sourceSavedDraftSelect');
            if (!selectEl) return;
            const drafts = Array.isArray(items) ? items : [];
            latestSavedDraftItems = drafts;
            if (!drafts.length) {
                selectEl.innerHTML = '<option value="">No saved drafts yet</option>';
                return;
            }
            selectEl.innerHTML = [
                '<option value="">Select saved draft...</option>',
                ...drafts.map(item => {
                    const filename = String(item?.filename || '').trim();
                    if (!filename) return '';
                    const createdText = formatTimestamp(item?.created_at, item?.created_ts);
                    return `<option value="${escapeHtml(filename)}">${escapeHtml(filename)} · ${escapeHtml(createdText)}</option>`;
                }),
            ].join('');
            if (selectedFilename) {
                const exists = drafts.some(item => String(item?.filename || '') === selectedFilename);
                if (exists) {
                    selectEl.value = selectedFilename;
                }
            }
        }

        async function refreshSavedDraftList(options = {}) {
            const silent = Boolean(options?.silent);
            const selectFilename = String(options?.selectFilename || '').trim();
            if (!silent) showStatus('Loading saved drafts...', 'processing');
            try {
                const data = await fetchApi(
                    '/api/content/drafts?limit=200',
                    {},
                    'Failed to load saved drafts.',
                );
                const drafts = Array.isArray(data?.drafts) ? data.drafts : [];
                renderSavedDraftOptions(drafts, selectFilename);
                if (!silent) showStatus(`Loaded ${drafts.length} saved drafts.`, 'done');
            } catch (err) {
                if (!silent) showApiError(err, 'Failed to load saved drafts.');
            }
        }

        async function loadSavedDraftToEditor() {
            const selectEl = document.getElementById('sourceSavedDraftSelect');
            const filename = String(selectEl?.value || '').trim();
            if (!filename) {
                showApiError(
                    {
                        code: 'DRAFT_NAME_REQUIRED',
                        message: 'Select a saved draft file first.',
                    },
                    'Select a saved draft first.',
                );
                return;
            }

            showStatus(`Loading draft ${filename}...`, 'processing');
            clearErrorPanel();
            try {
                const data = await fetchApi(
                    `/api/content/drafts/${encodeURIComponent(filename)}`,
                    {},
                    'Failed to load saved draft.',
                );
                const sourceDraft = data?.source_draft;
                if (!_hasSourceDraftSections(sourceDraft)) {
                    throw normalizeApiError(
                        {
                            code: 'INVALID_SOURCE_DRAFT',
                            message: 'Saved draft does not contain sections.',
                        },
                        'Failed to load saved draft.',
                    );
                }

                latestSourceDraft = sourceDraft;
                latestHandledScript = null;
                setSourceDraftEditorValue(sourceDraft);

                const draftStats = computeDraftStatsFromSections(sourceDraft);
                const sourceMeta = {
                    source: sourceDraft.source || 'source_1',
                    title: sourceDraft.title || sourceDraft.topic_query || filename,
                    url: sourceDraft.source_url || '',
                    lang: sourceDraft.language || (document.getElementById('scriptLang')?.value || 'vi-VN'),
                    fetched_at: sourceDraft.fetched_at || null,
                    confidence: null,
                    cache_hit: true,
                };
                renderSourceDraftResult({
                    source_meta: sourceMeta,
                    source_draft: sourceDraft,
                    draft_stats: draftStats,
                    warnings: [`Loaded saved draft: ${filename}`],
                });
                renderSourceHandleResult({
                    handling_mode: 'loaded_saved',
                    script: null,
                    llm_meta: null,
                    warnings: [],
                    meta: {
                        section_count: draftStats.section_count,
                        raw_block_count: 0,
                        output_block_count: 0,
                    },
                });
                showStatus(`Loaded saved draft: ${filename}`, 'done');
            } catch (err) {
                showApiError(err, 'Failed to load saved draft.');
            }
        }

        function parseTopicLines(rawValue) {
            const lines = String(rawValue || '').split(/\r?\n/);
            const seen = new Set();
            const topics = [];
            for (const raw of lines) {
                const topic = raw.trim();
                if (!topic) continue;
                const key = topic.toLowerCase();
                if (seen.has(key)) continue;
                seen.add(key);
                topics.push(topic);
            }
            return topics;
        }

        function toOptionalFloat(value) {
            const raw = String(value ?? '').trim();
            if (!raw) return null;
            const parsed = Number(raw);
            if (!Number.isFinite(parsed)) return null;
            return parsed;
        }

        function getFactBankFilterState() {
            return {
                status: (document.getElementById('factBankStatus')?.value || '').trim().toLowerCase(),
                topicQuery: (document.getElementById('factBankTopicId')?.value || '').trim().toLowerCase(),
                query: (document.getElementById('factBankQuery')?.value || '').trim().toLowerCase(),
                minScore: toOptionalFloat(document.getElementById('factBankMinScore')?.value),
            };
        }

        function parseTopicFilterTokens(topicQuery) {
            return String(topicQuery || '')
                .split(',')
                .map(part => part.trim().toLowerCase())
                .filter(Boolean);
        }

        function filterFactBankItems(items, filters) {
            const facts = Array.isArray(items) ? items : [];
            const status = String(filters?.status || '').trim().toLowerCase();
            const query = String(filters?.query || '').trim().toLowerCase();
            const topicTokens = parseTopicFilterTokens(filters?.topicQuery || '');
            const minScore = filters?.minScore;

            return facts.filter(fact => {
                if (status && String(fact?.status || '').toLowerCase() !== status) {
                    return false;
                }

                if (Number.isFinite(minScore) && Number(fact?.score || 0) < Number(minScore)) {
                    return false;
                }

                const topicId = String(fact?.topic_id || '').toLowerCase();
                const topicLabel = String(fact?.topic_label || '').toLowerCase();
                if (topicTokens.length) {
                    const matchesTopicToken = topicTokens.some(token =>
                        topicId.includes(token) || topicLabel.includes(token)
                    );
                    if (!matchesTopicToken) return false;
                }

                if (query) {
                    const haystack = [
                        fact?.fact_text || '',
                        fact?.topic_label || '',
                        fact?.topic_id || '',
                        fact?.hook_text || '',
                    ].join(' ').toLowerCase();
                    if (!haystack.includes(query)) return false;
                }
                return true;
            });
        }

        function factBankStatusCounts(items) {
            const counts = { unused: 0, used: 0, archived: 0 };
            for (const fact of Array.isArray(items) ? items : []) {
                const status = String(fact?.status || '').toLowerCase();
                if (status in counts) counts[status] += 1;
            }
            return counts;
        }

        function renderFactBankTags({ showingCount, totalCount, counts, filters }) {
            const tagsEl = document.getElementById('factBankTags');
            if (!tagsEl) return;
            const activeTags = [];
            if (filters?.status) activeTags.push(`status:${filters.status}`);
            if (filters?.topicQuery) activeTags.push(`topic:${filters.topicQuery}`);
            if (filters?.query) activeTags.push(`search:${filters.query}`);
            if (Number.isFinite(filters?.minScore)) activeTags.push(`min:${Number(filters.minScore).toFixed(2)}`);

            tagsEl.innerHTML = [
                `<span class="fact-bank-tag">showing ${showingCount}/${totalCount}</span>`,
                `<span class="fact-bank-tag">unused ${counts.unused}</span>`,
                `<span class="fact-bank-tag">used ${counts.used}</span>`,
                `<span class="fact-bank-tag">archived ${counts.archived}</span>`,
                ...activeTags.map(tag => `<span class="fact-bank-tag">${escapeHtml(tag)}</span>`),
            ].join('');
        }

        function renderFactBankList(items, meta = {}) {
            const listEl = document.getElementById('factBankList');
            const statsEl = document.getElementById('factBankStats');
            if (!listEl || !statsEl) return;

            const facts = Array.isArray(items) ? items : [];
            latestFactBankItems = facts;
            if (Number.isFinite(Number(meta?.serverTotal))) {
                latestFactBankTotal = Number(meta.serverTotal);
            } else {
                latestFactBankTotal = facts.length;
            }
            const counts = factBankStatusCounts(facts);
            const filters = meta?.filters || getFactBankFilterState();

            if (!facts.length) {
                listEl.innerHTML = '<p class="fact-bank-empty">No facts matched current filters.</p>';
                statsEl.textContent = `Showing 0/${latestFactBankTotal} facts.`;
                renderFactBankTags({
                    showingCount: 0,
                    totalCount: latestFactBankTotal,
                    counts,
                    filters,
                });
                return;
            }

            listEl.innerHTML = facts.map(fact => {
                const status = String(fact.status || 'unused').toLowerCase();
                const score = Number.isFinite(Number(fact.score)) ? Number(fact.score).toFixed(3) : '-';
                const topic = String(fact.topic_label || fact.topic_id || 'Unknown topic');
                const text = String(fact.fact_text || '').trim();
                const usedAt = fact.used_at ? formatTimestamp(fact.used_at) : null;
                const canMarkUsed = status !== 'used' && status !== 'archived';
                const factId = String(fact.fact_id || '');
                const encodedFactId = encodeURIComponent(factId);
                return `
                    <article class="fact-bank-item ${status === 'used' ? 'is-used' : ''} ${status === 'archived' ? 'is-archived' : ''}">
                        <div class="fact-bank-item-top">
                            <span class="fact-bank-topic">${escapeHtml(topic)}</span>
                            <span class="fact-bank-score">score ${escapeHtml(score)} · ${escapeHtml(status)}</span>
                        </div>
                        <p class="fact-bank-text">${escapeHtml(text || '(empty fact)')}</p>
                        <p class="fact-bank-meta">fact_id: ${escapeHtml(factId || '-')}</p>
                        ${usedAt ? `<p class="fact-bank-meta">used_at: ${escapeHtml(usedAt)}</p>` : ''}
                        <div class="fact-bank-actions">
                            ${canMarkUsed ? `<button class="btn-secondary btn-small" type="button" onclick="markFactUsedByEncoded('${encodedFactId}')">Mark Used</button>` : ''}
                        </div>
                    </article>
                `;
            }).join('');
            statsEl.textContent = `Showing ${facts.length}/${latestFactBankTotal} facts.`;
            renderFactBankTags({
                showingCount: facts.length,
                totalCount: latestFactBankTotal,
                counts,
                filters,
            });
        }

        function applyFactBankFiltersAndRender(options = {}) {
            const filters = getFactBankFilterState();
            const filtered = filterFactBankItems(latestFactBankServerItems, filters);
            renderFactBankList(filtered, {
                serverTotal: latestFactBankTotal,
                filters,
            });
            if (options.toast) {
                showStatus('Fact bank filters applied.', 'done');
            }
        }

        function scheduleFactBankFilterApply(delayMs = 160) {
            if (factBankFilterDebounceTimer) clearTimeout(factBankFilterDebounceTimer);
            factBankFilterDebounceTimer = setTimeout(() => {
                if (!latestFactBankServerItems.length) {
                    void loadFactBank({ silent: true });
                    return;
                }
                applyFactBankFiltersAndRender();
            }, delayMs);
        }

        function setFactBankFilterBindings() {
            const ids = ['factBankStatus', 'factBankTopicId', 'factBankQuery', 'factBankMinScore'];
            for (const id of ids) {
                const el = document.getElementById(id);
                if (!el || el.dataset.bound === '1') continue;
                const eventName = el.tagName === 'SELECT' ? 'change' : 'input';
                el.addEventListener(eventName, () => scheduleFactBankFilterApply(180));
                el.dataset.bound = '1';
            }
        }

        function clearFactBankFilters() {
            const statusEl = document.getElementById('factBankStatus');
            const topicEl = document.getElementById('factBankTopicId');
            const queryEl = document.getElementById('factBankQuery');
            const scoreEl = document.getElementById('factBankMinScore');
            if (statusEl) statusEl.value = 'unused';
            if (topicEl) topicEl.value = '';
            if (queryEl) queryEl.value = '';
            if (scoreEl) scoreEl.value = '';
            applyFactBankFiltersAndRender({ toast: true });
        }

        async function loadFactBank(options = {}) {
            const force = Boolean(options?.force);
            const silent = Boolean(options?.silent);
            if (!force && latestFactBankServerItems.length) {
                applyFactBankFiltersAndRender();
                return;
            }

            if (!silent) showStatus('Syncing fact bank...', 'processing');
            try {
                const params = new URLSearchParams();
                params.set('limit', '500');
                params.set('offset', '0');
                const data = await fetchApi(
                    `/api/content/bank/facts?${params.toString()}`,
                    {},
                    'Failed to load fact bank.',
                );
                latestFactBankServerItems = Array.isArray(data?.facts) ? data.facts : [];
                latestFactBankTotal = Number.isFinite(Number(data?.total))
                    ? Number(data.total)
                    : latestFactBankServerItems.length;
                applyFactBankFiltersAndRender();
                if (!silent) showStatus('Fact bank synced.', 'done');
            } catch (err) {
                showApiError(err, 'Failed to load fact bank.');
            }
        }

        async function markFactUsedByEncoded(encodedFactId) {
            const factId = decodeURIComponent(encodedFactId);
            await markFactUsed(factId);
        }

        async function markFactUsed(factId) {
            const cleanFactId = String(factId || '').trim();
            if (!cleanFactId) return;
            try {
                await fetchApi(
                    '/api/content/bank/facts/mark-used',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ fact_ids: [cleanFactId] }),
                    },
                    'Failed to mark fact as used.',
                );
                showStatus('Fact marked as used.', 'done');
                await loadFactBank({ force: true, silent: true });
            } catch (err) {
                showApiError(err, 'Failed to mark fact as used.');
            }
        }

        async function ingestTopicPool() {
            const topics = parseTopicLines(document.getElementById('topicPoolInput')?.value);
            const maxTopics = Number(document.getElementById('topicPoolMaxTopics')?.value || '20');
            const factsPerTopic = Number(document.getElementById('topicPoolFactsPerTopic')?.value || '8');
            const nearDupThreshold = Number(document.getElementById('topicPoolNearDup')?.value || '0.88');
            const promptOverride = (document.getElementById('topicPoolPromptOverride')?.value || '').trim();
            const language = (document.getElementById('scriptLang')?.value || 'en-US').trim();
            const statusEl = document.getElementById('topicPoolStatus');

            if (!topics.length) {
                showApiError(
                    {
                        code: 'TOPICS_REQUIRED',
                        message: 'Enter at least one topic (one per line).',
                    },
                    'Topics are required.',
                );
                return;
            }
            if (!Number.isInteger(maxTopics) || maxTopics < 1 || maxTopics > 50) {
                showApiError(
                    {
                        code: 'INVALID_MAX_TOPICS',
                        message: 'Max topics must be an integer between 1 and 50.',
                    },
                    'Invalid max topics value.',
                );
                return;
            }
            if (!Number.isInteger(factsPerTopic) || factsPerTopic < 1 || factsPerTopic > 20) {
                showApiError(
                    {
                        code: 'INVALID_FACTS_PER_TOPIC',
                        message: 'Facts/topic must be an integer between 1 and 20.',
                    },
                    'Invalid facts/topic value.',
                );
                return;
            }
            if (!Number.isFinite(nearDupThreshold) || nearDupThreshold < 0.6 || nearDupThreshold > 0.99) {
                showApiError(
                    {
                        code: 'INVALID_NEAR_DUP_THRESHOLD',
                        message: 'Near-dup threshold must be between 0.6 and 0.99.',
                    },
                    'Invalid near-dup threshold.',
                );
                return;
            }
            if (promptOverride.length > 6000) {
                showApiError(
                    {
                        code: 'INVALID_PROMPT_OVERRIDE',
                        message: 'Fact prompt override is too long.',
                        hint: 'Keep prompt override under 6000 characters.',
                    },
                    'Invalid prompt override.',
                );
                return;
            }

            const payload = {
                topics,
                language,
                max_topics: maxTopics,
                facts_per_topic_target: factsPerTopic,
                near_dup_threshold: nearDupThreshold,
            };
            if (promptOverride) payload.fact_prompt_override = promptOverride;

            if (statusEl) statusEl.textContent = 'Ingesting topics and extracting facts...';
            showStatus('Ingesting topic pool...', 'processing');
            try {
                const data = await fetchApi(
                    '/api/content/bank/ingest',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    },
                    'Failed to ingest topic pool.',
                );
                const accepted = Number(data?.accepted_topics || 0);
                const createdTopics = Number(data?.created_topics || 0);
                const savedFacts = Number(data?.saved_facts || 0);
                const dedupedFacts = Number(data?.deduped_facts || 0);
                if (statusEl) {
                    statusEl.textContent = `Accepted ${accepted} topics · created ${createdTopics} topics · saved ${savedFacts} facts · deduped ${dedupedFacts}.`;
                }
                showStatus('Topic pool ingested.', 'done');
                await loadFactBank({ force: true, silent: true });
            } catch (err) {
                if (statusEl) statusEl.textContent = 'Topic ingest failed.';
                showApiError(err, 'Failed to ingest topic pool.');
            }
        }

        function getComposeTopicIdsFromFilter() {
            const raw = (document.getElementById('factBankTopicId')?.value || '').trim();
            if (!raw) return [];
            const fromVisible = Array.from(
                new Set(
                    (latestFactBankItems || [])
                        .map(item => String(item?.topic_id || '').trim())
                        .filter(Boolean)
                )
            );
            if (fromVisible.length) return fromVisible;
            return raw.split(',').map(part => part.trim()).filter(Boolean);
        }

        async function composeFromFactBank(applyToEditor = true) {
            const selectionMode = (document.getElementById('bankSelectionMode')?.value || 'balanced').trim().toLowerCase();
            const pickTopicsCount = Number(document.getElementById('bankPickTopics')?.value || '3');
            const pickFactsCount = Number(document.getElementById('bankPickFacts')?.value || '3');
            const excludeUsed = Boolean(document.getElementById('bankExcludeUsed')?.checked);
            const language = (document.getElementById('scriptLang')?.value || 'en-US').trim();
            const query = (document.getElementById('factBankQuery')?.value || '').trim();
            const minScore = toOptionalFloat(document.getElementById('factBankMinScore')?.value);
            const composeStatus = document.getElementById('bankComposeStatus');
            const topicIds = getComposeTopicIdsFromFilter();

            if (!BANK_SELECTION_MODES.includes(selectionMode)) {
                showApiError(
                    {
                        code: 'INVALID_SELECTION_MODE',
                        message: 'Selection mode must be top, balanced, or random_weighted.',
                    },
                    'Invalid selection mode.',
                );
                return;
            }
            if (!Number.isInteger(pickTopicsCount) || pickTopicsCount < 1 || pickTopicsCount > 50) {
                showApiError(
                    {
                        code: 'INVALID_PICK_COUNTS',
                        message: 'Pick topics must be an integer between 1 and 50.',
                    },
                    'Invalid pick topics value.',
                );
                return;
            }
            if (!Number.isInteger(pickFactsCount) || pickFactsCount < 1 || pickFactsCount > 100) {
                showApiError(
                    {
                        code: 'INVALID_PICK_COUNTS',
                        message: 'Pick facts must be an integer between 1 and 100.',
                    },
                    'Invalid pick facts value.',
                );
                return;
            }

            const payload = {
                selection_mode: selectionMode,
                pick_topics_count: pickTopicsCount,
                pick_facts_count: pickFactsCount,
                exclude_used: excludeUsed,
                language,
            };
            if (query) payload.query = query;
            if (topicIds.length) payload.topic_ids = topicIds;
            if (Number.isFinite(minScore)) payload.min_score = minScore;

            if (composeStatus) composeStatus.textContent = 'Composing draft from fact bank...';
            showStatus('Composing from fact bank...', 'processing');
            try {
                const data = await fetchApi(
                    '/api/content/bank/compose',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    },
                    'Failed to compose from fact bank.',
                );
                const selectedCount = Number(data?.selection_meta?.selected_facts || 0);
                const selectedTopics = Number(data?.selection_meta?.selected_topics || 0);
                const scriptBlocks = Array.isArray(data?.script?.blocks) ? data.script.blocks : [];

                if (applyToEditor && scriptBlocks.length) {
                    const current = getScript();
                    parseScriptData(
                        {
                            ...current,
                            language: data?.script?.language || current.language || language,
                            blocks: scriptBlocks,
                        },
                        false,
                    );
                }

                if (Array.isArray(data?.selected_facts)) {
                    renderFactBankList(data.selected_facts, {
                        serverTotal: data.selected_facts.length,
                        filters: getFactBankFilterState(),
                    });
                }

                const modeText = applyToEditor ? 'applied to editor' : 'preview only';
                const available = Number(data?.available_after_filter || 0);
                if (composeStatus) {
                    composeStatus.textContent = `Selected ${selectedCount}/${available || selectedCount} facts across ${selectedTopics} topics (${modeText}).`;
                }
                showStatus('Fact-bank compose complete.', 'done');
            } catch (err) {
                if (composeStatus) composeStatus.textContent = 'Compose failed.';
                showApiError(err, 'Failed to compose from fact bank.');
            }
        }

        function computeLineColumnFromIndex(text, index) {
            const safeIndex = Math.max(0, Math.min(Number(index) || 0, text.length));
            const upto = text.slice(0, safeIndex);
            const lines = upto.split('\n');
            const line = lines.length;
            const column = (lines[lines.length - 1] || '').length + 1;
            return { line, column };
        }

        function parseJsonErrorLocation(text, error) {
            const msg = String(error?.message || '');
            const posMatch = msg.match(/position\s+(\d+)/i);
            if (posMatch) {
                const idx = Number(posMatch[1]);
                return { ...computeLineColumnFromIndex(text, idx), index: idx };
            }
            const lineColMatch = msg.match(/line\s+(\d+)\s+column\s+(\d+)/i);
            if (lineColMatch) {
                return {
                    line: Number(lineColMatch[1]),
                    column: Number(lineColMatch[2]),
                    index: null,
                };
            }
            return { line: null, column: null, index: null };
        }

        function parseJsonWithDiagnostics(text) {
            try {
                return { ok: true, value: JSON.parse(text), error: null, location: null };
            } catch (err) {
                return {
                    ok: false,
                    value: null,
                    error: err,
                    location: parseJsonErrorLocation(text, err),
                };
            }
        }

        function clearJsonLintResult() {
            const lint = document.getElementById('jsonLintResult');
            if (!lint) return;
            lint.hidden = true;
            lint.classList.remove('is-valid', 'is-error');
            lint.textContent = '';
        }

        function setJsonLintResult({ isValid, message }) {
            const lint = document.getElementById('jsonLintResult');
            if (!lint) return;
            lint.hidden = false;
            lint.classList.remove('is-valid', 'is-error');
            lint.classList.add(isValid ? 'is-valid' : 'is-error');
            lint.textContent = message;
        }

        function clearJsonDiffPanel() {
            const panel = document.getElementById('jsonDiffPanel');
            const header = document.getElementById('jsonDiffHeader');
            const body = document.getElementById('jsonDiffBody');
            if (!panel || !header || !body) return;
            panel.hidden = true;
            header.textContent = '';
            body.innerHTML = '';
        }

        function buildLineDiffLcs(oldLines, newLines) {
            const m = oldLines.length;
            const n = newLines.length;
            const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));

            for (let i = m - 1; i >= 0; i -= 1) {
                for (let j = n - 1; j >= 0; j -= 1) {
                    if (oldLines[i] === newLines[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
                    else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
                }
            }

            const out = [];
            let i = 0;
            let j = 0;
            while (i < m && j < n) {
                if (oldLines[i] === newLines[j]) {
                    out.push({ type: 'context', text: oldLines[i] });
                    i += 1;
                    j += 1;
                } else if (dp[i + 1][j] >= dp[i][j + 1]) {
                    out.push({ type: 'removed', text: oldLines[i] });
                    i += 1;
                } else {
                    out.push({ type: 'added', text: newLines[j] });
                    j += 1;
                }
            }
            while (i < m) {
                out.push({ type: 'removed', text: oldLines[i] });
                i += 1;
            }
            while (j < n) {
                out.push({ type: 'added', text: newLines[j] });
                j += 1;
            }
            return out;
        }

        function buildLineDiffFast(oldLines, newLines) {
            const out = [];
            let i = 0;
            let j = 0;
            while (i < oldLines.length && j < newLines.length) {
                if (oldLines[i] === newLines[j]) {
                    out.push({ type: 'context', text: oldLines[i] });
                    i += 1;
                    j += 1;
                    continue;
                }
                if (i + 1 < oldLines.length && oldLines[i + 1] === newLines[j]) {
                    out.push({ type: 'removed', text: oldLines[i] });
                    i += 1;
                    continue;
                }
                if (j + 1 < newLines.length && oldLines[i] === newLines[j + 1]) {
                    out.push({ type: 'added', text: newLines[j] });
                    j += 1;
                    continue;
                }
                out.push({ type: 'removed', text: oldLines[i] });
                out.push({ type: 'added', text: newLines[j] });
                i += 1;
                j += 1;
            }
            while (i < oldLines.length) {
                out.push({ type: 'removed', text: oldLines[i] });
                i += 1;
            }
            while (j < newLines.length) {
                out.push({ type: 'added', text: newLines[j] });
                j += 1;
            }
            return out;
        }

        function buildLineDiff(oldText, newText) {
            const oldLines = String(oldText || '').split('\n');
            const newLines = String(newText || '').split('\n');
            const complexity = oldLines.length * newLines.length;
            if (complexity <= 250000) {
                return buildLineDiffLcs(oldLines, newLines);
            }
            return buildLineDiffFast(oldLines, newLines);
        }

        function renderJsonDiff(savedName, diffLines) {
            const panel = document.getElementById('jsonDiffPanel');
            const header = document.getElementById('jsonDiffHeader');
            const body = document.getElementById('jsonDiffBody');
            if (!panel || !header || !body) return;

            const added = diffLines.filter(line => line.type === 'added').length;
            const removed = diffLines.filter(line => line.type === 'removed').length;
            const total = diffLines.length;
            const capped = diffLines.slice(0, 1200);
            const truncated = total > capped.length;

            header.textContent = `Diff vs saved "${savedName}" · +${added} / -${removed}${truncated ? ' · truncated' : ''}`;
            body.innerHTML = capped.map(line => {
                const marker = line.type === 'added' ? '+' : line.type === 'removed' ? '-' : ' ';
                return `<div class="diff-line ${line.type}"><span class="diff-marker">${marker}</span><span>${escapeHtml(line.text)}</span></div>`;
            }).join('') || '<div class="diff-line context"><span class="diff-marker"> </span><span>No differences.</span></div>';
            panel.hidden = false;
        }

        function formatJsonEditor() {
            const area = document.getElementById('jsonArea');
            if (!area) return;
            const parsed = parseJsonWithDiagnostics(area.value);
            if (!parsed.ok) {
                const { line, column } = parsed.location || {};
                setJsonLintResult({
                    isValid: false,
                    message: `Invalid JSON${line ? ` at line ${line}, column ${column}` : ''}: ${parsed.error.message}`,
                });
                showApiError(
                    {
                        code: 'INVALID_JSON',
                        message: 'Cannot format invalid JSON.',
                        details: parsed.error.message,
                        hint: line ? `Fix syntax near line ${line}, column ${column}.` : 'Fix JSON syntax and try again.',
                    },
                    'Cannot format invalid JSON.',
                );
                return;
            }

            area.value = JSON.stringify(parsed.value, null, 2);
            setJsonLintResult({ isValid: true, message: 'JSON is valid and formatted.' });
            showStatus('JSON formatted.', 'done');
            markDirty();
        }

        function lintJsonEditor() {
            const area = document.getElementById('jsonArea');
            if (!area) return;
            const parsed = parseJsonWithDiagnostics(area.value);
            if (parsed.ok) {
                setJsonLintResult({ isValid: true, message: 'JSON is valid.' });
                showStatus('JSON lint passed.', 'done');
                return;
            }
            const { line, column } = parsed.location || {};
            const message = `Invalid JSON${line ? ` at line ${line}, column ${column}` : ''}: ${parsed.error.message}`;
            setJsonLintResult({ isValid: false, message });
            showStatus('JSON lint failed.', 'error');
        }

        async function diffJsonVsSaved() {
            const name = document.getElementById('scriptName').value.trim();
            if (!name) {
                showApiError(
                    {
                        code: 'SCRIPT_NAME_REQUIRED',
                        message: 'Script name is required for diff.',
                        hint: 'Enter script name, save/load it, then run diff.',
                    },
                    'Script name is required for diff.',
                );
                return;
            }

            const area = document.getElementById('jsonArea');
            if (!area) return;
            const currentRaw = area.value;
            const parsedCurrent = parseJsonWithDiagnostics(currentRaw);
            const currentText = parsedCurrent.ok ? JSON.stringify(parsedCurrent.value, null, 2) : currentRaw;

            if (!parsedCurrent.ok) {
                const { line, column } = parsedCurrent.location || {};
                setJsonLintResult({
                    isValid: false,
                    message: `Current editor JSON invalid${line ? ` at line ${line}, column ${column}` : ''}. Diff uses raw text.`,
                });
            }

            try {
                const savedData = await fetchApi(
                    `/api/scripts/${encodeURIComponent(name)}`,
                    {},
                    'Failed to load saved script for diff.',
                );
                const savedText = JSON.stringify(savedData, null, 2);
                const diffLines = buildLineDiff(savedText, currentText);
                renderJsonDiff(name, diffLines);
                showStatus('Diff generated.', 'done');
            } catch (err) {
                showApiError(err, 'Failed to load saved script for diff.');
            }
        }

        function sanitizeToken(value) {
            const safe = String(value || '')
                .trim()
                .replace(/[^a-zA-Z0-9_-]+/g, '_')
                .replace(/^_+|_+$/g, '');
            return safe || 'untitled';
        }

        function buildCompareOutputName(baseScriptName, preset, compareGroup) {
            const base = sanitizeToken(baseScriptName);
            const safePreset = sanitizeToken(preset).toLowerCase();
            return `${base}__cmp-${compareGroup}__${safePreset}`;
        }

        function formatBytes(bytes) {
            const size = Number(bytes);
            if (!Number.isFinite(size) || size < 0) return '-';
            if (size < 1024) return `${size} B`;
            if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
            if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
            return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
        }

        function formatTimestamp(isoText, createdTs = null) {
            let dateObj = null;
            if (isoText) {
                const parsed = new Date(isoText);
                if (!Number.isNaN(parsed.getTime())) dateObj = parsed;
            }
            if (!dateObj && Number.isFinite(Number(createdTs)) && Number(createdTs) > 0) {
                dateObj = new Date(Number(createdTs) * 1000);
            }
            if (!dateObj) return '-';
            return dateObj.toLocaleString(undefined, {
                year: 'numeric',
                month: 'short',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
            });
        }

        function normalizeOutputItem(raw) {
            if (typeof raw === 'string') {
                const filename = raw;
                const url = `/output/${encodeURIComponent(filename)}`;
                return {
                    file: filename,
                    sizeBytes: null,
                    createdAt: null,
                    createdTs: null,
                    url,
                    downloadUrl: url,
                };
            }
            if (!raw || typeof raw !== 'object') return null;

            const filename = raw.filename || raw.name || '';
            if (!filename) return null;
            const encodedUrl = `/output/${encodeURIComponent(filename)}`;
            return {
                file: filename,
                sizeBytes: Number.isFinite(Number(raw.size_bytes)) ? Number(raw.size_bytes) : null,
                createdAt: raw.created_at || null,
                createdTs: Number.isFinite(Number(raw.created_ts)) ? Number(raw.created_ts) : null,
                url: raw.url || encodedUrl,
                downloadUrl: raw.download_url || encodedUrl,
                audioFilename: raw.audio_filename || null,
                audioUrl: raw.audio_url || null,
                audioDownloadUrl: raw.audio_download_url || null,
            };
        }

        function parseOutputMetadata(filename) {
            const stem = String(filename || '').replace(/\.mp4$/i, '');
            const match = stem.match(/^(.*)__cmp-(\d{6,})__([a-z0-9_-]+)$/i);
            if (!match) {
                return { filename, isCompare: false, baseName: stem, compareGroup: null, preset: null };
            }
            return {
                filename,
                isCompare: true,
                baseName: match[1],
                compareGroup: match[2],
                preset: match[3].toLowerCase(),
            };
        }

        function renderCompareGroup(outputsWithMeta) {
            const panel = document.getElementById('compareGroupPanel');
            const title = document.getElementById('compareGroupTitle');
            const list = document.getElementById('compareGroupList');
            if (!panel || !title || !list) return;

            const groups = new Map();
            for (const item of outputsWithMeta) {
                if (!item.meta.isCompare || !item.meta.compareGroup) continue;
                const key = item.meta.compareGroup;
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(item);
            }

            if (!groups.size) {
                panel.hidden = true;
                title.textContent = '';
                list.innerHTML = '';
                return;
            }

            const latestGroup = [...groups.keys()]
                .sort((a, b) => Number(b) - Number(a))[0];
            const latestItems = groups.get(latestGroup) || [];
            const presetRank = (preset) => {
                const idx = COMPARE_PRESET_ORDER.indexOf(preset);
                return idx === -1 ? 999 : idx;
            };
            latestItems.sort((a, b) => presetRank(a.meta.preset) - presetRank(b.meta.preset));

            title.textContent = `Latest Compare Group ${latestGroup}`;
            list.innerHTML = latestItems.map(item => `
                <div class="compare-group-item">
                    <div>
                        <strong>${escapeHtml(item.meta.preset || 'preset')}</strong>
                        <span class="output-meta">
                            <span class="output-badge group">cmp ${escapeHtml(item.meta.compareGroup || '')}</span>
                        </span>
                    </div>
                    <button class="btn-secondary btn-small" onclick="previewVideoByEncoded('${encodeURIComponent(item.file)}')">Preview</button>
                </div>
            `).join('');
            panel.hidden = false;
        }
