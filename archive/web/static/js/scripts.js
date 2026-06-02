        function renderScriptList(activeName = activeScriptName) {
            const list = document.getElementById('scriptList');
            if (!list) return;

            const query = (document.getElementById('scriptSearch')?.value || '').trim().toLowerCase();
            const filtered = scriptNames.filter(name => name.toLowerCase().includes(query));
            const recentPool = recentScriptNames.filter(name => filtered.includes(name));
            const remaining = filtered.filter(name => !recentPool.includes(name));
            const visible = [...recentPool, ...remaining].slice(0, SCRIPT_RECENT_LIMIT);
            const showAllBtn = document.getElementById('showAllScriptsBtn');

            if (!visible.length) {
                list.innerHTML = '<p class="output-empty">No matching scripts</p>';
                if (showAllBtn) showAllBtn.hidden = true;
                renderScriptPickerList(filtered, activeName);
                return;
            }

            list.innerHTML = visible.map(name => {
                const active = name === activeName ? 'active' : '';
                const encoded = encodeURIComponent(name);
                const selected = name === activeName ? 'true' : 'false';
                return `<button type="button" class="script-chip ${active}" role="option" aria-selected="${selected}" onclick="loadScriptByEncoded('${encoded}')">${escapeHtml(name)}</button>`;
            }).join('');
            if (showAllBtn) {
                showAllBtn.hidden = filtered.length <= SCRIPT_RECENT_LIMIT;
                showAllBtn.textContent = 'Show All';
            }
            renderScriptPickerList(filtered, activeName);
        }

        function renderScriptPickerList(items, activeName = activeScriptName) {
            const list = document.getElementById('scriptPickerList');
            if (!list) return;
            const scripts = Array.isArray(items) ? items : [];
            if (!scripts.length) {
                list.innerHTML = '<p class="output-empty">No scripts yet.</p>';
                return;
            }
            list.innerHTML = scripts.map(name => {
                const active = name === activeName;
                const encoded = encodeURIComponent(name);
                return `
                    <div class="output-picker-item">
                        <div class="output-picker-main">
                            <div class="output-picker-name">${escapeHtml(name)}</div>
                            <div class="output-picker-meta">${active ? 'Current script' : 'Saved script'}</div>
                        </div>
                        <div class="output-picker-actions">
                            <button class="btn-primary btn-small" type="button" onclick="selectScriptByEncoded('${encoded}')">Load</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function loadScriptByEncoded(encodedName) {
            loadScript(decodeURIComponent(encodedName));
        }

        function openScriptPicker() {
            const backdrop = document.getElementById('scriptPickerBackdrop');
            if (!backdrop) return;
            backdrop.hidden = false;
        }

        function closeScriptPicker() {
            const backdrop = document.getElementById('scriptPickerBackdrop');
            if (!backdrop) return;
            backdrop.hidden = true;
        }

        function onScriptPickerBackdropClick(event) {
            if (!event || event.target?.id !== 'scriptPickerBackdrop') return;
            closeScriptPicker();
        }

        function selectScriptByEncoded(encodedName) {
            closeScriptPicker();
            loadScript(decodeURIComponent(encodedName));
        }

        function rememberRecentScript(name) {
            const normalized = String(name || '').trim();
            if (!normalized) return;
            recentScriptNames = [normalized, ...recentScriptNames.filter(item => item !== normalized)].slice(0, 12);
        }

        async function loadScriptList(activeName = activeScriptName) {
            try {
                const data = await fetchApi('/api/scripts', {}, 'Failed to load script list.');
                scriptNames = Array.isArray(data) ? data : [];
            } catch (err) {
                showApiError(err, 'Failed to load script list.');
            }
            renderScriptList(activeName);
        }

        function updateSelectedHint() {
            const hint = document.getElementById('selectedBlockHint');
            if (!hint) return;
            if (selectedBlockIndex == null || !blocks[selectedBlockIndex]) {
                hint.textContent = 'No block selected. Click a block to enable shortcut actions.';
                return;
            }
            hint.textContent = `Selected: Block ${selectedBlockIndex + 1}. Shortcut: Ctrl/Cmd+Shift+D to duplicate.`;
        }

        function refreshBlockSelection() {
            const container = document.getElementById('blocksContainer');
            if (!container) return;
            container.querySelectorAll('.block').forEach((el, idx) => {
                el.classList.toggle('is-selected', idx === selectedBlockIndex);
            });
            updateSelectedHint();
        }


        function getScript() {
            const script = {
                language: document.getElementById('scriptLang').value,
                blocks: blocks.map(b => {
                    const block = {
                        voice: b.voice || (VOICES[0]?.id || 'NamMinh'),
                        text: b.text || '',
                    };
                    if (b.background) block.background = b.background;
                    if (b.image) block.image = b.image;
                    if (b.voice_rate) block.voice_rate = b.voice_rate;
                    if (b.voice_pitch) block.voice_pitch = b.voice_pitch;
                    if (b.voice_volume) block.voice_volume = b.voice_volume;
                    if (b.subtitle_preset) block.subtitle_preset = b.subtitle_preset;
                    if (b.subtitle_mode) block.subtitle_mode = b.subtitle_mode;
                    if (b.subtitle_alignment_mode) block.subtitle_alignment_mode = b.subtitle_alignment_mode;
                    if (b.subtitle_style) block.subtitle_style = b.subtitle_style;
                    if (b.fact_id) block.fact_id = b.fact_id;
                    if (b.source_topic_id) block.source_topic_id = b.source_topic_id;
                    if (b.source_topic) block.source_topic = b.source_topic;
                    if (b.source_section_id) block.source_section_id = b.source_section_id;
                    if (b.source_section_title) block.source_section_title = b.source_section_title;
                    if (b.source_url) block.source_url = b.source_url;

                    const keywords = parseKeywordText(b.subtitle_keywords);
                    if (keywords.length) block.subtitle_keywords = keywords;
                    return block;
                }),
            };

            const background = emptyToNull(document.getElementById('scriptBackground').value);
            if (background) script.background = background;

            const subtitlePreset = emptyToNull(document.getElementById('scriptSubtitlePreset').value);
            if (subtitlePreset) script.subtitle_preset = subtitlePreset;
            const subtitleAlignmentMode = normalizeAlignmentModeValue(
                document.getElementById('scriptSubtitleAlignmentMode').value
            );
            if (subtitleAlignmentMode) script.subtitle_alignment_mode = subtitleAlignmentMode;

            const voiceRate = emptyToNull(document.getElementById('scriptVoiceRate').value);
            const voicePitch = emptyToNull(document.getElementById('scriptVoicePitch').value);
            const voiceVolume = emptyToNull(document.getElementById('scriptVoiceVolume').value);
            if (voiceRate) script.voice_rate = voiceRate;
            if (voicePitch) script.voice_pitch = voicePitch;
            if (voiceVolume) script.voice_volume = voiceVolume;

            return script;
        }

        function parseScriptData(data, fromSavedScript = false) {
            suppressDirtyTracking = true;
            document.getElementById('scriptLang').value = data.language || 'vi-VN';
            document.getElementById('scriptBackground').value = data.background || '';
            const requestedPreset = data.subtitle_preset || DEFAULT_SUBTITLE_PRESET;
            document.getElementById('scriptSubtitlePreset').value =
                SUBTITLE_PRESETS.includes(requestedPreset) ? requestedPreset : DEFAULT_SUBTITLE_PRESET;
            document.getElementById('scriptSubtitleAlignmentMode').value = normalizeAlignmentModeValue(
                data.subtitle_alignment_mode,
                DEFAULT_SUBTITLE_ALIGNMENT_MODE,
            );
            document.getElementById('scriptVoiceRate').value = data.voice_rate || '';
            document.getElementById('scriptVoicePitch').value = data.voice_pitch || '';
            document.getElementById('scriptVoiceVolume').value = data.voice_volume || '';

            blocks = (data.blocks || []).map(b => ({
                voice: b.voice || (VOICES[0]?.id || 'NamMinh'),
                text: b.text || '',
                voice_rate: b.voice_rate || null,
                voice_pitch: b.voice_pitch || null,
                voice_volume: b.voice_volume || null,
                background: b.background || null,
                image: b.image || null,
                fact_id: b.fact_id || null,
                source_topic_id: b.source_topic_id || null,
                source_topic: b.source_topic || null,
                source_section_id: b.source_section_id || null,
                source_section_title: b.source_section_title || null,
                source_url: b.source_url || null,
                subtitle_preset: b.subtitle_preset || null,
                subtitle_mode: b.subtitle_mode || null,
                subtitle_alignment_mode: SUBTITLE_ALIGNMENT_MODES.includes(String(b.subtitle_alignment_mode || '').toLowerCase())
                    ? String(b.subtitle_alignment_mode).toLowerCase()
                    : null,
                subtitle_style: b.subtitle_style || null,
                subtitle_keywords: keywordTextFromArray(b.subtitle_keywords),
                __collapsed: false,
            }));

            const firstVoice = blocks[0]?.voice || (VOICES[0]?.id || 'NamMinh');
            document.getElementById('scriptDefaultVoice').value = firstVoice;
            selectedBlockIndex = blocks.length ? 0 : null;
            renderBlocks();
            suppressDirtyTracking = false;
            if (currentMode === 'json') {
                const area = document.getElementById('jsonArea');
                if (area) area.value = JSON.stringify(getScript(), null, 2);
                clearJsonLintResult();
                clearJsonDiffPanel();
            }

            if (fromSavedScript) markSaved();
            else markDirty();
        }

        async function loadScript(name) {
            activeScriptName = name;
            document.getElementById('scriptName').value = name;
            rememberRecentScript(name);
            renderScriptList(name);

            try {
                const data = await fetchApi(
                    `/api/scripts/${encodeURIComponent(name)}`,
                    {},
                    'Failed to load script.',
                );
                parseScriptData(data, true);
                setWorkspaceView('editor');
            } catch (err) {
                showApiError(err, 'Failed to load script.');
            }
        }

        async function saveScript() {
            const name = document.getElementById('scriptName').value.trim();
            if (!name) return alert('Enter a script name');
            const script = getScript();

            try {
                await fetchApi(
                    `/api/scripts/${encodeURIComponent(name)}`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(script),
                    },
                    'Failed to save script.',
                );
                activeScriptName = name;
                showStatus('Script saved!', 'done');
                markSaved();
                await loadScriptList(name);
            } catch (err) {
                showApiError(err, 'Failed to save script.');
            }
        }

        async function duplicateScript() {
            const currentName = document.getElementById('scriptName').value.trim();
            if (!currentName) return alert('Enter a script name first');
            const suggested = `${currentName}_copy`;
            const newName = prompt('Duplicate script as:', suggested);
            if (!newName) return;

            const script = getScript();
            try {
                await fetchApi(
                    `/api/scripts/${encodeURIComponent(newName.trim())}`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(script),
                    },
                    'Failed to duplicate script.',
                );
                activeScriptName = newName.trim();
                document.getElementById('scriptName').value = activeScriptName;
                showStatus(`Duplicated to ${activeScriptName}`, 'done');
                markSaved();
                await loadScriptList(activeScriptName);
            } catch (err) {
                showApiError(err, 'Failed to duplicate script.');
            }
        }

        function exportScript() {
            const name = document.getElementById('scriptName').value.trim() || 'script';
            const scriptText = JSON.stringify(getScript(), null, 2);
            const blob = new Blob([scriptText], { type: 'application/json;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${name}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            showStatus('Script exported.', 'done');
        }

        async function deleteCurrentScript() {
            const name = document.getElementById('scriptName').value.trim();
            if (!name) return alert('No script selected');
            if (!confirm(`Delete script "${name}"?`)) return;

            try {
                await fetchApi(
                    `/api/scripts/${encodeURIComponent(name)}`,
                    { method: 'DELETE' },
                    'Failed to delete script.',
                );
                activeScriptName = null;
                document.getElementById('scriptName').value = '';
                parseScriptData({ language: 'vi-VN', blocks: [] }, true);
                await loadScriptList(null);
                showStatus('Script deleted.', 'done');
            } catch (err) {
                showApiError(err, 'Failed to delete script.');
            }
        }


        function setMode(mode) {
            currentMode = mode;
            document.getElementById('visualEditor').style.display = mode === 'visual' ? 'block' : 'none';
            document.getElementById('jsonEditor').style.display = mode === 'json' ? 'block' : 'none';
            document.getElementById('btnVisual').classList.toggle('is-active', mode === 'visual');
            document.getElementById('btnJson').classList.toggle('is-active', mode === 'json');
            document.getElementById('btnVisual').setAttribute('aria-pressed', mode === 'visual' ? 'true' : 'false');
            document.getElementById('btnJson').setAttribute('aria-pressed', mode === 'json' ? 'true' : 'false');

            if (mode === 'json') {
                document.getElementById('jsonArea').value = JSON.stringify(getScript(), null, 2);
                clearJsonLintResult();
                clearJsonDiffPanel();
            }
        }

        function applyJson() {
            const area = document.getElementById('jsonArea');
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
                        message: 'Invalid JSON in editor.',
                        details: parsed.error.message,
                        hint: line ? `Fix syntax near line ${line}, column ${column}.` : 'Fix JSON syntax, then apply again.',
                    },
                    'Invalid JSON in editor.',
                );
                return;
            }
            try {
                parseScriptData(parsed.value, false);
                setJsonLintResult({ isValid: true, message: 'JSON is valid and applied.' });
                clearJsonDiffPanel();
                showStatus('JSON applied!', 'done');
            } catch (e) {
                showApiError(
                    {
                        code: 'INVALID_JSON',
                        message: 'Invalid JSON in editor.',
                        details: e.message,
                        hint: 'Fix JSON syntax, then apply again.',
                    },
                    'Invalid JSON in editor.',
                );
            }
        }

        function importFile(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    parseScriptData(JSON.parse(e.target.result), false);
                    const name = file.name.replace('.json', '');
                    document.getElementById('scriptName').value = name;
                    activeScriptName = null;
                    renderScriptList(null);
                    if (currentMode === 'json') {
                        document.getElementById('jsonArea').value = JSON.stringify(getScript(), null, 2);
                        clearJsonLintResult();
                        clearJsonDiffPanel();
                    }
                    showStatus('Imported: ' + file.name, 'done');
                } catch (err) {
                    showApiError(
                        {
                            code: 'INVALID_JSON_FILE',
                            message: 'Imported file is not valid JSON.',
                            details: err.message,
                            hint: 'Choose a valid JSON script export and retry.',
                        },
                        'Imported file is not valid JSON.',
                    );
                }
            };
            reader.readAsText(file);
            event.target.value = '';
        }
