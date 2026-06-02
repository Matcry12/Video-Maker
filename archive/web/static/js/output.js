        function renderOutputList(files) {
            latestOutputItems = Array.isArray(files) ? files : [];
            const showThumbnails = true;
            const outputsWithMeta = latestOutputItems
                .map(normalizeOutputItem)
                .filter(Boolean)
                .map(item => ({
                    ...item,
                    meta: parseOutputMetadata(item.file),
                }));

            outputsWithMeta.sort((a, b) => {
                const aTs = Number(a.createdTs || 0);
                const bTs = Number(b.createdTs || 0);
                if (aTs !== bTs) return bTs - aTs;
                return b.file.localeCompare(a.file);
            });

            renderCompareGroup(outputsWithMeta);

            const recentItems = outputsWithMeta.slice(0, OUTPUT_RECENT_LIMIT);
            document.getElementById('outputList').innerHTML = recentItems.map(item => {
                const badges = [];
                if (item.meta.preset) badges.push(`<span class="output-badge preset">${escapeHtml(item.meta.preset)}</span>`);
                if (item.meta.compareGroup) badges.push(`<span class="output-badge group">cmp ${escapeHtml(item.meta.compareGroup)}</span>`);
                const sizeText = formatBytes(item.sizeBytes);
                const createdText = formatTimestamp(item.createdAt, item.createdTs);
                const encodedFile = encodeURIComponent(item.file);
                const audioActions = item.audioUrl ? `
                                <a class="btn-link" href="${item.audioUrl}" target="_blank" rel="noopener">MP3</a>
                                <a class="btn-link" href="${item.audioDownloadUrl}" download="${escapeHtml(item.audioFilename || '')}">Download MP3</a>
                ` : '';
                return `
                    <div class="output-item">
                        <div class="output-main">
                            <div class="output-title">
                                <a href="${item.url}" target="_blank" rel="noopener">${escapeHtml(item.file)}</a>
                                <span class="output-meta">${badges.join('')}</span>
                            </div>
                            <p class="output-meta-line">Size: ${escapeHtml(sizeText)} · Created: ${escapeHtml(createdText)}</p>
                            ${showThumbnails ? `<video class="output-thumb" src="${item.url}#t=0.1" preload="metadata" muted></video>` : ''}
                        </div>
                        <div class="output-actions">
                            <div class="output-actions-main">
                                <button class="btn-primary btn-small" onclick="previewVideoByEncoded('${encodedFile}')">Preview</button>
                                <a class="btn-link" href="${item.url}" target="_blank" rel="noopener">Open</a>
                                <a class="btn-link" href="${item.downloadUrl}" download="${escapeHtml(item.file)}">Download</a>
                                ${audioActions}
                            </div>
                            <div class="output-actions-danger">
                                <button class="btn-danger btn-small" onclick="deleteOutputByEncoded('${encodedFile}')">Delete</button>
                            </div>
                        </div>
                    </div>
                `;
            }).join('') || '<p class="output-empty">No videos yet</p>';

            const showAllBtn = document.getElementById('showAllOutputsBtn');
            if (showAllBtn) {
                const totalCount = outputsWithMeta.length;
                showAllBtn.hidden = totalCount <= OUTPUT_RECENT_LIMIT;
                showAllBtn.textContent = 'Show All';
            }

            renderOutputPickerList(outputsWithMeta);
        }

        function renderOutputPickerList(outputsWithMeta) {
            const list = document.getElementById('outputPickerList');
            if (!list) return;

            const items = Array.isArray(outputsWithMeta) ? outputsWithMeta : [];
            if (!items.length) {
                list.innerHTML = '<p class="output-empty">No videos yet.</p>';
                return;
            }

            list.innerHTML = items.map(item => {
                const sizeText = formatBytes(item.sizeBytes);
                const createdText = formatTimestamp(item.createdAt, item.createdTs);
                const encodedFile = encodeURIComponent(item.file);
                const audioActions = item.audioUrl ? `
                            <a class="btn-link" href="${item.audioUrl}" target="_blank" rel="noopener">MP3</a>
                            <a class="btn-link" href="${item.audioDownloadUrl}" download="${escapeHtml(item.audioFilename || '')}">Download MP3</a>
                ` : '';
                return `
                    <div class="output-picker-item">
                        <div class="output-picker-main">
                            <div class="output-picker-name">${escapeHtml(item.file)}</div>
                            <div class="output-picker-meta">Size: ${escapeHtml(sizeText)} · Created: ${escapeHtml(createdText)}</div>
                        </div>
                        <div class="output-picker-actions">
                            <button class="btn-primary btn-small" type="button" onclick="selectOutputByEncoded('${encodedFile}')">Select</button>
                            <a class="btn-link" href="${item.url}" target="_blank" rel="noopener">Open</a>
                            ${audioActions}
                        </div>
                    </div>
                `;
            }).join('');
        }

        function openOutputPicker() {
            const backdrop = document.getElementById('outputPickerBackdrop');
            if (!backdrop) return;
            backdrop.hidden = false;
        }

        function closeOutputPicker() {
            const backdrop = document.getElementById('outputPickerBackdrop');
            if (!backdrop) return;
            backdrop.hidden = true;
        }

        function onOutputPickerBackdropClick(event) {
            if (!event || event.target?.id !== 'outputPickerBackdrop') return;
            closeOutputPicker();
        }

        function selectOutputByEncoded(encodedFilename) {
            const filename = decodeURIComponent(encodedFilename);
            previewVideo(filename);
            closeOutputPicker();
            showStatus(`Selected preview: ${filename}`, 'done');
        }

        function getGenerateContext() {
            const name = document.getElementById('scriptName').value.trim();
            if (!name) {
                alert('Enter a script name');
                return null;
            }
            if (!blocks.length) {
                alert('Add at least one block');
                return null;
            }
            const validation = validateForm();
            if (!validation.valid) {
                showStatus('Fix validation errors before generating.', 'error');
                focusValidationError(0);
                return null;
            }
            return { name, script: getScript() };
        }

        async function startGenerationRequest(scriptName, script, outputName, meta = null, saveScript = true) {
            const payload = {
                script_name: scriptName,
                output_name: outputName || scriptName,
                script,
                save_script: saveScript,
            };
            if (meta && typeof meta === 'object') payload.meta = meta;

            const data = await fetchApi(
                '/api/generate',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                },
                'Failed to start generation.',
            );
            if (!data || !data.job_id) {
                throw normalizeApiError(
                    {
                        code: 'INVALID_JOB_RESPONSE',
                        message: 'Invalid generate response from server.',
                        hint: 'Retry generation. If this continues, restart the backend.',
                    },
                    'Failed to start generation.',
                );
            }
            return data.job_id;
        }

        async function quickRenderPreset(preset) {
            const context = getGenerateContext();
            if (!context) return;
            if (!SUBTITLE_PRESETS.includes(preset)) {
                showApiError(
                    {
                        code: 'PRESET_NOT_FOUND',
                        message: `Subtitle preset "${preset}" is not available.`,
                        hint: 'Choose an available preset from the dropdown.',
                    },
                    'Preset is not available.',
                );
                return;
            }

            const compareGroup = String(Date.now());
            const compareScript = { ...context.script, subtitle_preset: preset };
            const outputName = buildCompareOutputName(context.name, preset, compareGroup);
            showStatus(`Rendering ${preset} preset...`, 'processing');
            renderJobTimeline({
                status: 'loading',
                stage: 'loading',
                progress: `Preparing ${preset} preset...`,
                current_block: 0,
                total_blocks: blocks.length,
            });

            try {
                const jobId = await startGenerationRequest(
                    context.name,
                    compareScript,
                    outputName,
                    { compare_group: compareGroup, preset },
                    false,
                );
                await pollJob(jobId, { previewOnDone: true });
                showStatus(`Rendered ${preset} preset.`, 'done');
            } catch (err) {
                showApiError(err, `Failed to render ${preset} preset.`);
            }
        }

        async function renderPresetComparison() {
            const context = getGenerateContext();
            if (!context) return;

            const presets = COMPARE_PRESET_ORDER.filter(preset => SUBTITLE_PRESETS.includes(preset));
            if (!presets.length) {
                showApiError(
                    {
                        code: 'NO_COMPARE_PRESETS',
                        message: 'No compare presets are available.',
                        hint: 'Add subtitle presets in the profile, then retry.',
                    },
                    'No compare presets available.',
                );
                return;
            }

            const compareGroup = String(Date.now());
            showStatus(`Starting compare group ${compareGroup}...`, 'processing');

            for (const preset of presets) {
                const compareScript = { ...context.script, subtitle_preset: preset };
                const outputName = buildCompareOutputName(context.name, preset, compareGroup);
                showStatus(`Rendering ${preset} preset...`, 'processing');
                renderJobTimeline({
                    status: 'loading',
                    stage: 'loading',
                    progress: `Preparing ${preset} preset...`,
                    current_block: 0,
                    total_blocks: blocks.length,
                });
                try {
                    const jobId = await startGenerationRequest(
                        context.name,
                        compareScript,
                        outputName,
                        { compare_group: compareGroup, preset },
                        false,
                    );
                    await pollJob(jobId, { previewOnDone: false });
                } catch (err) {
                    showApiError(err, `Failed to render ${preset} preset.`);
                    return;
                }
            }

            showStatus(`Compare group ${compareGroup} complete.`, 'done');
            await loadOutputs();
        }

        function escapeHtml(value) {
            return String(value)
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        }

        function voiceOptions(selected) {
            return VOICES.map(v =>
                `<option value="${v.id}" ${v.id === selected ? 'selected' : ''}>${v.label}</option>`
            ).join('');
        }

        function presetOptions(selected, includeInherit = false) {
            const opts = [];
            if (includeInherit) {
                opts.push(`<option value="" ${!selected ? 'selected' : ''}>(Use Script Default)</option>`);
            }
            opts.push(...SUBTITLE_PRESETS.map(name =>
                `<option value="${name}" ${name === selected ? 'selected' : ''}>${name}</option>`
            ));
            return opts.join('');
        }

        function modeOptions(selected) {
            const modes = [
                { value: '', label: '(Use Preset Default)' },
                { value: 'standard', label: 'standard' },
            ];
            return modes.map(m =>
                `<option value="${m.value}" ${m.value === (selected || '') ? 'selected' : ''}>${m.label}</option>`
            ).join('');
        }

        function normalizeAlignmentModeValue(value, fallback = DEFAULT_SUBTITLE_ALIGNMENT_MODE) {
            const candidate = String(value || '').trim().toLowerCase();
            if (SUBTITLE_ALIGNMENT_MODES.includes(candidate)) return candidate;
            return fallback;
        }

        function alignmentModeOptions(selected, includeInherit = false) {
            const opts = [];
            if (includeInherit) {
                opts.push(`<option value="" ${!selected ? 'selected' : ''}>(Use Script Default)</option>`);
            }
            opts.push(...SUBTITLE_ALIGNMENT_MODES.map(mode =>
                `<option value="${mode}" ${mode === (selected || '') ? 'selected' : ''}>${mode}</option>`
            ));
            return opts.join('');
        }

        function blockSummaryText(block, index) {
            const chars = (block.text || '').trim().length;
            if (!chars) return `Block ${index + 1}`;
            return `Block ${index + 1} · ${chars} chars`;
        }
