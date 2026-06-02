        async function generate() {
            const context = getGenerateContext();
            if (!context) return;
            showStatus('Starting...', 'processing');
            renderJobTimeline({
                status: 'loading',
                stage: 'loading',
                progress: 'Starting...',
                current_block: 0,
                total_blocks: blocks.length,
            });

            try {
                const jobId = await startGenerationRequest(
                    context.name,
                    context.script,
                    context.name,
                    null,
                    true,
                );
                activeScriptName = context.name;
                markSaved();
                await loadScriptList(context.name);
                await pollJob(jobId, { previewOnDone: true });
            } catch (err) {
                showApiError(err, 'Failed to start generation.');
            }
        }

        async function pollJob(jobId, options = {}) {
            const previewOnDone = options.previewOnDone !== false;
            return new Promise((resolve, reject) => {
                const poll = async () => {
                    try {
                        const job = await fetchApi(
                            `/api/jobs/${jobId}`,
                            {},
                            'Failed to fetch job status.',
                        );
                        renderJobTimeline(job);

                        if (job.status === 'done') {
                            showStatus(job.progress || 'Video generated!', 'done');
                            loadOutputs();
                            if (previewOnDone && job.output) previewVideo(job.output);
                            resolve(job);
                            return;
                        }
                        if (job.status === 'error') {
                            const jobError = job.error || {
                                code: 'GENERATION_FAILED',
                                message: 'Video generation failed.',
                            };
                            const normalized = showApiError(jobError, 'Video generation failed.');
                            reject(normalized);
                            return;
                        }

                        showStatus(job.progress || 'Processing...', 'processing');
                        setTimeout(poll, 1500);
                    } catch (err) {
                        const normalized = showApiError(err, 'Failed to fetch job status.');
                        reject(normalized);
                    }
                };
                poll();
            });
        }

        function showStatus(msg, type) {
            const bar = document.getElementById('statusBar');
            bar.textContent = msg;
            bar.className = `status-bar show status-${type}`;
            if (type !== 'error') clearErrorPanel();
        }

        function previewVideo(filename) {
            const video = document.getElementById('preview');
            video.src = `/output/${encodeURIComponent(filename)}`;
            video.load();
        }

        function previewVideoByEncoded(encodedFilename) {
            previewVideo(decodeURIComponent(encodedFilename));
        }

        async function loadOutputs() {
            try {
                const files = await fetchApi('/api/outputs', {}, 'Failed to load output list.');
                renderOutputList(files);
            } catch (err) {
                showApiError(err, 'Failed to load output list.');
            }
        }

        async function deleteOutput(filename) {
            if (!confirm('Delete ' + filename + '?')) return;
            try {
                await fetchApi(
                    `/api/outputs/${encodeURIComponent(filename)}`,
                    { method: 'DELETE' },
                    'Failed to delete output file.',
                );
                loadOutputs();
            } catch (err) {
                showApiError(err, 'Failed to delete output file.');
            }
        }

        async function deleteOutputByEncoded(encodedFilename) {
            await deleteOutput(decodeURIComponent(encodedFilename));
        }


        function handleShortcuts(event) {
            const key = event.key.toLowerCase();
            const isMod = event.ctrlKey || event.metaKey;

            if (key === 'escape') {
                const picker = document.getElementById('outputPickerBackdrop');
                if (picker && !picker.hidden) {
                    event.preventDefault();
                    closeOutputPicker();
                    return;
                }
            }

                if (isMod) {
                if (key === 's') {
                    event.preventDefault();
                    saveScript();
                    return;
                }
                if (key === 'enter') {
                    event.preventDefault();
                    generate();
                    return;
                }
                if (key === '[') {
                    event.preventDefault();
                    togglePanel('left');
                    return;
                }
                if (key === ']') {
                    event.preventDefault();
                    togglePanel('right');
                    return;
                }
                if (['1', '2', '3', '4'].includes(key)) {
                    event.preventDefault();
                    const tabs = ['load', 'project', 'subtitles', 'voice'];
                    setTab(tabs[parseInt(key) - 1]);
                    return;
                }
            }

            if (!event.shiftKey || !isMod) return;

            if (key === 'b') {
                event.preventDefault();
                addBlock();
                return;
            }
            if (key === 'd') {
                event.preventDefault();
                duplicateSelectedBlock();
                return;
            }
            if (key === 'j') {
                event.preventDefault();
                setMode(currentMode === 'visual' ? 'json' : 'visual');
            }
        }

        document.getElementById('scriptSubtitlePreset').innerHTML = presetOptions(DEFAULT_SUBTITLE_PRESET, false);
        document.getElementById('scriptSubtitleAlignmentMode').innerHTML = alignmentModeOptions(
            normalizeAlignmentModeValue(DEFAULT_SUBTITLE_ALIGNMENT_MODE),
            false,
        );
        document.getElementById('scriptDefaultVoice').innerHTML = voiceOptions(VOICES[0]?.id || 'NamMinh');

        document.addEventListener('input', (event) => {
            if (event.target.matches('input, textarea, select')) {
                if (!shouldIgnoreDirtyTracking(event.target)) {
                    markDirty();
                    scheduleValidation(120);
                }
                if (event.target.id === 'jsonArea') {
                    clearJsonDiffPanel();
                }
            }
        });
        document.addEventListener('change', (event) => {
            if (event.target.matches('input, textarea, select')) {
                if (!shouldIgnoreDirtyTracking(event.target)) {
                    markDirty();
                    scheduleValidation(40);
                }
            }
        });
        document.addEventListener('keydown', handleShortcuts);

        clearSourceDraftView();
        setTab('load');
        setWorkspaceView('editor');
        renderScriptList();
        renderBlocks();
        validateForm();
        markSaved();
        refreshSavedDraftList({ silent: true });
        loadOutputs();
