const { VOICES, SUBTITLE_PRESETS, DEFAULT_SUBTITLE_PRESET, DEFAULT_SUBTITLE_ALIGNMENT_MODE, INITIAL_SCRIPTS } = window.APP_BOOTSTRAP;

        let scriptNames = [...INITIAL_SCRIPTS];
        let activeScriptName = null;
        let blocks = [];
        let currentMode = 'visual';
        let isDirty = false;
        let suppressDirtyTracking = false;
        let selectedBlockIndex = null;
        let dragSourceIndex = null;
        let validationErrors = [];
        let latestOutputItems = [];
        let validateDebounceTimer = null;
        let factBankFilterDebounceTimer = null;
        let latestFactBankItems = [];
        let latestFactBankServerItems = [];
        let latestFactBankTotal = 0;
        let latestSourceDraft = null;
        let latestHandledScript = null;
        let latestSavedDraftItems = [];
        let latestSourceMeta = null;
        let latestSourceWarnings = [];
        let recentScriptNames = [];

        const SUBTITLE_MODES = ['standard', 'progressive'];
        const SUBTITLE_ALIGNMENT_MODES = ['edge', 'corrected', 'forced'];
        const BANK_SELECTION_MODES = ['top', 'balanced', 'random_weighted'];
        const OUTPUT_RECENT_LIMIT = 3;
        const SCRIPT_RECENT_LIMIT = 3;
        const RATE_PERCENT_RE = /^[+-]\d+%$/;
        const PITCH_HZ_RE = /^[+-]\d+Hz$/i;
        const JOB_STAGE_ORDER = ['loading', 'tts', 'subtitle', 'rendering', 'done'];
        const JOB_STAGE_LABELS = {
            loading: 'Loading Models',
            tts: 'TTS',
            subtitle: 'Subtitle',
            rendering: 'Rendering',
            done: 'Done',
            error: 'Error',
        };
        const COMPARE_PRESET_ORDER = ['minimal', 'energetic', 'cinematic'];

        function togglePanel(side) {
            const workspace = document.getElementById('workspace');
            const panel = side === 'left' ? document.getElementById('panelLeft') : document.getElementById('panelRight');
            const btn = panel.querySelector('.btn-panel-toggle');
            
            const isCollapsed = panel.classList.toggle('is-collapsed');
            workspace.classList.toggle(`${side}-collapsed`, isCollapsed);
            
            if (side === 'left') {
                btn.textContent = isCollapsed ? '»' : '«';
            } else {
                btn.textContent = isCollapsed ? '«' : '»';
            }
        }

        function setTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('is-active', btn.textContent.toLowerCase().includes(tabId.substring(0, 3)));
            });
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.toggle('is-active', content.id === `tab-${tabId}`);
            });
        }

        function setWorkspaceView(view) {
            const normalized = view === 'crawl' ? 'crawl' : view === 'agent' ? 'agent' : 'editor';
            const crawlScreen = document.getElementById('crawlScreen');
            const editorScreen = document.getElementById('editorScreen');
            const agentScreen = document.getElementById('workspaceAgent');
            const btnCrawl = document.getElementById('btnWorkspaceCrawl');
            const btnEditor = document.getElementById('btnWorkspaceEditor');
            const btnAgent = document.getElementById('btnWorkspaceAgent');

            if (crawlScreen) crawlScreen.classList.toggle('is-active', normalized === 'crawl');
            if (editorScreen) editorScreen.classList.toggle('is-active', normalized === 'editor');
            if (agentScreen) {
                agentScreen.hidden = normalized !== 'agent';
                agentScreen.classList.toggle('is-active', normalized === 'agent');
            }
            if (btnCrawl) btnCrawl.classList.toggle('is-active', normalized === 'crawl');
            if (btnEditor) btnEditor.classList.toggle('is-active', normalized === 'editor');
            if (btnAgent) btnAgent.classList.toggle('is-active', normalized === 'agent');
        }

        function updateSaveState(dirty) {
            isDirty = dirty;
            const badge = document.getElementById('saveState');
            if (!badge) return;
            badge.classList.toggle('is-dirty', dirty);
            badge.classList.toggle('is-saved', !dirty);
            badge.textContent = dirty ? 'Unsaved changes' : 'Saved';
        }

        function markDirty() {
            if (suppressDirtyTracking) return;
            updateSaveState(true);
        }

        function markSaved() {
            updateSaveState(false);
        }

        function shouldIgnoreDirtyTracking(target) {
            return Boolean(target?.dataset?.ignoreDirty === 'true');
        }

        function scheduleValidation(delayMs = 120) {
            if (validateDebounceTimer) clearTimeout(validateDebounceTimer);
            validateDebounceTimer = setTimeout(() => {
                validateDebounceTimer = null;
                validateForm();
            }, delayMs);
        }

        function emptyToNull(value) {
            const v = (value ?? '').toString().trim();
            return v === '' ? null : v;
        }

        function keywordTextFromArray(value) {
            if (Array.isArray(value)) return value.join(', ');
            if (typeof value === 'string') return value;
            return '';
        }

        function parseKeywordText(value) {
            return (value || '')
                .split(',')
                .map(s => s.trim())
                .filter(Boolean);
        }

        function updateBlockField(index, field, value, nullIfEmpty = false) {
            if (!blocks[index]) return;
            blocks[index][field] = nullIfEmpty ? emptyToNull(value) : value;
        }

        function addValidationError(errors, group, fieldId, message, blockIndex = null) {
            errors.push({ group, fieldId, message, blockIndex });
        }

        function validatePercentValue(value) {
            return !value || RATE_PERCENT_RE.test(value);
        }

        function validatePitchValue(value) {
            return !value || PITCH_HZ_RE.test(value);
        }

        function setFieldValidationState(errors) {
            document.querySelectorAll('.field-invalid').forEach(el => el.classList.remove('field-invalid'));
            document.querySelectorAll('[aria-invalid="true"]').forEach(el => el.removeAttribute('aria-invalid'));

            const seen = new Set();
            for (const err of errors) {
                if (!err.fieldId || seen.has(err.fieldId)) continue;
                const el = document.getElementById(err.fieldId);
                if (!el) continue;
                el.classList.add('field-invalid');
                el.setAttribute('aria-invalid', 'true');
                seen.add(err.fieldId);
            }
        }

        function renderValidationSummary() {
            const panel = document.getElementById('validationPanel');
            const list = document.getElementById('validationList');
            if (!panel || !list) return;

            if (!validationErrors.length) {
                panel.hidden = true;
                list.innerHTML = '';
                return;
            }

            const grouped = validationErrors.reduce((acc, err, idx) => {
                if (!acc[err.group]) acc[err.group] = [];
                acc[err.group].push({ ...err, idx });
                return acc;
            }, {});

            list.innerHTML = Object.entries(grouped).map(([groupName, items]) => {
                const row = items.map(item => `
                    <button type="button" class="validation-item" onclick="focusValidationError(${item.idx})">${escapeHtml(item.message)}</button>
                `).join('');
                return `
                    <div class="validation-group">${escapeHtml(groupName)}</div>
                    ${row}
                `;
            }).join('');
            panel.hidden = false;
        }

        function validateForm() {
            const errors = [];

            const scriptVoiceRate = emptyToNull(document.getElementById('scriptVoiceRate').value);
            const scriptVoicePitch = emptyToNull(document.getElementById('scriptVoicePitch').value);
            const scriptVoiceVolume = emptyToNull(document.getElementById('scriptVoiceVolume').value);
            const scriptSubtitlePreset = emptyToNull(document.getElementById('scriptSubtitlePreset').value);
            const scriptSubtitleAlignmentMode = emptyToNull(document.getElementById('scriptSubtitleAlignmentMode').value);

            if (!validatePercentValue(scriptVoiceRate)) {
                addValidationError(errors, 'Script Settings', 'scriptVoiceRate', 'Global voice rate must match +10% or -5%.');
            }
            if (!validatePitchValue(scriptVoicePitch)) {
                addValidationError(errors, 'Script Settings', 'scriptVoicePitch', 'Global voice pitch must match +8Hz or -3Hz.');
            }
            if (!validatePercentValue(scriptVoiceVolume)) {
                addValidationError(errors, 'Script Settings', 'scriptVoiceVolume', 'Global voice volume must match +6% or -4%.');
            }
            if (scriptSubtitlePreset && !SUBTITLE_PRESETS.includes(scriptSubtitlePreset)) {
                addValidationError(errors, 'Script Settings', 'scriptSubtitlePreset', `Script subtitle preset "${scriptSubtitlePreset}" is not valid.`);
            }
            if (scriptSubtitleAlignmentMode && !SUBTITLE_ALIGNMENT_MODES.includes(scriptSubtitleAlignmentMode)) {
                addValidationError(errors, 'Script Settings', 'scriptSubtitleAlignmentMode', `Alignment mode "${scriptSubtitleAlignmentMode}" is not valid.`);
            }

            blocks.forEach((block, i) => {
                const group = `Block ${i + 1}`;
                if (!(block.text || '').trim()) {
                    addValidationError(errors, group, `block-${i}-text`, 'Text is required.', i);
                }

                const voiceRate = emptyToNull(block.voice_rate);
                const voicePitch = emptyToNull(block.voice_pitch);
                const voiceVolume = emptyToNull(block.voice_volume);
                const subtitlePreset = emptyToNull(block.subtitle_preset);
                const subtitleMode = emptyToNull(block.subtitle_mode);
                const subtitleAlignmentMode = emptyToNull(block.subtitle_alignment_mode);

                if (!validatePercentValue(voiceRate)) {
                    addValidationError(errors, group, `block-${i}-voice-rate`, 'Voice rate must match +10% or -5%.', i);
                }
                if (!validatePitchValue(voicePitch)) {
                    addValidationError(errors, group, `block-${i}-voice-pitch`, 'Voice pitch must match +8Hz or -3Hz.', i);
                }
                if (!validatePercentValue(voiceVolume)) {
                    addValidationError(errors, group, `block-${i}-voice-volume`, 'Voice volume must match +6% or -4%.', i);
                }
                if (subtitlePreset && !SUBTITLE_PRESETS.includes(subtitlePreset)) {
                    addValidationError(errors, group, `block-${i}-subtitle-preset`, `Subtitle preset "${subtitlePreset}" is not valid.`, i);
                }
                if (subtitleMode && !SUBTITLE_MODES.includes(subtitleMode)) {
                    addValidationError(errors, group, `block-${i}-subtitle-mode`, `Subtitle mode "${subtitleMode}" is not valid.`, i);
                }
                if (subtitleAlignmentMode && !SUBTITLE_ALIGNMENT_MODES.includes(subtitleAlignmentMode)) {
                    addValidationError(errors, group, `block-${i}-subtitle-alignment-mode`, `Alignment mode "${subtitleAlignmentMode}" is not valid.`, i);
                }
            });

            validationErrors = errors;
            setFieldValidationState(errors);
            renderValidationSummary();
            return { valid: errors.length === 0, errors };
        }

        function focusValidationError(index) {
            const err = validationErrors[index];
            if (!err) return;

            if (typeof err.blockIndex === 'number' && blocks[err.blockIndex]) {
                selectedBlockIndex = err.blockIndex;
                if (blocks[err.blockIndex].__collapsed) {
                    blocks[err.blockIndex].__collapsed = false;
                    renderBlocks();
                } else {
                    refreshBlockSelection();
                }
            }

            requestAnimationFrame(() => {
                const el = document.getElementById(err.fieldId);
                if (!el) return;
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                el.focus();
            });
        }

        function normalizeJobStage(job) {
            if (job && JOB_STAGE_ORDER.includes(job.stage)) return job.stage;
            if (!job) return 'loading';
            if (job.status === 'done') return 'done';
            if (job.status === 'error') return 'rendering';
            if (job.status === 'loading' || job.status === 'starting') return 'loading';
            return 'rendering';
        }

        function renderJobTimeline(job) {
            const panel = document.getElementById('jobTimeline');
            const steps = document.getElementById('timelineSteps');
            const blockPanel = document.getElementById('blockProgress');
            const blockLabel = document.getElementById('blockProgressLabel');
            const blockFill = document.getElementById('blockProgressFill');
            if (!panel || !steps || !blockPanel || !blockLabel || !blockFill) return;

            const stage = normalizeJobStage(job);
            const stageIndex = JOB_STAGE_ORDER.indexOf(stage);
            const resolvedIndex = stageIndex === -1 ? 0 : stageIndex;

            steps.innerHTML = JOB_STAGE_ORDER.map((stageKey, idx) => {
                const classes = ['timeline-step'];
                const isDoneState = job.status === 'done';
                const isErrorState = job.status === 'error';

                if (idx < resolvedIndex || (isDoneState && idx <= resolvedIndex)) {
                    classes.push('is-complete');
                }
                if (!isDoneState && !isErrorState && idx === resolvedIndex) {
                    classes.push('is-active');
                }
                if (isErrorState && idx === resolvedIndex) {
                    classes.push('is-error');
                }

                return `<div class="${classes.join(' ')}">${JOB_STAGE_LABELS[stageKey] || stageKey}</div>`;
            }).join('');

            const totalBlocks = Number(job.total_blocks || 0);
            const currentBlock = Number(job.current_block || 0);
            const canShowBlockProgress =
                totalBlocks > 0 && ['tts', 'subtitle', 'rendering'].includes(stage);

            if (canShowBlockProgress) {
                const safeCurrent = Math.max(0, Math.min(currentBlock, totalBlocks));
                const percent = Math.round((safeCurrent / totalBlocks) * 100);
                blockLabel.textContent = `${JOB_STAGE_LABELS[stage]}: block ${safeCurrent}/${totalBlocks}`;
                blockFill.style.width = `${percent}%`;
                blockPanel.hidden = false;
            } else {
                blockLabel.textContent = '';
                blockFill.style.width = '0%';
                blockPanel.hidden = true;
            }

            panel.hidden = false;
        }

        function clearErrorPanel() {
            const panel = document.getElementById('errorPanel');
            const headline = document.getElementById('errorHeadline');
            const hint = document.getElementById('errorHint');
            const detailsWrap = document.getElementById('errorDetailsWrap');
            const details = document.getElementById('errorDetails');
            if (!panel || !headline || !hint || !detailsWrap || !details) return;
            panel.hidden = true;
            headline.textContent = '';
            hint.textContent = '';
            hint.hidden = true;
            details.textContent = '';
            detailsWrap.hidden = true;
            detailsWrap.open = false;
        }

        function showErrorPanel(errorInfo) {
            const panel = document.getElementById('errorPanel');
            const headline = document.getElementById('errorHeadline');
            const hint = document.getElementById('errorHint');
            const detailsWrap = document.getElementById('errorDetailsWrap');
            const details = document.getElementById('errorDetails');
            if (!panel || !headline || !hint || !detailsWrap || !details) return;

            headline.textContent = errorInfo.headline || 'Request failed.';
            const hintText = (errorInfo.hint || '').trim();
            hint.textContent = hintText;
            hint.hidden = !hintText;

            const detailsText = errorInfo.details
                ? (typeof errorInfo.details === 'string'
                    ? errorInfo.details
                    : JSON.stringify(errorInfo.details, null, 2))
                : '';
            details.textContent = detailsText;
            detailsWrap.hidden = !detailsText;
            panel.hidden = false;
        }

        function normalizeApiError(errorLike, fallbackMessage = 'Request failed', status = null) {
            let raw = errorLike;
            if (raw && typeof raw === 'object' && raw.error) raw = raw.error;

            if (raw && typeof raw === 'object' && raw.headline) {
                const code = raw.code || 'UNKNOWN_ERROR';
                return {
                    code,
                    status: raw.status ?? status,
                    headline: raw.headline || fallbackMessage,
                    details: raw.details || null,
                    hint: raw.hint || null,
                };
            }

            if (raw && typeof raw === 'object') {
                const code = raw.code || 'UNKNOWN_ERROR';
                return {
                    code,
                    status: raw.status ?? status,
                    headline: raw.message || raw.error || fallbackMessage,
                    details: raw.details || null,
                    hint: raw.hint || null,
                };
            }

            if (typeof raw === 'string') {
                return {
                    code: 'UNKNOWN_ERROR',
                    status,
                    headline: raw || fallbackMessage,
                    details: null,
                    hint: null,
                };
            }

            if (raw instanceof Error) {
                return {
                    code: 'UNKNOWN_ERROR',
                    status,
                    headline: fallbackMessage,
                    details: raw.message,
                    hint: null,
                };
            }

            return {
                code: 'UNKNOWN_ERROR',
                status,
                headline: fallbackMessage,
                details: null,
                hint: null,
            };
        }

        function showApiError(errorLike, fallbackMessage = 'Request failed') {
            const normalized = normalizeApiError(errorLike, fallbackMessage);
            showStatus(normalized.headline, 'error');
            showErrorPanel(normalized);
            return normalized;
        }

        async function fetchApi(url, options = {}, fallbackMessage = 'Request failed') {
            let response;
            try {
                response = await fetch(url, options);
            } catch (err) {
                throw normalizeApiError(
                    {
                        code: 'NETWORK_ERROR',
                        message: 'Network request failed.',
                        details: String(err),
                        hint: 'Ensure the server is running and retry.',
                    },
                    fallbackMessage,
                );
            }

            let payload = null;
            try {
                payload = await response.json();
            } catch (err) {
                payload = null;
            }

            if (!response.ok) {
                throw normalizeApiError(payload, fallbackMessage, response.status);
            }
            return payload;
        }

        function setSourceBadge(text, options = {}) {
            const badge = document.getElementById('sourceBadge');
            if (!badge) return;
            const isDebug = Boolean(options.debug);
            const value = String(text || '').trim();
            if (!value) {
                badge.hidden = true;
                badge.textContent = '';
                badge.classList.remove('is-debug');
                return;
            }
            badge.classList.toggle('is-debug', isDebug);
            badge.textContent = value;
            badge.hidden = false;
        }

        function resetSourcePromptOverride() {
            const promptInput = document.getElementById('sourcePromptOverride');
            if (!promptInput) return;
            promptInput.value = '';
            showStatus('Using default handling prompt.', 'done');
        }

        function createDefaultSourceDraft(language = 'vi-VN') {
            return {
                id: '',
                source: 'source_1',
                topic_query: '',
                language,
                title: '',
                source_url: '',
                fetched_at: '',
                sections: [],
                warnings: [],
            };
        }

        function setSourceDraftEditorValue(sourceDraft) {
            const editor = document.getElementById('sourceDraftEditor');
            if (!editor) return;
            const language = (document.getElementById('scriptLang')?.value || 'vi-VN').trim() || 'vi-VN';
            const payload = sourceDraft && typeof sourceDraft === 'object'
                ? sourceDraft
                : createDefaultSourceDraft(language);
            editor.value = JSON.stringify(payload, null, 2);
        }

        function readSourceDraftEditorValue() {
            const editor = document.getElementById('sourceDraftEditor');
            if (!editor) {
                throw normalizeApiError(
                    {
                        code: 'SOURCE_DRAFT_EDITOR_MISSING',
                        message: 'Raw draft editor is not available in UI.',
                    },
                    'Source draft editor is missing.',
                );
            }

            const raw = (editor.value || '').trim();
            if (!raw) {
                throw normalizeApiError(
                    {
                        code: 'SOURCE_DRAFT_REQUIRED',
                        message: 'Raw draft JSON is empty.',
                        hint: 'Fetch a draft first or paste source_draft JSON.',
                    },
                    'Raw draft JSON is empty.',
                );
            }

            let parsed = null;
            try {
                parsed = JSON.parse(raw);
            } catch (err) {
                throw normalizeApiError(
                    {
                        code: 'INVALID_SOURCE_DRAFT_JSON',
                        message: 'Raw draft JSON is invalid.',
                        details: err.message,
                        hint: 'Fix JSON syntax and retry.',
                    },
                    'Raw draft JSON is invalid.',
                );
            }

            if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.sections)) {
                throw normalizeApiError(
                    {
                        code: 'INVALID_SOURCE_DRAFT',
                        message: 'Raw draft JSON must include sections array.',
                        hint: 'Use the fetched source draft shape with sections.',
                    },
                    'Invalid source draft shape.',
                );
            }
            return parsed;
        }

        function renderSourceWarnings(elementId, warnings) {
            const warningsEl = document.getElementById(elementId);
            if (!warningsEl) return;
            const items = Array.isArray(warnings) ? warnings : [];
            if (!items.length) {
                warningsEl.hidden = true;
                warningsEl.innerHTML = '';
                return;
            }
            warningsEl.innerHTML = items.map(warning => `
                <li class="source-warning-item">${escapeHtml(String(warning))}</li>
            `).join('');
            warningsEl.hidden = false;
        }

        function clearSourceDraftView() {
            latestSourceDraft = null;
            latestHandledScript = null;
            latestSourceMeta = null;
            latestSourceWarnings = [];
            setSourceBadge('');

            const status = document.getElementById('sourceFetchStatus');
            if (status) {
                status.textContent = 'Fetch section-based draft from Wikipedia for this topic.';
            }
            const handleStatus = document.getElementById('sourceHandleStatus');
            if (handleStatus) {
                handleStatus.textContent = 'Edit raw draft JSON or run handling to build video-ready script blocks.';
            }

            const metaPanel = document.getElementById('sourceMetaPanel');
            const metaTitle = document.getElementById('sourceMetaTitle');
            const line1 = document.getElementById('sourceMetaLine1');
            const line2 = document.getElementById('sourceMetaLine2');
            const line3 = document.getElementById('sourceMetaLine3');
            if (metaPanel) metaPanel.hidden = true;
            if (metaTitle) metaTitle.textContent = '';
            if (line1) line1.textContent = '';
            if (line2) line2.textContent = '';
            if (line3) line3.textContent = '';

            const summaryPanel = document.getElementById('sourceDraftSummary');
            const statsEl = document.getElementById('sourceDraftStats');
            const sectionListEl = document.getElementById('sourceSectionList');
            if (summaryPanel) summaryPanel.hidden = true;
            if (statsEl) statsEl.textContent = '';
            if (sectionListEl) sectionListEl.innerHTML = '';

            const handleMetaPanel = document.getElementById('sourceHandleMetaPanel');
            const handleTitle = document.getElementById('sourceHandleMetaTitle');
            const handleLine1 = document.getElementById('sourceHandleMetaLine1');
            const handleLine2 = document.getElementById('sourceHandleMetaLine2');
            if (handleMetaPanel) handleMetaPanel.hidden = true;
            if (handleTitle) handleTitle.textContent = '';
            if (handleLine1) handleLine1.textContent = '';
            if (handleLine2) handleLine2.textContent = '';

            renderSourceWarnings('sourceWarnings', []);
            renderSourceWarnings('sourceHandleWarnings', []);
            const draftSectionSelector = document.getElementById('sourceDraftSectionSelector');
            if (draftSectionSelector) draftSectionSelector.value = '';
            setSourceDraftEditorValue(null);
        }

        function _hasSourceDraftSections(sourceDraft) {
            return Boolean(
                sourceDraft
                && typeof sourceDraft === 'object'
                && Array.isArray(sourceDraft.sections)
                && sourceDraft.sections.length > 0
            );
        }

        function renderSourceDraftResult(data) {
            const sourceMeta = (data && typeof data === 'object' ? data.source_meta : null) || {};
            const sourceDraft = (data && typeof data === 'object' ? data.source_draft : null) || {};
            const draftStats = (data && typeof data === 'object' ? data.draft_stats : null) || {};
            const sectionSelector = String((data && typeof data === 'object' ? data.section_selector : '') || '').trim();
            const sections = Array.isArray(sourceDraft.sections) ? sourceDraft.sections : [];
            const warnings = Array.isArray(data?.warnings) ? data.warnings : [];
            latestSourceMeta = sourceMeta;
            latestSourceWarnings = warnings;
            const title = String(sourceMeta.title || 'Wikipedia result').trim();
            const url = String(sourceMeta.url || '').trim();
            const confidenceText = Number.isFinite(Number(sourceMeta.confidence))
                ? `${Math.round(Number(sourceMeta.confidence) * 100)}%`
                : '-';
            const fetchedText = formatTimestamp(sourceMeta.fetched_at);
            const langText = String(sourceMeta.lang || document.getElementById('scriptLang')?.value || 'vi-VN');
            const cacheText = sourceMeta.cache_hit ? 'cache' : 'live';

            const metaPanel = document.getElementById('sourceMetaPanel');
            const metaTitle = document.getElementById('sourceMetaTitle');
            const line1 = document.getElementById('sourceMetaLine1');
            const line2 = document.getElementById('sourceMetaLine2');
            const line3 = document.getElementById('sourceMetaLine3');
            if (metaPanel && metaTitle && line1 && line2 && line3) {
                if (url) {
                    metaTitle.innerHTML = `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(title)}</a>`;
                } else {
                    metaTitle.textContent = title;
                }
                line1.textContent = `Source: ${sourceMeta.source || 'source_1'} · ${cacheText} · Confidence: ${confidenceText}`;
                line2.textContent = `Fetched: ${fetchedText} · Language: ${langText}`;
                const sectionCount = Number.isFinite(Number(draftStats.section_count))
                    ? Number(draftStats.section_count)
                    : sections.length;
                const sentenceCount = Number.isFinite(Number(draftStats.sentence_count))
                    ? Number(draftStats.sentence_count)
                    : '-';
                const wordCount = Number.isFinite(Number(draftStats.word_count))
                    ? Number(draftStats.word_count)
                    : '-';
                line3.textContent = `Sections: ${sectionCount} · Sentences: ${sentenceCount} · Words: ${wordCount}${sectionSelector ? ` · Filter: ${sectionSelector}` : ''}`;
                metaPanel.hidden = false;
            }

            const summaryPanel = document.getElementById('sourceDraftSummary');
            const statsEl = document.getElementById('sourceDraftStats');
            const sectionListEl = document.getElementById('sourceSectionList');
            if (summaryPanel && statsEl && sectionListEl) {
                const sectionCount = sections.length;
                const sentenceCount = Number.isFinite(Number(draftStats.sentence_count))
                    ? Number(draftStats.sentence_count)
                    : '-';
                const wordCount = Number.isFinite(Number(draftStats.word_count))
                    ? Number(draftStats.word_count)
                    : '-';
                statsEl.textContent = `${sectionCount} sections · ${sentenceCount} sentences · ${wordCount} words`;
                const previewLimit = 10;
                const preview = sections.slice(0, previewLimit);
                sectionListEl.innerHTML = preview.map((section, idx) => {
                    const rank = Number.isFinite(Number(section.rank)) ? Number(section.rank) : idx + 1;
                    const sectionId = String(section.section_id || '').trim();
                    const sectionTitle = String(section.title || `Section ${rank}`).trim() || `Section ${rank}`;
                    const sectionText = String(section.text || '').trim();
                    const snippet = sectionText.length > 200 ? `${sectionText.slice(0, 197)}...` : sectionText;
                    return `
                        <article class="source-section-item">
                            <div class="source-section-title">${escapeHtml(`${rank}. ${sectionTitle}${sectionId ? ` (${sectionId})` : ''}`)}</div>
                            <p class="source-section-snippet">${escapeHtml(snippet || '(empty section)')}</p>
                        </article>
                    `;
                }).join('');
                if (sections.length > previewLimit) {
                    sectionListEl.innerHTML += `<p class="source-meta-line">Showing ${previewLimit}/${sections.length} sections.</p>`;
                }
                summaryPanel.hidden = false;
            }

            renderSourceWarnings('sourceWarnings', warnings);

            const status = document.getElementById('sourceFetchStatus');
            if (status) {
                status.textContent = `Draft fetched (${sections.length} sections). Ready for handling.`;
            }

            setSourceBadge(`Source: ${title} · ${sections.length} sections`, { debug: false });
        }

        function renderSourceHandleResult(data) {
            const mode = String(data?.handling_mode || '').trim() || '-';
            const warnings = Array.isArray(data?.warnings) ? data.warnings : [];
            const meta = (data && typeof data === 'object' ? data.meta : null) || {};
            const llmMeta = (data && typeof data === 'object' ? data.llm_meta : null) || null;
            const script = (data && typeof data === 'object' ? data.script : null) || null;
            const hasScript = Boolean(script && Array.isArray(script.blocks) && script.blocks.length > 0);
            latestHandledScript = hasScript ? script : null;

            const status = document.getElementById('sourceHandleStatus');
            if (status) {
                if (mode === 'raw_json') {
                    status.textContent = 'Raw draft validated. Save JSON or switch mode to build script.';
                } else if (hasScript) {
                    status.textContent = `Handled with mode "${mode}" (${Number(meta.output_block_count || 0)} blocks).`;
                } else {
                    status.textContent = `Handled with mode "${mode}" but no script was produced.`;
                }
            }

            const panel = document.getElementById('sourceHandleMetaPanel');
            const title = document.getElementById('sourceHandleMetaTitle');
            const line1 = document.getElementById('sourceHandleMetaLine1');
            const line2 = document.getElementById('sourceHandleMetaLine2');
            if (panel && title && line1 && line2) {
                title.textContent = `Handling: ${mode}`;
                line1.textContent = `Sections: ${Number(meta.section_count || 0)} · Raw blocks: ${Number(meta.raw_block_count || 0)} · Output blocks: ${Number(meta.output_block_count || 0)}`;
                if (llmMeta && (llmMeta.status || llmMeta.model)) {
                    const provider = llmMeta.provider || 'groq';
                    const model = llmMeta.model ? ` / ${llmMeta.model}` : '';
                    const statusText = llmMeta.status ? ` · ${llmMeta.status}` : '';
                    line2.textContent = `LLM: ${provider}${model}${statusText}`;
                } else {
                    line2.textContent = 'LLM: not used.';
                }
                panel.hidden = false;
            }
            renderSourceWarnings('sourceHandleWarnings', warnings);
        }
