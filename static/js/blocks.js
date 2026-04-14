        function renderBlocks() {
            const container = document.getElementById('blocksContainer');
            container.innerHTML = blocks.map((b, i) => `
                <div class="block ${i === selectedBlockIndex ? 'is-selected' : ''}" data-index="${i}" draggable="true"
                    tabindex="0"
                    onkeydown="onBlockKeydown(event, ${i})"
                    onclick="selectBlock(${i}, event)"
                    ondragstart="onBlockDragStart(event, ${i})"
                    ondragover="onBlockDragOver(event, ${i})"
                    ondragleave="onBlockDragLeave(event)"
                    ondrop="onBlockDrop(event, ${i})"
                    ondragend="onBlockDragEnd(event)">
                    
                    <button type="button" class="btn-danger btn-small" aria-label="Delete block ${i + 1}" 
                        onclick="event.stopPropagation(); removeBlock(${i})" 
                        style="position: absolute; top: 12px; right: 12px; padding: 4px 8px; font-size: 0.7rem;">✕</button>

                    <div class="block-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="drag-handle" title="Drag to reorder" style="cursor: grab; opacity: 0.4;">⋮⋮</span>
                            <div>
                                <span class="block-num">${blockSummaryText(b, i)}</span>
                                ${b.source_topic || b.fact_id || b.source_section_id || b.source_url ? `
                                    <div class="block-source-meta">
                                        ${b.source_topic ? `<span class="block-source-pill">${escapeHtml(String(b.source_topic))}</span>` : ''}
                                        ${b.source_section_id ? `<span class="block-source-pill">sec ${escapeHtml(String(b.source_section_id))}</span>` : ''}
                                        ${b.fact_id ? `<span class="block-source-pill">fact ${escapeHtml(String(b.fact_id))}</span>` : ''}
                                        ${b.source_url ? `<span class="block-source-pill">${escapeHtml(String(b.source_url))}</span>` : ''}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                        <div class="block-header-tools">
                            <button type="button" class="btn-ghost btn-small" title="${b.__collapsed ? 'Expand' : 'Collapse'}" 
                                onclick="event.stopPropagation(); toggleBlockCollapse(${i})">${b.__collapsed ? '展开' : '收起'}</button>
                            <button type="button" class="btn-ghost btn-small" title="Duplicate" onclick="event.stopPropagation(); duplicateBlock(${i})">⧉</button>
                            <button type="button" class="btn-ghost btn-small" title="Move up" onclick="event.stopPropagation(); moveBlock(${i}, -1)">↑</button>
                            <button type="button" class="btn-ghost btn-small" title="Move down" onclick="event.stopPropagation(); moveBlock(${i}, 1)">↓</button>
                        </div>
                    </div>

                    <div class="block-body ${b.__collapsed ? 'is-hidden' : ''}">
                        <div class="block-row" style="grid-template-columns: 1.5fr 1fr; gap: 16px;">
                            <section>
                                <label for="block-${i}-text">Script Narrative</label>
                                <textarea id="block-${i}-text" oninput="updateBlockField(${i}, 'text', this.value)" 
                                    placeholder="Enter the voice-over text for this block...">${escapeHtml(b.text || '')}</textarea>
                            </section>
                            
                            <div style="display: flex; flex-direction: column; gap: 8px;">
                                <section>
                                    <label for="block-${i}-voice">Voice</label>
                                    <select id="block-${i}-voice" onchange="updateBlockField(${i}, 'voice', this.value)">${voiceOptions(b.voice)}</select>
                                </section>
                                <section>
                                    <label for="block-${i}-subtitle-preset">Style Preset</label>
                                    <select id="block-${i}-subtitle-preset" onchange="updateBlockField(${i}, 'subtitle_preset', this.value, true)">${presetOptions(b.subtitle_preset, true)}</select>
                                </section>
                            </div>
                        </div>

                        <details class="advanced-panel mt-sm" style="background: rgba(15, 23, 42, 0.2); border-radius: 8px;">
                            <summary style="padding: 8px 12px; font-size: 0.75rem; font-weight: 700; color: var(--text-secondary); cursor: pointer; user-select: none;">Media & Overrides</summary>
                            <div class="advanced-body" style="padding: 12px; border-top: 1px solid var(--border-default);">
                                <div class="block-row" style="margin-bottom: 12px;">
                                    <div>
                                        <label style="font-size: 0.7rem;">Background Asset</label>
                                        <input type="text" id="block-${i}-background" value="${escapeHtml(b.background || '')}" placeholder="path/to/video.mp4" oninput="updateBlockField(${i}, 'background', this.value, true)">
                                    </div>
                                    <div>
                                        <label style="font-size: 0.7rem;">Overlay Image</label>
                                        <input type="text" id="block-${i}-image" value="${escapeHtml(b.image || '')}" placeholder="path/to/image.png" oninput="updateBlockField(${i}, 'image', this.value, true)">
                                    </div>
                                </div>
                                <div class="block-row-3">
                                    <div>
                                        <label style="font-size: 0.7rem;">Speed</label>
                                        <input type="text" id="block-${i}-voice-rate" value="${escapeHtml(b.voice_rate || '')}" placeholder="+0%" oninput="updateBlockField(${i}, 'voice_rate', this.value, true)">
                                    </div>
                                    <div>
                                        <label style="font-size: 0.7rem;">Pitch</label>
                                        <input type="text" id="block-${i}-voice-pitch" value="${escapeHtml(b.voice_pitch || '')}" placeholder="+0Hz" oninput="updateBlockField(${i}, 'voice_pitch', this.value, true)">
                                    </div>
                                    <div>
                                        <label style="font-size: 0.7rem;">Align</label>
                                        <select id="block-${i}-subtitle-alignment-mode" onchange="updateBlockField(${i}, 'subtitle_alignment_mode', this.value, true)">${alignmentModeOptions(b.subtitle_alignment_mode, true)}</select>
                                    </div>
                                </div>
                                <div class="mt-sm">
                                    <label style="font-size: 0.7rem;">Highlight Keywords (comma separated)</label>
                                    <input type="text" id="block-${i}-subtitle-keywords" value="${escapeHtml(b.subtitle_keywords || '')}" placeholder="key word, highlight..." oninput="updateBlockField(${i}, 'subtitle_keywords', this.value)">
                                </div>
                            </div>
                        </details>
                    </div>
                </div>
            `).join('');
            refreshBlockSelection();
            validateForm();
        }

        function selectBlock(index, event) {
            if (event && event.target && event.target.closest('button, input, textarea, select, summary, a')) {
                return;
            }
            selectedBlockIndex = index;
            refreshBlockSelection();
        }

        function onBlockKeydown(event, index) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                selectedBlockIndex = index;
                refreshBlockSelection();
            }
        }

        function toggleBlockCollapse(index) {
            blocks[index].__collapsed = !blocks[index].__collapsed;
            renderBlocks();
            markDirty();
        }

        function addBlock() {
            const selectedVoice = document.getElementById('scriptDefaultVoice').value || (VOICES[0]?.id || 'NamMinh');
            const insertAt = selectedBlockIndex == null ? blocks.length : selectedBlockIndex + 1;
            blocks.splice(insertAt, 0, {
                voice: selectedVoice,
                text: '',
                voice_rate: null,
                voice_pitch: null,
                voice_volume: null,
                background: null,
                image: null,
                fact_id: null,
                source_topic_id: null,
                source_topic: null,
                source_url: null,
                subtitle_preset: null,
                subtitle_mode: null,
                subtitle_alignment_mode: null,
                subtitle_style: null,
                subtitle_keywords: '',
                __collapsed: false,
            });
            selectedBlockIndex = insertAt;
            renderBlocks();
            markDirty();
        }

        function removeBlock(index) {
            blocks.splice(index, 1);
            if (!blocks.length) selectedBlockIndex = null;
            else selectedBlockIndex = Math.max(0, Math.min(selectedBlockIndex ?? 0, blocks.length - 1));
            renderBlocks();
            markDirty();
        }

        function duplicateBlock(index) {
            if (!blocks[index]) return;
            const clone = JSON.parse(JSON.stringify(blocks[index]));
            clone.__collapsed = false;
            blocks.splice(index + 1, 0, clone);
            selectedBlockIndex = index + 1;
            renderBlocks();
            markDirty();
        }

        function duplicateSelectedBlock() {
            if (!blocks.length) return;
            const index = selectedBlockIndex == null ? blocks.length - 1 : selectedBlockIndex;
            duplicateBlock(index);
        }

        function moveBlock(index, direction) {
            const target = index + direction;
            if (target < 0 || target >= blocks.length) return;
            const [item] = blocks.splice(index, 1);
            blocks.splice(target, 0, item);
            selectedBlockIndex = target;
            renderBlocks();
            markDirty();
        }

        function onBlockDragStart(event, index) {
            dragSourceIndex = index;
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', String(index));
            event.currentTarget.classList.add('dragging');
            selectedBlockIndex = index;
            refreshBlockSelection();
        }

        function onBlockDragOver(event, index) {
            event.preventDefault();
            if (dragSourceIndex == null || dragSourceIndex === index) return;
            event.currentTarget.classList.add('drag-over');
        }

        function onBlockDragLeave(event) {
            event.currentTarget.classList.remove('drag-over');
        }

        function onBlockDrop(event, index) {
            event.preventDefault();
            event.currentTarget.classList.remove('drag-over');
            if (dragSourceIndex == null || dragSourceIndex === index) return;

            const [moved] = blocks.splice(dragSourceIndex, 1);
            const insertIndex = dragSourceIndex < index ? index - 1 : index;
            blocks.splice(insertIndex, 0, moved);
            selectedBlockIndex = insertIndex;
            dragSourceIndex = null;
            renderBlocks();
            markDirty();
        }

        function onBlockDragEnd(event) {
            dragSourceIndex = null;
            event.currentTarget.classList.remove('dragging');
            document.querySelectorAll('.block.drag-over').forEach(el => el.classList.remove('drag-over'));
        }

        function applyVoiceToAll() {
            if (!blocks.length) return;
            const voice = document.getElementById('scriptDefaultVoice').value || (VOICES[0]?.id || 'NamMinh');
            blocks.forEach(block => { block.voice = voice; });
            renderBlocks();
            markDirty();
            showStatus('Applied voice to all blocks.', 'done');
        }

        function applyVoiceControlsToAll() {
            if (!blocks.length) return;
            const voiceRate = emptyToNull(document.getElementById('scriptVoiceRate').value);
            const voicePitch = emptyToNull(document.getElementById('scriptVoicePitch').value);
            const voiceVolume = emptyToNull(document.getElementById('scriptVoiceVolume').value);

            blocks.forEach(block => {
                block.voice_rate = voiceRate;
                block.voice_pitch = voicePitch;
                block.voice_volume = voiceVolume;
            });
            renderBlocks();
            markDirty();
            showStatus('Applied voice controls to all blocks.', 'done');
        }

        function applySubtitlePresetToAll() {
            if (!blocks.length) return;
            const preset = emptyToNull(document.getElementById('scriptSubtitlePreset').value);
            blocks.forEach(block => { block.subtitle_preset = preset; });
            renderBlocks();
            markDirty();
            showStatus('Applied subtitle preset to all blocks.', 'done');
        }

        function applySubtitleAlignmentModeToAll() {
            if (!blocks.length) return;
            const mode = normalizeAlignmentModeValue(
                document.getElementById('scriptSubtitleAlignmentMode').value
            );
            blocks.forEach(block => { block.subtitle_alignment_mode = mode; });
            renderBlocks();
            markDirty();
            showStatus('Applied subtitle alignment mode to all blocks.', 'done');
        }
