# UI/UX Upgrade Execution Board

Reference plan: `UI_UX_Upgrade_Plan.md`

## Phase A: Foundation (High Impact)

- [x] **A1. Introduce design tokens and theme foundation**
  - [x] Create CSS variables for colors, spacing, radii, typography, shadows.
  - [x] Replace hard-coded style values with tokens.
  - [x] Add light component states: hover, focus, disabled.
  - [x] Verify contrast for primary text and form controls.
  - Done when:
    - [x] Main UI uses tokenized styles consistently.
    - [x] No obvious contrast/accessibility regressions.

- [x] **A2. Refactor page layout into clear editor zones**
  - [x] Implement responsive 3-column shell (script settings / blocks / preview).
  - [x] Add sticky header with script name + save state + quick actions.
  - [x] Ensure mobile and small-laptop fallbacks are clean.
  - Done when:
    - [x] Layout is stable on desktop and mobile widths.
    - [x] Core actions remain visible without excessive scrolling.

- [x] **A3. Group settings by intent**
  - [x] Script panel sections: `Project`, `Global Voice`, `Subtitle Defaults`, `Background`.
  - [x] Block cards: `Content`, `Voice`, `Subtitle`, `Assets`.
  - [x] Move advanced controls behind collapsible sections.
  - Done when:
    - [x] New users can find global vs block overrides quickly.

- [x] **A4. Add inline helper text and formatting hints**
  - [x] Add helper rows for `rate`, `pitch`, `volume` formats.
  - [x] Show preset/mode explanation near subtitle controls.
  - Done when:
    - [x] Invalid format attempts are reduced in testing.

## Phase B: Editing Productivity

- [x] **B1. Block actions and speed editing**
  - [x] Add duplicate block action.
  - [x] Add move up/down actions.
  - [x] Add drag-and-drop reorder.
  - [x] Add block collapse/expand toggle.
  - Done when:
    - [x] Reordering and duplicating blocks requires minimal clicks.

- [x] **B2. Script-level "apply to all blocks" tools**
  - [x] Apply voice to all blocks.
  - [x] Apply subtitle preset to all blocks.
  - [x] Apply selected voice controls to all blocks (rate/pitch/volume).
  - Done when:
    - [x] Bulk style changes complete in one action.

- [x] **B3. Keyboard shortcuts**
  - [x] `Ctrl/Cmd+S`: Save script.
  - [x] `Ctrl/Cmd+Enter`: Generate video.
  - [x] `Ctrl/Cmd+Shift+B`: Add block.
  - [x] `Ctrl/Cmd+Shift+D`: Duplicate selected block.
  - [x] `Ctrl/Cmd+Shift+J`: Toggle JSON mode.
  - Done when:
    - [x] Shortcuts are documented and work without conflicts.

- [x] **B4. Script management improvements**
  - [x] Add search/filter for existing scripts.
  - [x] Add duplicate script action.
  - [x] Add export script button.
  - [x] Remove full-page reload after save/delete.
  - Done when:
    - [x] Script workflows are entirely in-place and fast.

## Phase C: Reliability and Feedback

- [x] **C1. Client-side validation layer**
  - [x] Validate required block text.
  - [x] Validate `%` and `Hz` field formats.
  - [x] Validate known enums (`subtitle_mode`, `subtitle_preset`).
  - [x] Prevent generate on invalid form state.
  - Done when:
    - [x] User gets actionable feedback before API call.

- [x] **C2. Validation summary panel**
  - [x] Show grouped error list by section/block.
  - [x] Click error focuses the problematic field.
  - [x] Keep summary synced with edits.
  - Done when:
    - [x] Users can resolve all validation errors from one panel.

- [x] **C3. Better job progress UX**
  - [x] Replace single status line with stage timeline:
    - [x] loading models
    - [x] tts
    - [x] subtitle
    - [x] rendering
    - [x] done/error
  - [x] Show current block progress for long jobs.
  - Done when:
    - [x] Render progress is understandable at a glance.

- [x] **C4. Better error reporting**
  - [x] Standardize backend error response shape.
  - [x] Show concise error headline + expandable details.
  - [x] Provide next-action hints in error UI.
  - Done when:
    - [x] Common failures can be fixed without checking server logs.

## Phase D: Advanced Workflow

- [x] **D1. Preset compare workflow**
  - [x] Add quick "render with minimal/energetic/cinematic" actions.
  - [x] Label outputs with preset metadata.
  - [x] Keep latest comparison outputs grouped.
  - Done when:
    - [x] A/B subtitle testing can be run from UI in a few clicks.

- [x] **D2. Output card improvements**
  - [x] Add file size and created time.
  - [x] Add optional thumbnail preview.
  - [x] Add quick open/download/delete actions.
  - Done when:
    - [x] Output list feels like a usable production panel.

- [x] **D3. JSON power tools**
  - [x] Add format JSON button.
  - [x] Add JSON lint with line/column errors.
  - [x] Add diff vs saved script.
  - Done when:
    - [x] JSON mode is practical for advanced users.

## Cross-Cutting Tasks

- [x] **X1. Accessibility QA**
  - [x] Keyboard-only navigation for all major controls.
  - [x] Focus ring visible on interactive elements.
  - [x] Check label/input associations.
  - [x] Contrast checks for text and controls.

- [x] **X2. Performance QA**
  - [x] Debounce frequent UI updates.
  - [x] Avoid full rerender of all blocks on small edits.
  - [x] Test long scripts (20+ blocks) for responsiveness.

- [x] **X3. Regression Testing**
  - [x] Existing script load/save/generate still works.
  - [x] JSON import/export unchanged and valid.
  - [x] Output preview and delete still work.

## Milestone Acceptance

- [ ] **Milestone M1 (After Phase A)**
  - [ ] Visual polish clearly improved.
  - [ ] Layout and settings hierarchy are clearer.

- [ ] **Milestone M2 (After Phase B)**
  - [ ] Editing speed improved for multi-block scripts.
  - [ ] Core repetitive actions are shortcut/bulk enabled.

- [ ] **Milestone M3 (After Phase C)**
  - [ ] Validation catches most user mistakes before generate.
  - [ ] Progress and errors are transparent.

- [ ] **Milestone M4 (After Phase D)**
  - [ ] A/B subtitle testing and output review are efficient.
  - [ ] Advanced JSON and output workflows are usable.

## Suggested Execution Order

- [ ] Week 1: Phase A
- [ ] Week 2: Phase B
- [ ] Week 3: Phase C
- [ ] Week 4: Phase D + cross-cutting QA + polish
