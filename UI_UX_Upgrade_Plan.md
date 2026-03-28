# UI/UX Upgrade Plan for Video Maker

## 1. Purpose
Upgrade the current web UI so it looks more professional and makes script editing faster, safer, and easier for daily production.

## 2. Goals
- Improve visual quality (modern, clean, high-contrast, less "tool-like").
- Reduce editing friction for script/block workflows.
- Make advanced settings (voice, subtitle, presets) easier to understand.
- Improve reliability of editing with validation, safer defaults, and recovery.
- Keep render workflow fast and clear with better progress/error feedback.

## 3. Current Pain Points (Observed)
- Form layout is dense and repetitive when many blocks are present.
- Advanced controls are available but not grouped by intent.
- Visual and JSON editing modes are separate but not tightly synchronized.
- Limited inline guidance for formatting values (`+10%`, `+8Hz`, etc.).
- Script management is basic (no search/filter/duplicate/export shortcut flow).
- Rendering feedback is minimal (single status line, limited stage visibility).

## 4. Product UX Principles
- Fast first: common actions must take fewer clicks.
- Progressive disclosure: basic controls visible, advanced controls collapsible.
- Safe by default: sensible defaults + clear validation before render.
- Predictable structure: script-level settings first, then block-level overrides.
- Production-focused: support batch-like repetitive editing with shortcuts.

## 5. Information Architecture (New Structure)
- Top header:
  - Project title, active script name, save status, quick actions.
- Left column: Script settings
  - Language, background, subtitle preset, global voice controls.
  - Import/export, save, duplicate, delete.
- Center: Block timeline/editor
  - Reorderable block cards, add/duplicate/delete, collapse/expand.
  - Inline text + quick voice/subtitle controls.
- Right column: Preview + output
  - Current render preview, recent outputs, job history/progress.

## 6. Visual Design Upgrade
- Introduce design tokens (CSS variables):
  - color palette, spacing scale, radius scale, font scale.
- Typography:
  - clear heading hierarchy and readable body text.
- Components:
  - consistent card styles, compact controls, clear focus states.
- Color usage:
  - neutral base + one strong accent + one warning color.
- Density:
  - compact mode toggle for power users.

## 7. Editing Experience Upgrade

### 7.1 Script-Level Editing
- Group global settings in labeled sections:
  - `Project`, `Global Voice`, `Subtitle Defaults`, `Background`.
- Add helper text for value formats:
  - `rate`: `+10%`, `-5%`
  - `pitch`: `+8Hz`, `-3Hz`
  - `volume`: `+6%`, `-4%`
- Add "apply to all blocks" actions for selected fields.

### 7.2 Block Editor
- Block card with:
  - title, quick stats (chars, estimated duration), voice badge.
- Advanced settings collapsible by default.
- Block actions:
  - duplicate, move up/down, drag reorder, delete confirm.
- Add keyboard shortcuts:
  - add block, duplicate block, save, generate, toggle JSON mode.

### 7.3 JSON + Visual Sync
- Keep both views synchronized at all times (single source of truth).
- Show JSON parse errors with exact line hints.
- Add "format JSON" and "diff from saved script" helpers.

## 8. Validation and Error UX
- Inline field validation before save/generate:
  - invalid `%` and `Hz` formats, empty text blocks, missing required fields.
- Validation summary panel:
  - list all issues with click-to-focus on broken field.
- Render errors:
  - show failed stage (`TTS`, `subtitle`, `ffmpeg`) and actionable message.

## 9. Output and Job UX
- Job progress timeline:
  - `loading models` -> `tts` -> `subtitle` -> `rendering` -> `done`.
- Output cards:
  - thumbnail, duration, created time, open/download/delete.
- Add "render with preset A/B/C" quick actions from current script.

## 10. Accessibility and Responsiveness
- Fully usable on 1366px laptop and mobile width.
- Visible keyboard focus for all controls.
- Label every input and keep contrast AA-compliant.
- Avoid layout break with long script names and long block text.

## 11. Performance Plan
- Virtualize long block lists (render only visible cards).
- Debounced autosave and state updates.
- Avoid full-page reload after save/delete script.
- Lazy-load heavy preview sections.

## 12. Implementation Roadmap

### Phase A: Foundation (High Impact)
- Add design tokens and new layout shell (3-column responsive).
- Refactor existing controls into structured sections.
- Add inline help text and validation messages.
- Acceptance:
  - no functionality regression, better readability at first glance.

### Phase B: Editing Productivity
- Add block duplicate/reorder/collapse.
- Add keyboard shortcuts.
- Add apply-to-all controls.
- Acceptance:
  - creating/editing 10-block script is noticeably faster.

### Phase C: Reliability and Feedback
- Add validation summary and stage-based job progress.
- Improve error surfacing and recovery actions.
- Acceptance:
  - user can identify/fix issues without checking backend logs.

### Phase D: Advanced Workflow
- Add preset compare actions and richer output cards.
- Add JSON formatting and diff helper.
- Acceptance:
  - easier A/B workflow and script version confidence.

## 13. Suggested Technical Tasks
- Frontend structure:
  - split large inline JS into modules (`state`, `ui`, `api`, `validators`).
- UI components:
  - reusable card/input/toggle/alert/progress components.
- State model:
  - single store for script + dirty state + validation state.
- API usage:
  - keep existing endpoints, improve client-side handling and retries.

## 14. Success Metrics
- Time to create a 5-block script: target < 3 minutes.
- Edit error rate before generate: reduce by at least 40%.
- Number of clicks to duplicate/reorder blocks: reduce by at least 50%.
- User-reported clarity score (internal): target >= 8/10.

## 15. Immediate Next Step
- Start with a low-risk UI shell refactor and validation layer (Phase A).
- Keep all current features functional while migrating incrementally.
