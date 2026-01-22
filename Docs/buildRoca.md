# ROCA Animator (PyQt6) — Build Plan & Status

Modern 2D animation authoring built on the **Routed Orbital Capsule Architecture (ROCA)**.

**Runtime:** Python 3.10+ • **UI:** PyQt6 • **Imaging:** OpenCV + NumPy

---

## Quick Start

1) Install deps: `pip install -r requirements.txt`
2) Launch: `python main.py`

---

## Current Status

*Updated: December 15, 2025*

✅ Shipping MVP (usable end-to-end)

- **Layout:** Canvas (center) • Timeline (bottom) • Orbital Capsule Map (right dock)
- **Timeline:** Custom-painted (`QPainter`) with thumbnails, playback, scrubbing
- **Capsules:** Deterministic UUID5 IDs + usage salience (`orbit_score`)
- **Import:** Folder/video import + Explorer drag-and-drop (timeline and canvas)
- **Autonomy:** Deterministic self-labeling via image-derived hash (`asset_hash`)
- **Persistence:** `.roca` save/load (manifest + assets + agreement edges)

---

## Phase 1 — Foundation & Core Logic

- [x] **1.1 Project Setup**
    - Dependencies in `requirements.txt` (PyQt6, numpy, opencv-python)
    - Project structure (`core/`, `gui/`, assets)
    - GPU/OpenCL detection for OpenCV (`core/device_manager.py`)

- [x] **1.2 Capsule Data Structure**
    - Deterministic UUID5 IDs from `(kind, name)`
    - `CapsuleKind`: Pose / Transition / Timing / Cycle / Character / Memory
    - Extension: `UNASSIGNED` (library-only capsules; not auto-placed on timeline)
    - Metadata: `created_at`, `use_count`, `last_used_at`, `orbit_score`

- [x] **1.3 Capsule Store & Agreement Graph**
    - Capsule registry + agreement edges (compatibility history)

- [x] **1.4 Symbolic Router (Engine)**
    - Deterministic in-between generation via OpenCV blending + timing curves

---

## Phase 2 — UI (Canvas + Orbit)

- [x] **2.1 Main Window Shell**
    - QMainWindow toolbars + dock widgets
    - Dark theme (readable controls)

- [x] **2.2 Canvas (Tools + Layers)**
    - Tools: Pen, Eraser, Line, Rectangle, Fill
    - Undo/Redo
    - Layers: Add/Del + active layer selector
    - Shortcuts: `Ctrl+[` / `Ctrl+]` switches layers
    - Accepts file drops (creates `UNASSIGNED` capsules, does not pollute timeline)

- [x] **2.3 Orbital Capsule Map**
    - Ring visualization + drag capsule to timeline
    - Salience affects orbit placement (visual-only)

---

## Phase 3 — Timeline & Animation Workflow

- [x] **3.1 Timeline Playback & Interaction**
    - Scrub, select, play

- [x] **3.2 Canvas ⇄ Timeline Sync**
    - Selecting a frame loads its capsule image onto the canvas
    - Capture Pose creates a Pose capsule and assigns it to the current frame

- [x] **3.3 Capture Pose**
    - Self-labels with `pose_<hash>` if no name is provided

- [x] **3.4 Transitions**
    - Transition generator dialog inserts generated frames at the playhead
    - Generated transition-frame capsules use content-hash naming to avoid UUID5 collisions

- [x] **3.5 Cycles**
    - Create Cycle capsule from frame range
    - Apply Cycle by drag-and-drop (unroll with loop count)

- [x] **3.6 Drag & Drop Import**
    - Drop image files/folders onto the timeline to auto-import as Pose capsules
    - Drop onto canvas for “library-only” Unassigned capsules

- [x] **3.7 Agreement Checks (Lightweight)**
    - Warns on “untested” capsule pairings (low agreement score)

---

## Phase 4 — Persistence & Export/Import

- [x] **4.1 Project Persistence (.roca)**
    - Saves capsules + agreement edges + timeline list
    - Saves per-capsule PNG assets

- [~] **4.2 Determinism Verification**
    - Determinism is by design (UUID5 + content hashing)
    - No automated test suite checked in (optional)

- [x] **4.3 Import/Export Utilities**
    - Load Images (folder → frames)
    - Load Video (video → frames)
    - Export Video (timeline → MP4)
    - Export SVG: placeholder (vector tracing not implemented)

---

## Phase 5 — Performance & Audio

- [x] **5.1 Background Processing**
    - Transition generation runs in a background worker (QThread)

- [x] **5.2 Audio Support (MVP)**
    - Timeline supports audio load via `QMediaPlayer` + `QAudioOutput`
    - Waveform visualization: not implemented

---

## Known Gaps (Intentional / Planned)

- “Train” is a legacy button placeholder in this PyQt build (no ML training pipeline here)
- In-between quality is intentionally basic; higher-quality symbolic interpolation can be added
- No capsule merge/shadow-identity system yet (dedupe is currently hash/name-based)