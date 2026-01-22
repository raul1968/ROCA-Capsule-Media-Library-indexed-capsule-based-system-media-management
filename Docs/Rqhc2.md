# RQHC — ROCA Quantum / Hinton Capsule Compiler (Developer Paper)

**Authors:** Threadripper-ROCA Research Collective
**Date:** 2026-01-20

## Abstract

This document (RQHC) synthesizes the ROCA — Routed Orbital Capsule Architecture — conversation and implementation notes into a developer-oriented paper. It clarifies the thesis that capsule-style symbolic architectures are uniquely suited for professional creative intelligence (PCI), documents the Threadripper-ROCA implementation, and provides reproducible guidance for engineers: design principles, architecture, core algorithms (including BVH/mocap ingestion, timeline-driven animation, deterministic capsule routing), performance considerations, experiments, and extension points for production studios.

Importantly, we address a commonly-cited objection: that capsule networks "failed" in computer vision. We summarize why that assessment is a domain-mismatch rather than an architectural condemnation, and we present measurable claims and experimental designs to validate that capsules—when applied to structured creative workflows—deliver determinism, one-shot learning, interpretability, and scalable parallel routing.


We propose ROCA (Routed Orbital Capsule Architecture), a systems architecture and visualization layer for long-lived AI assistants in which structured “capsules” represent characters, styles, skills, workflows, and memories as explicit first-class objects. ROCA combines (i) dynamic routing over a capsule graph, (ii) usage- and agreement-driven self-organization, and (iii) a radial “Saturn ring” spatial layout that makes the assistant’s evolving internal repertoire legible over months and years. The center of the layout represents an Identity Nucleus (core self capsules used constantly), while specialized capsules occupy functional ring lanes (e.g., characters, styles, skills, memories) and drift inward or outward over time as a continuous function of salience. Importantly, orbit distance affects UI only: it is a faithful visualization of capsule salience rather than a gate on routing. ROCA also introduces reversible coalescing via shadow identities: near-duplicate character capsules may merge into a proxy while preserving original identities to support later divergence. We describe data structures, update equations, spawning and coalescing rules, and a runtime loop analogous to an engine-style entity-component system. We outline an evaluation plan comparing ROCA’s orbital UI to conventional list-based memory browsing while keeping identical retrieval backends.

## Contributions

This paper targets engineers and product teams building long-lived creative assistants. Our primary contributions are:

- A practical system architecture for ROCA: capsule schema, append-only store, deterministic router, and UI projection.
- An operationalized orbital UI model (Identity Nucleus, explicit lanes, continuous drift) with reversible coalescing semantics.
- Concrete ingestion and runtime patterns (BVH/mocap parsing, timeline-driven frames, deterministic composition) and validation criteria.
- Threadripper-scale scaling patterns (NUMA distribution, lock-free updates, deterministic parallelism) and minimal prototype sketches.
- Reproducible tests and smoke suites to ensure determinism, ingest correctness, and transition evaluation.

## Background & Related Work

ROCA synthesizes ideas from multiple areas: capsule networks (Sabour et al., 2017), event-sourced memory systems, self-organizing maps, and entity-component runtime loops common in game engines. Unlike prior capsule work focused on vision benchmarks, ROCA applies capsule-style primitives to structured creative cognition where determinism, provenance, and compositional routing are primary requirements. We position ROCA as a systems-level architecture that can sit above LLMs and perception modules, providing a deterministic, explainable layer for professional creative workflows.

## 1. Motivation & Thesis

- Problem: Modern creative production demands absolute reproducibility, explainability, long-term memory, and deterministic composition — requirements poorly served by stochastic foundation models.
- Thesis: Capsule-based, deterministic, additive knowledge structures ("capsules") are a superior fit for professional creative workflows. ROCA reframes Hinton's capsules as creative cognition primitives rather than vision primitives.
- Goals: Provide a practical, high-quality developer reference to implement, evaluate, and extend ROCA-style systems for animation, VFX, and collaborative creative pipelines.

## 2. Terminology

- Capsule: a self-describing, versioned knowledge unit (ID, vector content, metadata, provenance, links).
- Capsule Routing: deterministic selection and composition of capsules to produce outputs.
- Capsule Genome: representation of a capsule's parameters subject to genetic-style evolution.
- Orbital Visualization: UI metaphor representing capsule clusters and relations.
- PCI: Professional Creative Intelligence — a product category distinct from AGI and general LLM tooling.

## 3. Design Principles

1. Determinism by design: identical inputs and constraints → identical outputs (UUID-based identity, deterministic composition).
2. Additive memory: never delete; always append new capsules and links to avoid catastrophic forgetting.
3. Explainability: keep provenance and composition traces for every output.
4. Domain-first architecture: adapt algorithms to creative use-cases (composition, transitions, stylistic constraints) rather than forcing perceptual benchmarks.
5. Scale horizontally on CPUs: favor NUMA-aware multiprocessing and vectorized ops over monolithic GPUs for deterministic pipelines.

## 4. Capsule Domain Fit: "Why Capsules Failed in Vision" — A Reappraisal

Summary: many early evaluations concluded capsules "failed" because they were judged by vision benchmarks (ImageNet-style classification) that favor invariance, stochastic robustness, and massive distributed representations. That testbed magnified properties of capsules that are weaknesses for generic perception but strengths for structured creative cognition.

Key points:

- Viewpoint equivariance vs invariance: vision benchmarks require invariance to pose/lighting; capsules emphasize equivariance and structured transforms — a mismatch for classification but essential for compositional creative reasoning.
- Stochastic vs deterministic needs: visual perception benefits from stochastic generalization across noisy sensors; creative production demands deterministic reproducibility and provenance.
- Scale and search complexity: routing-by-agreement is expensive when applied to thousands of low-level visual primitives; creative hierarchies are shallower and human-guided, dramatically reducing necessary routing combinations.
- Label noise and ground-truth: image labels are often ambiguous; creative tasks provide clearer intent and constraints (style guides, editor decisions), making structured capsules practical and testable.

Counter-claim (operational): When evaluated on creative-domain metrics (determinism, one-shot learning fidelity, interpretability, long-term style retention), capsule-style systems outperform stochastic baselines on the practical criteria studios care about. Section 11 (Experiments & Metrics) defines the validation procedures.


## 5. System Architecture (High Level)

- Ingest Layer: file parsers (text, PDF, image OCR, BVH mocap, code AST) that emit capsules.
- Capsule Store: indexed, immutable capsule repository with metadata, timestamps, and relationships.
- Router / Composer: deterministic composition engine implementing routing-by-agreement constrained by context and style rules.
- Evolution & Expansion Engine: optional genetic engine for exploring parameter spaces (capsule genomes) with parallel evaluation.
- Autonomous Brain: long-running cognitive loop (think/solidify/infer) that consolidates memories into higher-order capsules.
- UI Layer: interactive interfaces (chat collaborator, 2D canvas, timeline, orbital visualizer) for humans to inspect, teach, and accept suggestions.

Diagram (conceptual): Ingest → Capsules → Capsule Store ↔ Router/Composer → Outputs → Human Feedback → New Capsules

## 6. Core Data Model

## 6. Threadripper-ROCA: Scalability & NUMA Optimizations

This section summarizes practical techniques used to scale ROCA to 64–128 CPU cores (Threadripper-class machines). The goal is to give engineers concrete, low-risk patterns they can implement and test on high-core-count servers.

6.1 Key Innovations (practical summary)

- NUMA-aware capsule distribution: hash capsules to NUMA nodes to reduce cross-node memory latency; cache popular capsules locally and provide controlled eviction.
- Lock-free / low-lock data structures: use atomic updates, compare-and-swap, or thread-local batching to avoid global locks in capsule galaxy updates.
- Parallel orbital updates: update capsule physics or ranking in parallel using Numba/NumPy or ProcessPools with careful memory partitioning.
- Deterministic parallelism: ensure all parallel pipelines use deterministic ordering or seeded RNGs derived from stable keys so results are repeatable across runs.
- Multi-user sync engine: partition scene state spatially and resolve conflicts in parallel, using deterministic resolution policies and traceable merges.

6.2 Measurable Claims & How to Validate

Engineers should treat high-level performance numbers as hypotheses to be validated on target hardware. Suggested validation steps (see Section 11 for formal acceptance criteria):

- NUMA hit-rate: benchmark capsule fetch latency with and without hash-based NUMA placement; target hit-rate > 80% for active working sets.
- Parallel update throughput: measure capsules/sec for orbital updates at several core counts; plot scaling curve and identify memory or synchronization bottlenecks.
- Sync latency: measure end-to-end user action → global state commit latency under synthetic multi-user load; target 60 FPS perceived sync for interactive regions.

6.3 Implementation notes & caveats

- Start with a conservative design: implement a simple hash-to-node store and measure; premature optimization for micro-architectural details can add complexity without practical gains.
- Use process-based isolation for heavy compute (ProcessPoolExecutor) and shared-memory + mmap for large read-mostly arrays to avoid Python GIL limits.
- Profile NUMA effects on Windows vs Linux; many NUMA tuning knobs are Linux-specific (numactl, hugepages, thread affinity).
- Always add deterministic checks (unit tests) when introducing parallelism to ensure the core property of repeatability is not lost.


Capsule (minimal schema):

- id: deterministic UUID (e.g., UUID5 namespace + stable keys)
- type: e.g., `pose_capsule`, `style`, `frame`, `rule`
- vector: dense float vector (for similarity, optional)
- content: structured payload (JSON or binary blobs)
- metadata: provenance, source file, creator, timestamp, version
- links: directed edges to other capsule ids with relation types and confidence

Example JSON:

{
  "id": "uuid5:...",
  "type": "pose_capsule",
  "vector": [...],
  "content": {"bone_names": [...], "base_pose": [...]},
  "metadata": {"source":"/data/mocap/foo.bvh","frames":120},
  "links": [{"to":"uuid5:...","rel":"similar_to","score":0.92}]
}

## 7. Deterministic Routing (algorithm sketch)

Inputs: context C (project state, user constraints, timeline slot), candidate capsule set S.

1. Filter S by type & active constraints (fast index lookup).
2. Deterministic scoring: compute a stable composite key from (C, capsule.id) and use a deterministic ranking function (e.g., stable hash → seeded pseudo-rng + weighted scoring; or pure scoring with deterministic tie-breakers).
3. Compose: select top-k capsules; apply composition operators (transform, blend) in defined order.
4. Emit: output plus full composition trace (all inputs, scores, transforms).

Properties: repeatable across runs, traceable, and explainable.

## 8. Runtime Loop & Orbit Mapping

ROCA is well-modeled as an engine-style loop that emits domain events consumed by the UI and downstream systems. This section provides a concise developer reference for the runtime and the orbit mapping formulas used by the visualization layer.

Runtime tick (engine loop):

1. Collect intents (user actions, tool results, automated detectors).
2. Route: select relevant capsules via deterministic Router (see Section 7).
3. Execute: materialize working context and invoke model/tool chains.
4. Emit events: `UseCapsule`, `SpawnCapsule`, `MergeCapsules`, `DecayTick`, `OrbitUpdate`.
5. Apply physics: update orbitState (radius, phase) for UI smoothing.
6. Persist event log (append-only) for reproducibility and replay.

Gravity & orbit mapping (developer-ready equations):

Define a gravity score `g(c) ∈ [0,1]` per capsule updated from usage and agreement signals. One deployed formula used in the reference implementation:

$$
 g(c) = \sigma\Big( w_f \cdot \log(1 + \text{useCount}_c) + w_r \cdot e^{-\Delta t_c / \tau} + w_a \cdot \text{agreement}_c + w_k \cdot \text{connectivity}_c \Big)
$$

Map gravity to lane `L` radial band `[r_{min}(L), r_{max}(L)]` with smoothing toward target radius `r^*(c)`:

$$
 r^*(c) = r_{\min}(L) + (1 - g(c)) \cdot (r_{\max}(L) - r_{\min}(L))
$$

$$
 r_{t+1}(c) = r_t(c) + \lambda \cdot (r^*(c) - r_t(c))
$$

Angular phase `\theta(c)` is stable via a hash-based assignment to preserve identity:

$$
 	heta(c) = 2\pi \cdot \text{hash01}(id_c)
$$

Notes for implementers:

- Keep orbitState UI-only; do not expose `r`/`\theta` to routing decisions.
- Choose `\tau` and `\lambda` to match your product time horizon (e.g., months-scale `\tau`, small `\lambda` to avoid jitter).
- Emit event logs to reconstruct UI and routing decisions for audits and reproducibility.

## 8. BVH / Mocap Ingestion (implementation notes)

- Parse HIERARCHY into bone nodes with offsets, channels, and channel_start indices.
- Stream MOTION frames into NumPy arrays when frames are large; compute per-frame global transforms via forward kinematics.
- Emit a `pose_capsule` representing canonical poses (seed frames) and optionally sampled keyframes for timeline use.
- Serialize transforms into capsule content as lists (JSON) or compressed binary for storage.
- Normalize coordinate systems and annotate metadata (coordinate_up, handedness).

Code sketch (Python):

- Build bone table with `channel_start` indexes.
- Use NumPy to parse frames: `motion = np.fromfile(...).reshape((frames, total_channels))` or line-based streaming.
- Compute global matrices (FK) per frame; sample representative frames (e.g., every Nth or by motion-change metric) and create `pose_capsule`s.

## 9. 2D Animation GUI & Timeline (developer guidance)

- Canvas: lightweight immediate-mode widget to draw strokes; store stroke paths per frame as capsule `frame` objects (serializable arrays).
- Timeline: ordered list of frame capsules; support drag-and-drop reordering and diff/transition evaluation.
- Transition evaluator: compute similarity between frames (vectorized path encoding or joint-space distance); provide binary feedback (good/bad) with score and recorded reasoning (why flagged).
- Chat collaborator: accept commands (`/ingest`, `/animate`, `/suggest`) and present deterministic suggestions computed by the Router.

UX principle: always show composition trace and allow artist to lock capsules into the composition (immutability + override).

## 10. Genetic Engines for Capsule Genomes

- Use GA for parameter-space exploration (voice timbre, style blending ratios, procedural generator seeds).
- Highly parallel fitness evaluation: map fitness across many CPU cores; keep the evolutionary process deterministic by controlling RNG seeds via genome hashes.
- Use elitism + tournament selection; preserve top-n capsules as canonical variants (append-only store).

## 11. Experiments & Metrics (recommended benchmarks)
- Formal claims & acceptance criteria:

- Determinism: identical inputs and constraints produce byte-identical outputs and identical composition traces across repeated runs (target: 100% match over n=100 runs).
- One-shot learning: ingest a single exemplar and reproduce stylistic attributes on new prompts with high fidelity (target: >80% agreement with artist ratings on core attributes).
- Interpretability: every composed output must include a human-readable trace linking to source capsules (target: 100% traceability for production outputs).
- Long-term stability: recompose assets after simulated incremental updates (10k operations) and measure style-parameter drift (target: ≤1% drift on canonical parameters over simulated 1-year workload).

- Determinism test: repeated runs (n=100) of same composition → byte-identical outputs + identical composition traces.
- Consistency over time: simulate incremental use (10k operations) and measure drift in style parameters when recomposing earlier assets.
- Single-shot learning: ingest N=1 exemplar and measure reproduction fidelity on new prompts (qualitative scoring by artists).
- Transition quality: evaluate rated good/bad transitions over a seed dataset of 200 paired frames; metric = agreement with human labels.
- Throughput: capsules/sec ingestion and composition latency at various core counts (16, 32, 64, 128) with NUMA-aware pinning.

## 12. Reproducibility & Deployment

- Deterministic seeds: use UUID5 or stable hashing for deterministic IDs; avoid any time-seeded randomness in composition paths.
- Environment: document Python version, pip `requirements.txt`, NUMA affinity settings, and a `roca-start` launcher with profiles.
- Storage: use append-only capsule store (e.g., LevelDB / SQLite with WAL + immutable blobs) to maintain history and allow reproducible snapshots.
- CI: include unit tests for ingest/parsers, deterministic routing, and timeline transition evaluation.

## 13. Security, Ethics & Governance

- Provenance: preserve source attributions and timestamps for audit.
- Access control: enforce project-level policies (who can freeze or merge capsules into canonical libraries).
- Human-in-the-loop: require explicit acceptance for any automated commit to production assets.

## 14. Case Studies (short)

1. Animation Studio: ingest body-combat mocap BVH → create pose capsules → artist sketches keyframes → system matches sketches to pose capsules → timeline evaluates transitions → deterministic output used for final shot.
2. Style Preservation: ingest studio style guide capsules → new asset generation constrained by those capsules → exact reproducibility across render nodes.

## 15. Implementation Checklist (developer steps)

- [ ] Implement robust BVH parser + streaming motion ingest.
- [ ] Build Capsule Store with immutable append-only scheme.
- [ ] Implement Router with deterministic scoring and trace logging.
- [ ] Add UI: 2D canvas, timeline, chat collaborator (PyQt6 recommended for cross-platform desktop).
- [ ] Create GA module for capsule genome exploration (parallel evaluation harness).
- [ ] Add tests and performance harness for Threadripper/NUMA environments.

## 16. Example API (Python)

from roca import CapsuleStore, Router, Ingestor, AutonomousBrain

store = CapsuleStore(path='data/capsules')
ingestor = Ingestor(store)
brain = AutonomousBrain(store)
router = Router(store)

# Ingest a mocap file (deterministic capsule creation)
ingestor.ingest_bvh('data/walk.bvh')

# Compose for a timeline slot
context = {'project_id':'proj1','slot':'shot_001:frame_120','style_constraints':['cartoon','heroic']}
output, trace = router.compose(context)

## 17. Future Work

- Formalize capsule math for creativity (proofs around determinism and learning efficiency).
- Benchmarks for "10-year style retention" comparing ROCA vs fine-tuned LLMs.
- Secure, distributed capsule federation for multi-studio collaboration.
- Optional neuro-symbolic perception layer that remains strictly sandboxed (no stochastic contamination of core composition logic).

## 18. Conclusion

ROCA reframes capsule architectures as first-class primitives for professional creative intelligence. By prioritizing determinism, explainability, and append-only memory, ROCA addresses the core needs of studios and creative teams. This document gives engineers a practical starting point to implement, test, and adopt ROCA-style systems.

## References & Further Reading

- Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules.
- Threadripper-ROCA internal notes (2025–2026).
- BVH format references and forward-kinematics resources.

## Appendix A — Useful Snippets

- BVH parsing, deterministic UUID5 generation, timeline similarity metric sketches (see codebase `Docs` folder).

---

For next steps I can:
- polish `Rqhc.md` with inline code examples and figure placeholders,
- add a minimal runnable demo scaffold (PyQt6 canvas + ingest sample BVH),
- or run experiments on your machine to collect throughput numbers.

Which would you like next?

## Appendix B — Concrete APIs & Tests

This appendix provides concrete Python API sketches and example pytest tests to accelerate implementation and verification.

**API: capsule_store.py (conceptual)**

class CapsuleStore:
  def __init__(self, path: str):
    """Open or create append-only capsule store at `path`."""

  def put(self, capsule: dict) -> str:
    """Persist `capsule` dict and return deterministic `id` (UUID5)."""

  def get(self, capsule_id: str) -> dict:
    """Load capsule by id."""

  def query(self, *, type: str = None, meta: dict = None) -> list:
    """Simple filtered query returning capsule metadata list."""

**API: ingestor.py (conceptual)**

class Ingestor:
  def __init__(self, store: CapsuleStore):
    pass

  def ingest_bvh(self, path: str) -> dict:
    """Parse BVH and emit one or more `pose_capsule` objects (returns capsule metadata)."""

  def ingest_image(self, path: str) -> dict:
    """Optionally run OCR / sketch extraction and emit `frame` capsule."""

**API: router.py (conceptual)**

class Router:
  def __init__(self, store: CapsuleStore):
    pass

  def compose(self, context: dict, k: int = 5) -> tuple:
    """Deterministically select & compose top-k capsules for `context`. Returns (output, trace).
    Must be repeatable: same inputs -> same output + identical trace.
    """

**API: brain.py (conceptual)**

class AutonomousBrain:
  def __init__(self, store: CapsuleStore):
    pass

  def generate_thought(self, text: str, kind: str) -> None:
    """Record an insight/observation and optionally synthesize higher-order capsules."""

  def evaluate_transition(self, id_a: str, id_b: str) -> float:
    """Return similarity score in [0,1] and log reasoning."""

Example usage (sketch):

from capsule_store import CapsuleStore
from ingestor import Ingestor
from router import Router

store = CapsuleStore('data/capsules')
ing = Ingestor(store)
router = Router(store)

ing.ingest_bvh('assets/walk.bvh')
out, trace = router.compose({'project':'demo','slot':'shot01:frame120'})

**Requirements snippet**

Add to `requirements.txt` for tests and small utilities:

numpy
pytest
pyyaml
Pillow

Optional: PyQt6 for GUI demo.

**Pytest examples**

1) Deterministic routing test (tests/test_routing.py)

def test_router_determinism(tmp_path):
  store = CapsuleStore(str(tmp_path/'caps'))
  # insert deterministic sample capsules
  caps = [{'id': 'uuid5:a', 'type':'style','vector':[0.1,0.2]},{'id':'uuid5:b','type':'pose','vector':[0.9,0.8]}]
  for c in caps:
    store.put(c)
  router = Router(store)
  ctx = {'project':'x','slot':'s1','constraints':['style:cartoon']}
  out1, trace1 = router.compose(ctx)
  out2, trace2 = router.compose(ctx)
  assert out1 == out2
  assert trace1 == trace2

2) BVH ingest smoke test (tests/test_bvh.py)

def test_bvh_ingest(tmp_path, sample_bvh_file):
  store = CapsuleStore(str(tmp_path/'caps'))
  ing = Ingestor(store)
  meta = ing.ingest_bvh(sample_bvh_file)
  assert 'pose_capsule' in meta['type'] or meta['type']=='pose_capsule'
  loaded = store.get(meta['id'])
  assert 'base_pose' in loaded['content']

## Appendix C — AI-Powered Animation Features

This appendix summarizes practical features and integrations for combining AI capabilities with animation workflows.

- **AI-Powered Animation Tools:** A compact toolset for sketch-to-motion, motion retargeting, procedural secondary motion, style-transfer for motion, timeline-aware keyframe suggestion, and intelligent interpolation. Each tool emits deterministic candidate capsules (suggestion capsules) that contain provenance, confidence scores, and transform parameters for easy review and acceptance by artists.

- **Drawing analysis for intelligent suggestions:** Stroke- and gesture-aware analysis maps artist strokes to pose/keyframe candidates. The system performs stroke segmentation, temporal alignment, and visual similarity matching against `pose_capsule`s to propose precise keyframes, in-between frames, or corrective edits. Suggestions include confidence, reasoning trace, and quick-accept actions.

- **Creative idea generation based on context:** Context-aware generators produce shot-level and beat-level creative variants: staging notes, camera framing options, mood/color variants, cut-length proposals, and alternate blocking suggestions. Generations respect active project `style` capsules and recent user edits to remain stylistically consistent and reproducible.

- **12 Principles of Animation integration:** Encode the classical 12 principles as parameterized capsule templates so the Router can suggest principled edits. The principles include:
  1. Squash and Stretch
  2. Anticipation
  3. Staging
  4. Straight Ahead Action and Pose-to-Pose
  5. Follow Through and Overlapping Action
  6. Slow In and Slow Out
  7. Arcs
  8. Secondary Action
  9. Timing
  10. Exaggeration
  11. Solid Drawing (pose clarity)
  12. Appeal

  Each principle is represented as an annotation and a small transform-function (capsule) that can be applied as a non-destructive suggestion (e.g., apply `anticipation_capsule` before a primary keyframe). Suggestions are accompanied by visual previews and deterministic parameter seeds so artists can accept, tweak, or reject them while preserving full traceability.

- **Real-time collaboration between user and AI:** Live suggestion streaming allows artists to request interactive options (e.g., `/suggest beat`), receive ranked suggestion capsules in real time, and accept or lock suggestions into the timeline. Multi-user sync resolves conflicts deterministically (per-project policy capsules) and preserves per-user annotations. All actions produce append-only events for audit, rollback, and reproducible replay.

These features are designed to integrate with the ROCA Router and Capsule Store: all AI suggestions are capsules with provenance, deterministic ids, and composition traces so the studio retains full control and reproducibility.

3) Transition evaluator unit test (tests/test_transitions.py)

def test_transition_similarity(tmp_path):
  store = CapsuleStore(str(tmp_path/'caps'))
  brain = AutonomousBrain(store)
  # create two simple frame capsules differing slightly
  a = {'id':'a','type':'frame','data':[[0,0],[1,1]]}
  b = {'id':'b','type':'frame','data':[[0,0],[1.1,1.05]]}
  store.put(a); store.put(b)
  score = brain.evaluate_transition('a','b')
  assert 0.7 < score <= 1.0

**Run tests**

Install requirements and run pytest:

```bash
pip install -r requirements.txt
pytest -q
```

**Notes**

- Keep randomness out of core `Router.compose` and set deterministic tie-breakers (e.g., lexicographic capsule ids).
- For GA/evolution modules, control RNG with genome-derived seeds so runs can be replayed exactly.
- These tests are intentionally unit-level smoke tests; expand them to integration benchmarks on Threadripper hardware for throughput profiling.

---

Appendix B accelerates developer onboarding by providing concrete class/functional boundaries and minimal tests to verify critical properties (determinism, successful BVH ingest, and transition evaluation). Update or extract these into source files when you want runnable scaffolding.

### Appendix B.1 — NUMA store and parallel utilities (code sketches)

Below are compact, copy-paste-friendly sketches to accelerate a Threadripper-focused prototype. They are intentionally minimal and omit full error handling and persistence for clarity.

NUMA-aware capsule store (sketch):

```python
import hashlib
from typing import Optional

class NUMAOptimizedCapsuleStore:
  def __init__(self, node_stores: list):
    self.node_stores = node_stores  # list of CapsuleStore instances
    self.n_nodes = len(node_stores)

  def _node_for(self, capsule_id: str) -> int:
    h = int(hashlib.sha1(capsule_id.encode('utf8')).hexdigest(), 16)
    return h % self.n_nodes

  def get(self, capsule_id: str) -> Optional[dict]:
    node = self._node_for(capsule_id)
    c = self.node_stores[node].get(capsule_id)
    if c:
      return c
    # fallback: probe other nodes (cache on hit)
    for i, store in enumerate(self.node_stores):
      if i == node: continue
      c = store.get(capsule_id)
      if c:
        self.node_stores[node].put(c)
        return c
    return None

  def put(self, capsule: dict) -> str:
    capsule_id = capsule['id']
    node = self._node_for(capsule_id)
    return self.node_stores[node].put(capsule)
```

Parallel orbital update (Numba-style sketch):

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def parallel_orbital_update(positions, velocities, masses, dt):
  n = positions.shape[0]
  for i in prange(n):
    fx = 0.0; fy = 0.0; fz = 0.0
    xi, yi, zi = positions[i]
    for j in range(n):
      if i == j: continue
      xj, yj, zj = positions[j]
      dx = xj - xi; dy = yj - yi; dz = zj - zi
      r2 = dx*dx + dy*dy + dz*dz + 1e-6
      invr3 = 1.0 / (r2 * np.sqrt(r2))
      m = masses[j]
      fx += m * dx * invr3
      fy += m * dy * invr3
      fz += m * dz * invr3
    velocities[i,0] += fx * dt
    velocities[i,1] += fy * dt
    velocities[i,2] += fz * dt
    positions[i,0] += velocities[i,0] * dt
    positions[i,1] += velocities[i,1] * dt
    positions[i,2] += velocities[i,2] * dt
```

Real-time sync engine (concept sketch):

```python
from concurrent.futures import ProcessPoolExecutor

class RealTimeSyncEngine:
  def __init__(self, sync_workers: int):
    self.sync_workers = sync_workers

  def sync_user_actions(self, user_actions):
    regions = spatial_partition(user_actions, self.sync_workers)
    with ProcessPoolExecutor(max_workers=self.sync_workers) as ex:
      conflicts = list(ex.map(detect_region_conflicts, regions))
      # flatten and resolve in parallel
      flat = [c for region in conflicts for c in region]
      resolutions = list(ex.map(resolve_conflict, flat))
    return apply_resolutions(resolutions)
```

These sketches belong in `src/` when you convert prototypes into production modules. Add deterministic unit tests (Appendix B tests) that confirm repeatability on single-node and multi-process runs.