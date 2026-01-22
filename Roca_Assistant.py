#!/usr/bin/env python3
"""
ROCA - Enhanced PyQt6 AI Assistant with Orbital Visualization
Optimized with NumPy and enhanced with chatbot features
"""

import sys
import os
import json
import math
import random
import time
import threading
import queue
import hashlib
import psutil
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from pathlib import Path

# PyQt6 imports
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QSettings, QTimer, QPointF
from PyQt6.QtGui import (QFont, QIcon, QKeySequence, QPainter, QPen, QBrush, 
                         QColor, QAction, QPolygonF, QLinearGradient, QRadialGradient)
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSlider, QComboBox, QGroupBox, 
                            QTextEdit, QCheckBox, QSpinBox, QLineEdit, QSplitter,
                            QTabWidget, QStatusBar, QMenuBar, QFileDialog, QMessageBox,
                            QGridLayout, QProgressBar, QScrollArea, QFrame)

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Optional numba acceleration for pairwise physics
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    prange = range

# Try to import the autonomous brain (optional)
try:
    from Brain import autonomous_brain as autonomous_brain_mod
except Exception:
    autonomous_brain_mod = None
try:
    from Brain import ai_coding_assistant as ai_coding_assistant_mod
except Exception:
    ai_coding_assistant_mod = None
try:
    from Brain import enhanced_ai_assistant as enhanced_ai_assistant_mod
except Exception:
    enhanced_ai_assistant_mod = None
try:
    from Brain import Ai_fast as ai_fast_mod
except Exception:
    ai_fast_mod = None
try:
    from core_modules.Core_Modules import (
        PersonalitySystem, HierarchicalTemporalMemory, 
        IntrinsicMotivationSystem, KnowledgeNetwork,
        CausalDiscoveryModule, MetaLearner
    )
except Exception:
    PersonalitySystem = None
    HierarchicalTemporalMemory = None
    IntrinsicMotivationSystem = None
    KnowledgeNetwork = None
    CausalDiscoveryModule = None
    MetaLearner = None

# Optional voice imports
try:
    import speech_recognition as sr
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Unified voice controller (wraps VoiceSystem or VoiceListener)
try:
    from voice import VoiceController as VoiceController
except Exception:
    VoiceController = None

# Optional predictive creativity engine (heavy ML engine)
try:
    from roca_system.Predictive_Creativity_Engine import PredictiveCreativityEngine
except Exception:
    PredictiveCreativityEngine = None

# Optional router (hybrid symbolic + neural)
try:
    from Util.Router import HybridRouter
except Exception:
    try:
        # fallback to local utils implementation
        from utils.hybrid_router import HybridRouter  # type: ignore
    except Exception:
        HybridRouter = None

# Optional high-performance time capsule simulator
# Prefer the Threadripper-optimized implementation when available,
# fall back to the legacy Animation.TimeCapsule implementation.
try:
    from threadripper_core.Real_Time_Capsule_Galaxy import RealTimeCapsuleGalaxy
except Exception:
    try:
        from Animation.TimeCapsule import RealTimeCapsuleGalaxy
    except Exception:
        RealTimeCapsuleGalaxy = None

# Optional performance tuning utilities
try:
    from utils.performance_tuning import PerformanceTuner
except Exception:
    PerformanceTuner = None

# Optional learning orbital visualizer
try:
    from roca_system.OrbitalVisualizer import LearningOrbitalVisualizer
except Exception:
    LearningOrbitalVisualizer = None

# Optional shader-based OpenGL visualizer (high-fidelity)
try:
    from utils.opengl_renderer import ShaderVisualizerWidget, USE_GL
except Exception:
    ShaderVisualizerWidget = None
    USE_GL = False
# Optional Unified system integration (UniversalCapsule)
try:
    from unified_system.universal_capsule import UniversalCapsule
except Exception:
    UniversalCapsule = None

def create_universal_capsule(id: str, kind: str, **kwargs):
    """Factory for creating a `UniversalCapsule` if available."""
    if UniversalCapsule is None:
        raise ImportError("UniversalCapsule not available. Ensure unified_system package is present.")
    return UniversalCapsule(id=id, kind=kind, **kwargs)

def serialize_universal_capsule(cap) -> Dict:
    """Serialize a UniversalCapsule to a JSON-serializable dict."""
    return {
        'id': cap.id,
        'kind': cap.kind,
        'embedding': cap.embedding.tolist(),
        'semantic_context': cap.semantic_context,
        'code_manifest': cap.code_manifest,
        'orbit_score': cap.orbit_score,
        'orbit_lane': cap.orbit_lane,
        'gravity_well': cap.gravity_well,
        'growth_stage': cap.growth_stage,
        'learning_rate': cap.learning_rate,
        'curiosity_level': cap.curiosity_level,
        'innovation_score': cap.innovation_score,
        'can_expand': cap.can_expand,
        'generated_children': cap.generated_children,
        'creation_epoch': cap.creation_epoch,
        'evolution_path': cap.evolution_path,
        'future_potential': cap.future_potential,
        'shadows': cap.shadows,
        'merged_into': cap.merged_into,
        'merge_confidence': cap.merge_confidence,
        'use_count': getattr(cap, 'use_count', 0),
        'last_used_at': getattr(cap, 'last_used_at', None).isoformat() if getattr(cap, 'last_used_at', None) else None,
    }

def save_universal_capsule_json(cap, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialize_universal_capsule(cap), f, indent=2)

def load_universal_capsule_json(path: str):
    d = json.load(open(path, 'r', encoding='utf-8'))
    if UniversalCapsule is None:
        raise ImportError("UniversalCapsule not available. Ensure unified_system package is present.")
    emb = np.asarray(d.get('embedding', [0] * 64), dtype=float)
    cap = UniversalCapsule(id=d['id'], kind=d['kind'], embedding=emb, semantic_context=d.get('semantic_context', {}), code_manifest=d.get('code_manifest', ''))
    cap.use_count = d.get('use_count', 0)
    last_used = d.get('last_used_at')
    if last_used:
        try:
            cap.last_used_at = datetime.fromisoformat(last_used)
        except Exception:
            cap.last_used_at = None
    return cap

# ============================================================================
# NUMPY ENHANCED SETTINGS
# ============================================================================

class NumpySettings:
    """Persistent settings manager optimized with numpy operations"""
    
    def __init__(self):
        self.settings = QSettings("ROCA_NUMPY", "AI_Assistant")
        self.defaults = self._create_defaults()
        self.cached_settings = {}
        
    def _create_defaults(self):
        """Create default settings with numpy-friendly values"""
        return {
            # UI settings
            'window_geometry': None,
            'window_state': None,
            'theme': 'dark',
            'font_size': 11,
            'chat_font': 'Consolas',
            
            # Orbital visualization with numpy defaults
            'orbital_auto_rotate': True,
            'orbital_speed': np.float32(30.0),
            'orbital_radius': np.float32(80.0),
            'show_capsule_labels': True,
            'show_orbit_paths': True,
            'zoom_level': np.float32(1.0),
            'physics_enabled': True,
            'attraction_strength': np.float32(0.1),
            'repulsion_strength': np.float32(0.2),
            
            # Voice settings
            'voice_enabled': True,
            'auto_listen': False,
            'save_recordings': False,
            'speech_rate': 150,
            'voice_volume': 0.9,
            'voice_gender': 'female',
            
            # AI settings
            'ai_model': 'capsule_network',
            'response_length': 'medium',
            'temperature': np.float32(0.7),
            'use_context': True,
            
            # Performance (numpy optimized)
            'max_capsules': 1000,
            'auto_save_interval': 300,
            'enable_animations': True,
            'update_rate': 60.0,
            'parallel_processing': True,
            
            # Shortcuts
            'shortcuts': {
                'new_session': 'Ctrl+N',
                'open_session': 'Ctrl+O',
                'save_session': 'Ctrl+S',
                'voice_toggle': 'Ctrl+V',
                'orbital_toggle': 'Ctrl+O',
                'clear_chat': 'Ctrl+L',
                'export_data': 'Ctrl+E',
                'add_capsule': 'Ctrl+A',
                'chat_focus': 'Ctrl+T',
            }
        }
    
    def get_numpy(self, key, default=None):
        """Get setting with numpy type preservation"""
        if default is None:
            default = self.defaults.get(key)
        
        value = self.settings.value(key, default)
        
        # Convert to numpy types if needed
        if isinstance(default, np.floating):
            return np.float32(float(value))
        elif isinstance(default, np.integer):
            return np.int32(int(value))
        
        return value
    
    def set_numpy(self, key, value):
        """Set numpy-compatible settings"""
        if isinstance(value, np.generic):
            value = value.item()  # Convert numpy scalar to Python type
        
        self.settings.setValue(key, value)
        self.settings.sync()
        if key in self.cached_settings:
            del self.cached_settings[key]
    
    def get_optimal_thread_count(self, task_type="graph_analysis"):
        """Calculate optimal thread count based on system resources"""
        cpu_count = os.cpu_count()
        if not cpu_count:
            return 4
        
        if task_type == "graph_analysis":
            return min(32, cpu_count * 2)
        elif task_type == "physics":
            return min(16, cpu_count)
        else:
            return min(8, max(2, cpu_count // 2))

# ============================================================================
# NUMPY ENHANCED DATA CLASSES
# ============================================================================

@dataclass
class NumpyCapsule:
    """Capsule optimized with numpy arrays"""
    content: str
    kind: str = "concept"
    certainty: np.float32 = np.float32(0.6)
    orbit_radius: np.float32 = np.float32(1.0)
    angle: np.float32 = np.float32(0.0)
    character: Optional[str] = None
    id: str = field(default_factory=lambda: str(random.randint(100000, 999999)))
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Numpy arrays for physics
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    
    # Visualization properties
    color: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 1.0]))
    size: np.float32 = np.float32(16.0)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.character is None:
            self.character = self.content[:3].upper() if len(self.content) >= 3 else "CAP"
        
        # Initialize color based on kind
        color_map = {
            "theory": np.array([0.8, 0.4, 1.0]),
            "hypothesis": np.array([1.0, 0.8, 0.4]),
            "method": np.array([0.4, 0.8, 1.0]),
            "concept": np.array([0.6, 1.0, 0.6]),
            "data": np.array([1.0, 0.6, 0.6]),
            "model": np.array([0.6, 0.8, 1.0]),
        }
        self.color = color_map.get(self.kind, np.array([0.7, 0.7, 1.0]))
        
    def __len__(self):
        """Return the length of the capsule content"""
        return len(self.content)
    
    def __getitem__(self, key):
        """Allow dictionary-like access to capsule attributes"""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{key}' not found in capsule")
    
    def __setitem__(self, key, value):
        """Allow dictionary-like assignment to capsule attributes"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{key}' not found in capsule")
    
    def to_numpy_dict(self):
        """Convert to dict with numpy arrays"""
        return {
            'id': self.id,
            'content': self.content,
            'kind': self.kind,
            'certainty': float(self.certainty),
            'orbit_radius': float(self.orbit_radius),
            'angle': float(self.angle),
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'color': self.color.tolist(),
            'size': float(self.size),
            'created_at': self.created_at,
            'connections': self.connections
        }
    
    def update_position(self, center: np.ndarray, zoom: float, rotation: float):
        """Update position using numpy operations"""
        self.position[0] = center[0] + self.orbit_radius * 50 * zoom * np.cos(self.angle + rotation)
        self.position[1] = center[1] + self.orbit_radius * 50 * zoom * np.sin(self.angle + rotation)


"""Shim for `CapsuleGenomeSequencer`.

This module delegates to the canonical implementation in
`threadripper_core.CapsuleGenomeSequencer` when available. It exposes a
compatibility class `CapsuleGenomeSequencer` that provides the two main
APIs used across the codebase: `sequence_capsule_genome(...)` and a small
compatibility layer providing `encode`, `genome_to_params`, `mutate`, and
`crossover` (fallback implementations) so existing callers remain functional.
"""
import json
import hashlib
import numpy as np

try:
    from threadripper_core.CapsuleGenomeSequencer import CapsuleGenomeSequencer as _TRCapsuleGenomeSequencer
except Exception:
    _TRCapsuleGenomeSequencer = None


class CapsuleGenomeSequencer:
    def __init__(self, *args, **kwargs):
        self._tr = None
        if _TRCapsuleGenomeSequencer is not None:
            try:
                self._tr = _TRCapsuleGenomeSequencer(*args, **kwargs)
            except Exception:
                self._tr = None

    def sequence_capsule_genome(self, capsule):
        if self._tr is not None and hasattr(self._tr, 'sequence_capsule_genome'):
            try:
                return self._tr.sequence_capsule_genome(capsule)
            except Exception:
                pass
        # fallback: minimal analysis
        try:
            path = getattr(capsule, 'evolution_path', []) or []
            genes = []
            for item in path:
                genes.append({'marker_id': getattr(item, 'id', None) if not isinstance(item, dict) else item.get('id'), 'ts': getattr(item, 'ts', None) if not isinstance(item, dict) else item.get('ts', None)})
            genome_map = {g.get('marker_id') or '_anon_': [g for g in genes if (g.get('marker_id') or '_anon_') == (g.get('marker_id') or '_anon_')]} if genes else {}
            return {'genome': genome_map, 'mutation_points': [], 'evolution_potential': [], 'compatibility_matrix': {}}
        except Exception:
            return {'genome': {}, 'mutation_points': [], 'evolution_potential': [], 'compatibility_matrix': {}}

    # Compatibility helpers (keep original behavior)
    def _seed_from_capsule(self, capsule):
        meta = getattr(capsule, 'metadata', None) or {}
        meta_str = json.dumps(meta, sort_keys=True)
        seed = f"{getattr(capsule, 'content', '')}|{getattr(capsule, 'kind', '')}|{getattr(capsule, 'created_at', '')}|{meta_str}"
        return seed

    def encode(self, capsule):
        try:
            seed = self._seed_from_capsule(capsule)
            return hashlib.sha256(seed.encode('utf-8')).hexdigest()
        except Exception:
            return ''

    def genome_to_params(self, genome_hex):
        try:
            bigint = int(genome_hex, 16)
            certainty = 0.1 + (bigint % 1000) / 1000.0 * 0.9
            bigint //= 1000
            size = 8.0 + (bigint % 1000) / 1000.0 * 24.0
            bigint //= 1000
            orbit_radius = 0.5 + (bigint % 1000) / 1000.0 * 3.0
            bigint //= 1000
            r = (bigint % 1000) / 1000.0
            bigint //= 1000
            g = (bigint % 1000) / 1000.0
            bigint //= 1000
            b = (bigint % 1000) / 1000.0
            return {'certainty': float(certainty), 'size': float(size), 'orbit_radius': float(orbit_radius), 'color': [float(r), float(g), float(b)]}
        except Exception:
            return {'certainty': 0.6, 'size': 12.0, 'orbit_radius': 1.0, 'color': [0.5, 0.5, 0.5]}

    def mutate(self, genome_hex, rate=0.01):
        try:
            b = bytearray.fromhex(genome_hex)
            nbits = len(b) * 8
            nflip = max(1, int(nbits * rate))
            rnd = np.random.randint(0, nbits, size=nflip)
            for bit in rnd:
                idx = bit // 8
                off = bit % 8
                b[idx] ^= (1 << off)
            return bytes(b).hex()
        except Exception:
            return genome_hex

    def crossover(self, a_hex, b_hex):
        try:
            a = bytearray.fromhex(a_hex)
            b = bytearray.fromhex(b_hex)
            if len(a) != len(b):
                L = max(len(a), len(b))
                a = a.ljust(L, b'\x00')
                b = b.ljust(L, b'\x00')
            pivot = np.random.randint(1, len(a)) if len(a) > 1 else 0
            child = a[:pivot] + b[pivot:]
            return bytes(child).hex()
        except Exception:
            return a_hex


class CapsuleStore:
    """Simple persistent capsule store for NumpyCapsule objects.

    Stores capsules in-memory and can save/load to JSON files using the
    capsule's `to_numpy_dict` representation.
    """
    def __init__(self):
        self.capsules: Dict[str, NumpyCapsule] = {}
        # Store for UniversalCapsule instances (if used)
        try:
            self.universal_capsules: Dict[str, Any] = {}
        except Exception:
            self.universal_capsules = {}

    def add(self, capsule: NumpyCapsule):
        self.capsules[capsule.id] = capsule

    def remove(self, capsule_id: str):
        if capsule_id in self.capsules:
            del self.capsules[capsule_id]

    def get(self, capsule_id: str) -> Optional[NumpyCapsule]:
        return self.capsules.get(capsule_id)

    def list_all(self) -> List[NumpyCapsule]:
        return list(self.capsules.values())

    def to_dict(self) -> Dict[str, Any]:
        return {cid: c.to_numpy_dict() for cid, c in self.capsules.items()}

    # --- UniversalCapsule persistence helpers ---
    def add_universal(self, capsule) -> None:
        """Add a UniversalCapsule instance to the store."""
        try:
            cid = getattr(capsule, 'id', None)
            if cid is None:
                raise ValueError('Universal capsule must have an id')
            self.universal_capsules[cid] = capsule
        except Exception:
            pass

    def remove_universal(self, capsule_id: str) -> None:
        try:
            if capsule_id in self.universal_capsules:
                del self.universal_capsules[capsule_id]
        except Exception:
            pass

    def get_universal(self, capsule_id: str):
        return self.universal_capsules.get(capsule_id)

    def list_universal(self) -> List[Any]:
        return list(self.universal_capsules.values())

    def save_universal(self, path: str) -> None:
        """Save all UniversalCapsule instances to a single JSON file."""
        try:
            data = {}
            for cid, cap in self.universal_capsules.items():
                try:
                    data[cid] = serialize_universal_capsule(cap)
                except Exception:
                    # attempt best-effort shallow serialization
                    data[cid] = {
                        'id': getattr(cap, 'id', cid),
                        'kind': getattr(cap, 'kind', None),
                    }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_universal(self, path: str) -> None:
        """Load UniversalCapsule instances from a JSON file saved by `save_universal`.

        Existing entries are replaced.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return

        self.universal_capsules = {}
        for cid, d in data.items():
            try:
                # If helper exists, prefer loader
                if 'id' in d and 'kind' in d and UniversalCapsule is not None:
                    emb = np.asarray(d.get('embedding', [0] * 64), dtype=float)
                    cap = UniversalCapsule(id=d['id'], kind=d['kind'], embedding=emb, semantic_context=d.get('semantic_context', {}), code_manifest=d.get('code_manifest', ''))
                    cap.use_count = d.get('use_count', 0)
                    last_used = d.get('last_used_at')
                    if last_used:
                        try:
                            cap.last_used_at = datetime.fromisoformat(last_used)
                        except Exception:
                            cap.last_used_at = None
                    # copy other simple fields
                    cap.growth_stage = d.get('growth_stage', cap.growth_stage)
                    cap.learning_rate = float(d.get('learning_rate', cap.learning_rate))
                    cap.curiosity_level = float(d.get('curiosity_level', cap.curiosity_level))
                    cap.generated_children = d.get('generated_children', [])
                    cap.evolution_path = d.get('evolution_path', [])
                    self.universal_capsules[cid] = cap
                else:
                    # fallback: store raw dict
                    self.universal_capsules[cid] = d
            except Exception:
                continue

    def save(self, path: str):
        data = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.capsules = {}
        for cid, d in data.items():
            cap = NumpyCapsule(
                content=d.get('content', ''),
                kind=d.get('kind', 'concept'),
                certainty=np.float32(d.get('certainty', 0.6)),
                orbit_radius=np.float32(d.get('orbit_radius', 1.0)),
                angle=np.float32(d.get('angle', 0.0))
            )
            # restore arrays
            pos = d.get('position')
            vel = d.get('velocity')
            col = d.get('color')
            if pos is not None and len(pos) >= 2:
                cap.position = np.array(pos, dtype=np.float32)
            if vel is not None and len(vel) >= 2:
                cap.velocity = np.array(vel, dtype=np.float32)
            if col is not None and len(col) >= 3:
                cap.color = np.array(col, dtype=np.float32)
            cap.size = np.float32(d.get('size', cap.size))
            cap.metadata = d.get('metadata', {})
            self.capsules[cid] = cap


class CosmicCreationFlow:
    """Controller for the 'Cosmic Creation' end-to-end flow.

    It wires together the orbital widget, capsule store, genome sequencer,
    and optional engines (growth, expansion, router). For now, engines are
    optional and default to simple stubs.
    """
    def __init__(self, widget, store=None, sequencer=None, growth_engine=None, expansion_engine=None, router=None):
        self.widget = widget
        self.store = store or getattr(widget, 'store', None)
        self.sequencer = sequencer or getattr(widget, 'genome_sequencer', None)
        self.growth_engine = growth_engine
        self.expansion_engine = expansion_engine
        self.router = router

    def run(self, content: str, kind: str = 'artifact') -> Dict[str, Any]:
        """Run the flow: create capsule, persist, learn style, suggest animations.

        Returns a result dict with `capsule` and `suggestions`.
        """
        # Create capsule
        cap = NumpyCapsule(content=content, kind=kind,
                           certainty=np.float32(0.7),
                           orbit_radius=np.float32(np.random.uniform(0.8, 2.5)),
                           angle=np.float32(np.random.uniform(0, 2 * np.pi)))

        # Add to widget/store
        try:
            self.widget.add_capsule_object(cap)
            if self.store is not None:
                self.store.add(cap)
        except Exception:
            pass

        # Perpetual learning: growth engine stub
        try:
            if self.growth_engine is not None and hasattr(self.growth_engine, 'process_style_pattern'):
                self.growth_engine.process_style_pattern(content)
        except Exception:
            pass

        # Self-expansion: expansion_engine stub
        expansion_result = None
        try:
            if self.expansion_engine is not None and hasattr(self.expansion_engine, 'detects_capability_gap'):
                if self.expansion_engine.detects_capability_gap(content):
                    expansion_result = self.expansion_engine.generate_drawing_tool(content)
        except Exception:
            expansion_result = None

        # Routing: suggest animations (stub if router not provided)
        suggestions = None
        try:
            if self.router is not None and hasattr(self.router, 'suggest_animations'):
                suggestions = self.router.suggest_animations(cap)
            else:
                # simple heuristic suggestions
                suggestions = ['fade-in', 'orbit-spin', 'pulse']
        except Exception:
            suggestions = ['fade-in']

        # Visualizer: highlight capsule and show suggestions via chat
        try:
            # select and focus capsule in widget
            self.widget.selected_capsule = cap
            try:
                self.widget.capsule_selected.emit(cap)
            except Exception:
                pass
        except Exception:
            pass

        # Return an outcome
        return {'capsule': cap, 'suggestions': suggestions, 'expansion': expansion_result}


class StyleTransferController:
    """Wrapper to run style transfer on capsules.

    Tries to use the high-performance `ParallelStyleTransfer` from
    `Animation.Style_Transfer` when available; otherwise falls back to a
    lightweight color/size mapping heuristic so the GUI remains functional
    without heavy ML dependencies.
    """
    def __init__(self):
        self.backend = None
        try:
            from Animation.Style_Transfer import ParallelStyleTransfer
            self.backend = ParallelStyleTransfer()
        except Exception:
            self.backend = None

    def apply_style_to_capsule(self, capsule: NumpyCapsule, style: str):
        if self.backend is not None:
            try:
                # backend expected to accept list-like capsule descriptors
                res = self.backend.batch_style_transfer([capsule], style)
                # backend may return styled capsule-like objects; try to copy color/size
                if len(res) > 0:
                    styled = res[0]
                    if hasattr(styled, 'color'):
                        capsule.color = np.array(styled.color, dtype=np.float32)
                    if hasattr(styled, 'size'):
                        capsule.size = np.float32(styled.size)
                return True
            except Exception:
                pass

        # Fallback heuristic mapping from style name to color/size
        try:
            style = (style or '').lower()
            if 'dark' in style:
                capsule.color = np.array([0.2, 0.2, 0.3], dtype=np.float32)
                capsule.size *= 0.9
            elif 'vibrant' in style or 'pop' in style:
                capsule.color = np.array([1.0, 0.6, 0.2], dtype=np.float32)
                capsule.size *= 1.2
            elif 'pastel' in style:
                capsule.color = np.array([0.8, 0.7, 0.9], dtype=np.float32)
                capsule.size *= 1.0
            elif 'mono' in style or 'grayscale' in style:
                avg = float(np.mean(capsule.color))
                capsule.color = np.array([avg, avg, avg], dtype=np.float32)
                capsule.size *= 0.95
            else:
                # subtle tint based on hash
                h = int(hashlib.sha256(style.encode('utf-8')).hexdigest()[:6], 16)
                r = ((h >> 16) & 0xFF) / 255.0
                g = ((h >> 8) & 0xFF) / 255.0
                b = (h & 0xFF) / 255.0
                capsule.color = np.array([r, g, b], dtype=np.float32)
            return True
        except Exception:
            return False

    def apply_style_to_all(self, capsules: List[NumpyCapsule], style: str):
        if self.backend is not None:
            try:
                res = self.backend.batch_style_transfer(capsules, style)
                # If backend returns array-like descriptors, attempt to apply
                for i, styled in enumerate(res):
                    if i >= len(capsules):
                        break
                    cap = capsules[i]
                    if hasattr(styled, 'color'):
                        cap.color = np.array(styled.color, dtype=np.float32)
                    if hasattr(styled, 'size'):
                        cap.size = np.float32(styled.size)
                return True
            except Exception:
                pass

        # Fallback: map style to all capsules deterministically
        for cap in capsules:
            try:
                self.apply_style_to_capsule(cap, style)
            except Exception:
                continue
        return True


class PredictiveReasoner:
    """Simple predictive reasoning engine for capsule states.

    Provides deterministic, lightweight prediction methods that can be
    used when heavy ML engines are not available. The default predictor
    performs linear extrapolation using capsule velocity and optional
    acceleration if present.
    """
    def __init__(self, model=None, use_creative_engine: bool = True):
        self.model = model
        self.engine = None
        # If the heavier predictive creativity engine is available and requested,
        # instantiate it and use for richer predictions.
        if use_creative_engine and PredictiveCreativityEngine is not None:
            try:
                self.engine = PredictiveCreativityEngine()
            except Exception:
                self.engine = None

    def predict_positions(self, capsules: List[NumpyCapsule], dt: float = 1/60.0, steps: int = 60) -> np.ndarray:
        """Predict positions for each capsule `steps` steps into the future.

        Returns an (n,2) array of predicted positions at t = dt * steps.
        """
        n = len(capsules)
        preds = np.zeros((n, 2), dtype=np.float64)
        if n == 0:
            return preds

        t = dt * steps
        for i, cap in enumerate(capsules):
            pos = np.array(cap.position, dtype=np.float64)
            vel = np.array(cap.velocity, dtype=np.float64)
            # if capsule has an acceleration array, use it; else assume zero
            try:
                acc = np.array(getattr(cap, 'acceleration', np.zeros(2)), dtype=np.float64)
            except Exception:
                acc = np.zeros(2, dtype=np.float64)

            # simple kinematic prediction: p + v*t + 0.5*a*t^2
            preds[i, :] = pos + vel * t + 0.5 * acc * (t * t)

        return preds

    def predict_paths(self, capsules: List[NumpyCapsule], dt: float = 1/60.0, steps: int = 60) -> np.ndarray:
        """Return full paths: shape (n, steps, 2) with successive predicted positions."""
        # If the creative engine is available, attempt to use it for predictions
        if self.engine is not None:
            try:
                # Prepare lightweight context for the engine
                user_history = None
                try:
                    # If running within the app, some callers pass a ChatBot; we accept None here
                    user_history = getattr(self, 'chat_history', None)
                except Exception:
                    user_history = None

                current_context = {
                    'capsules': [
                        {
                            'position': np.array(cap.position, dtype=np.float64).tolist(),
                            'velocity': np.array(cap.velocity, dtype=np.float64).tolist(),
                            'acceleration': (np.array(getattr(cap, 'acceleration', np.zeros(2)), dtype=np.float64)).tolist(),
                        }
                        for cap in capsules
                    ]
                }

                res = self.engine.predict_creative_needs(user_history, current_context)
                # The engine may return pre-generated capsules or direct predicted positions
                if isinstance(res, dict):
                    # Try to extract final positions from pre_generated_capsules
                    pre = res.get('pre_generated_capsules')
                    if pre and isinstance(pre, list):
                        # Map any capsule-like dicts to positions
                        pts = []
                        for item in pre:
                            if isinstance(item, dict) and 'position' in item:
                                pts.append(np.array(item['position'], dtype=np.float64))
                        if len(pts) == len(capsules):
                            return np.stack([p for p in pts], axis=0).reshape((len(pts), 1, 2))
                    # Otherwise attempt to use predicted_needs -> not mappable; fall back
            except Exception:
                # Fall through to deterministic predictor on any engine error
                pass

        # Deterministic fallback: compute full paths via semi-implicit Euler
        n = len(capsules)
        if n == 0:
            return np.zeros((0, steps, 2), dtype=np.float64)

        paths = np.zeros((n, steps, 2), dtype=np.float64)
        for i, cap in enumerate(capsules):
            pos = np.array(cap.position, dtype=np.float64)
            vel = np.array(cap.velocity, dtype=np.float64)
            try:
                acc = np.array(getattr(cap, 'acceleration', np.zeros(2)), dtype=np.float64)
            except Exception:
                acc = np.zeros(2, dtype=np.float64)

            p = pos.copy()
            v = vel.copy()
            for s in range(steps):
                # semi-implicit Euler step for stable propagation
                v = v + acc * dt
                p = p + v * dt
                paths[i, s, :] = p

        return paths


class RouterController:
    """Lightweight wrapper around the optional `HybridRouter`.

    Provides a safe API for `Roca_Assistant` to use routing when available
    and a deterministic fallback when not.
    """
    def __init__(self, router=None):
        """Initialize RouterController. Optionally accept an injected router instance."""
        self.router = None
        # If an explicit router is provided, use it
        if router is not None:
            self.router = router
            return

        if HybridRouter is not None:
            try:
                self.router = HybridRouter()
            except Exception:
                self.router = None

    def is_available(self) -> bool:
        return self.router is not None

    def route(self, request: dict) -> dict:
        """Route a `request` dict and return a dict-like result.

        Falls back to a simple deterministic response when the heavy router
        is unavailable or errors occur.
        """
        if self.router is None:
            return {'path': [], 'fallback': True}

        try:
            # The external router may expect a specific RoutingRequest type;
            # attempt to call with the dict and fall back to a safe response.
            res = self.router.route(request)
            return res
        except Exception:
            return {'path': [], 'error': True}


class TimeCapsuleController:
    """Wrapper for optional `RealTimeCapsuleGalaxy` simulator and snapshot persistence."""
    def __init__(self):
        self.sim = None
        if RealTimeCapsuleGalaxy is not None:
            try:
                self.sim = RealTimeCapsuleGalaxy()
            except Exception:
                self.sim = None

        # directory to store capsules
        self.store_dir = os.path.join(os.getcwd(), 'time_capsules')
        try:
            os.makedirs(self.store_dir, exist_ok=True)
        except Exception:
            pass

    def is_available(self) -> bool:
        return self.sim is not None

    def simulate(self, capsules):
        """Run the high-performance simulation if available, else return None."""
        if self.sim is None:
            return None
        try:
            return self.sim.simulate_galaxy(capsules)
        except Exception:
            return None

    def save_time_capsule(self, name: str, description: str, data: dict) -> str:
        """Save a snapshot to JSON and return the path."""
        safe_name = ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name.replace(' ', '_')}.json"
        path = os.path.join(self.store_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'name': name, 'description': description, 'data': data, 'created': datetime.now().isoformat()}, f, indent=2)
            return path
        except Exception:
            return ''

    def list_time_capsules(self):
        try:
            return [f for f in os.listdir(self.store_dir) if f.endswith('.json')]
        except Exception:
            return []

    def load_time_capsule(self, filename: str) -> dict:
        path = os.path.join(self.store_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}


class LearningOrbitalVisualizerController:
    """Wrapper for optional `LearningOrbitalVisualizer` used to draw enhanced orbital views."""
    def __init__(self):
        self.vis = None
        if LearningOrbitalVisualizer is not None:
            try:
                self.vis = LearningOrbitalVisualizer()
            except Exception:
                self.vis = None

    def is_available(self) -> bool:
        return self.vis is not None

    def draw(self, capsules):
        """Invoke the visualizer's drawing method with a list of capsules.

        Returns True on success, False otherwise.
        """
        if self.vis is None:
            return False
        try:
            # Assume capsules is a list-like collection compatible with the visualizer
            self.vis.draw_cosmic_view(capsules)
            return True
        except Exception:
            return False

    def attach_widget(self, widget):
        """Attempt to attach the external visualizer to a Qt widget/canvas.

        This will try several common adapter methods (`set_target_widget`,
        `attach_to_widget`, `set_canvas`) and otherwise store the widget for
        potential use by the external visualizer.
        """
        if self.vis is None:
            return False


class PerformanceTuningController:
    """Wraps `PerformanceTuner` and exposes simple UI-friendly methods."""
    def __init__(self):
        self.tuner = None
        if PerformanceTuner is not None:
            try:
                self.tuner = PerformanceTuner()
            except Exception:
                self.tuner = None
        self.auto_tune = False

    def is_available(self) -> bool:
        return self.tuner is not None

    def optimize_system(self):
        try:
            info = {
                'cores': psutil.cpu_count(logical=False) or psutil.cpu_count()
                , 'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
            if self.tuner is not None:
                self.tuner.optimize_system_settings(info)
                return self.tuner.current_settings
        except Exception:
            pass
        return {}

    def adjust_based_on_metrics(self, metrics: dict):
        try:
            if self.tuner is not None:
                self.tuner.adjust_based_on_metrics(metrics)
                return self.tuner.current_settings
        except Exception:
            pass
        return {}
        try:
            if hasattr(self.vis, 'set_target_widget'):
                try:
                    self.vis.set_target_widget(widget)
                    return True
                except Exception:
                    pass

            if hasattr(self.vis, 'attach_to_widget'):
                try:
                    self.vis.attach_to_widget(widget)
                    return True
                except Exception:
                    pass

            if hasattr(self.vis, 'set_canvas'):
                try:
                    self.vis.set_canvas(widget)
                    return True
                except Exception:
                    pass

            # Last resort: remember widget so draw() can attempt to accept it
            self._attached_widget = widget
            return True
        except Exception:
            return False


class LearningVisualizerPane(QtWidgets.QWidget):
    capsule_selected = QtCore.pyqtSignal(object)
    """Embedded pane that renders a lightweight learning orbital visualization.

    Uses the optional `LearningOrbitalVisualizer` if available, otherwise falls
    back to a matplotlib scatter visualization of capsule positions.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # Use a 3D axes for richer orbital visualization (falls back if not available)
        try:
            self.ax = self.figure.add_subplot(111, projection='3d')
            self._is_3d = True
        except Exception:
            self.ax = self.figure.add_subplot(111)
            self._is_3d = False

        # Connect matplotlib pick event for capsule selection
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self._capsule_id_map = {}

    def update(self, capsules):
        # Build a map from scatter plot index to capsule object for selection
        self._capsule_id_map = {}
        try:
            self.ax.clear()
            if not capsules:
                self.ax.set_title('No capsules')
                self.canvas.draw()
                return

            # Build position map and plotting lists
            id_pos = {}
            xs = []
            ys = []
            zs = []
            sizes = []
            cols = []

            for idx, c in enumerate(capsules):
                try:
                    # prefer explicit 2D position; otherwise fall back to orbital projection
                    pos = getattr(c, 'position', None)
                    if pos is None:
                        x, y = self._project_orbit(getattr(c, 'orbit_score', 0.5), getattr(c, 'orbit_lane', 1), getattr(c, 'id', str(id(c))))
                    else:
                        arr = np.array(pos, dtype=np.float64)
                        x, y = float(arr[0]), float(arr[1])

                    z = float(getattr(c, 'innovation_score', getattr(c, 'learning_rate', 0.0)))
                    cid = getattr(c, 'id', str(id(c)))
                    id_pos[cid] = (x, y, z)

                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    sizes.append(float(getattr(c, 'size', 4.0)) * 12.0)
                    self._capsule_id_map[idx] = c

                    col = getattr(c, 'color', None)
                    if col is None:
                        cols.append('white')
                    else:
                        try:
                            arr = np.array(col, dtype=np.float32)
                            cols.append((float(arr[0]), float(arr[1]), float(arr[2])))
                        except Exception:
                            cols.append('white')
                except Exception:
                    continue

            # Draw 3D scatter if axes support it; otherwise 2D scatter
            try:
                if getattr(self, '_is_3d', False):
                    scatter = self.ax.scatter(xs, ys, zs, s=sizes, c=cols, alpha=0.9, depthshade=True, picker=True)
                    self.ax.set_zlabel('Learning Depth')
                else:
                    scatter = self.ax.scatter(xs, ys, s=sizes, c=cols, alpha=0.9, picker=True)
                self._scatter = scatter
                def _on_pick(self, event):
                    # Called when a capsule is clicked in the plot
                    ind = event.ind[0] if hasattr(event, 'ind') and event.ind else None
                    if ind is not None and ind in self._capsule_id_map:
                        capsule = self._capsule_id_map[ind]
                        self.capsule_selected.emit(capsule)
            except Exception:
                # fallback to 2D scatter with no z
                self.ax.scatter(xs, ys, s=sizes, c=cols, alpha=0.9)

            # Draw tendrils between known ids
            for c in capsules:
                edges = getattr(c, 'edges', []) or []
                src_id = getattr(c, 'id', str(id(c)))
                src_pos = id_pos.get(src_id)
                for e in edges:
                    tgt = e.get('target') if isinstance(e, dict) else getattr(e, 'target', None)
                    strength = e.get('strength', 1.0) if isinstance(e, dict) else getattr(e, 'strength', 1.0)
                    tgt_id = tgt if isinstance(tgt, str) else getattr(tgt, 'id', None)
                    if src_pos and tgt_id and tgt_id in id_pos and getattr(self, '_is_3d', False):
                        dst = id_pos.get(tgt_id)
                        self.ax.plot([src_pos[0], dst[0]], [src_pos[1], dst[1]], [src_pos[2], dst[2]], color='gray', alpha=min(0.9, 0.2 + strength * 0.8))
                    elif src_pos and tgt_id and tgt_id in id_pos:
                        dst = id_pos.get(tgt_id)
                        self.ax.plot([src_pos[0], dst[0]], [src_pos[1], dst[1]], color='gray', alpha=min(0.9, 0.2 + strength * 0.8))

            self.ax.set_title('Learning Orbital View')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            try:
                self.ax.grid(True, linestyle='--', alpha=0.3)
            except Exception:
                pass
            self.canvas.draw()
        except Exception:
            try:
                self.ax.clear()
                self.ax.text(0.5, 0.5, 'Visualizer error', ha='center')
                self.canvas.draw()
            except Exception:
                pass

# ============================================================================
# CHATBOT WITH SMILEY AVATAR
# ============================================================================

class ChatMessage:
    """Represents a chat message"""
    def __init__(self, text: str, sender: str = "ROCA", is_user: bool = False):
        self.text = text
        self.sender = sender
        self.is_user = is_user
        self.timestamp = datetime.now()
        self.id = hashlib.md5(f"{text}{sender}{self.timestamp}".encode()).hexdigest()[:8]
    
    def to_html(self) -> str:
        """Convert to HTML for display"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        sender_color = "#6cf" if not self.is_user else "#8f8"
        bg_color = "#2a2a3a" if not self.is_user else "#3a3a2a"
        
        return f"""
        <div style="margin: 10px 5px; padding: 8px; background: {bg_color}; border-radius: 10px;">
            <div style="color: #aaa; font-size: 9pt;">{time_str}</div>
            <div>
                <span style="color: {sender_color}; font-weight: bold;">{self.sender}:</span>
                <span style="color: #fff; margin-left: 5px;">{self.text}</span>
            </div>
        </div>
        """

class ChatBot:
    """Enhanced chatbot with contextual memory"""
    
    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.context_memory = []
        self.smiley_moods = ["happy", "thinking", "curious", "excited"]
        self.current_mood = "happy"
        
        # Initialize autonomous brain if available
        self.autonomous_brain = None
        if ai_fast_mod:
            try:
                self.autonomous_brain = ai_fast_mod.AutonomousBrain()
                print("🧠 Autonomous Brain wired to dialogue")
            except Exception as e:
                print(f"Failed to initialize autonomous brain: {e}")
        
        # Initialize AI coding assistant if available
        self.ai_coding_assistant = None
        if ai_coding_assistant_mod:
            try:
                self.ai_coding_assistant = ai_coding_assistant_mod.AICodingAssistant()
                print("💻 AI Coding Assistant wired to dialogue")
            except Exception as e:
                print(f"Failed to initialize AI coding assistant: {e}")
        
        # Initialize Core_Modules cognitive systems
        self.personality_system = None
        self.hierarchical_memory = None
        self.intrinsic_motivation = None
        self.knowledge_network = None
        self.causal_discovery = None
        self.meta_learner = None
        
        if PersonalitySystem:
            try:
                self.personality_system = PersonalitySystem()
                print("🎭 Personality System initialized")
            except Exception as e:
                print(f"Failed to initialize personality system: {e}")
        
        if HierarchicalTemporalMemory:
            try:
                self.hierarchical_memory = HierarchicalTemporalMemory(input_size=128, hidden_size=256)
                print("🧠 Hierarchical Memory initialized")
            except Exception as e:
                print(f"Failed to initialize hierarchical memory: {e}")
        
        if IntrinsicMotivationSystem:
            try:
                self.intrinsic_motivation = IntrinsicMotivationSystem(input_size=128, output_size=64)
                print("🎯 Intrinsic Motivation initialized")
            except Exception as e:
                print(f"Failed to initialize intrinsic motivation: {e}")
        
        if KnowledgeNetwork:
            try:
                self.knowledge_network = KnowledgeNetwork()
                print("🌐 Knowledge Network initialized")
            except Exception as e:
                print(f"Failed to initialize knowledge network: {e}")
        
        # Add welcome message
        self.add_message("Hello! I'm ROCA, your AI assistant with orbital visualization! 😊", "ROCA")
    
    def add_message(self, text: str, sender: str = "ROCA", is_user: bool = False):
        """Add a message to the chat"""
        message = ChatMessage(text, sender, is_user)
        self.messages.append(message)
        
        # Keep context memory
        if len(self.context_memory) > 10:
            self.context_memory.pop(0)
        self.context_memory.append(text[:100])
        
        # Update mood based on message
        self._update_mood(text)
        
        return message
    
    def _update_mood(self, text: str):
        """Update chatbot mood based on conversation"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["great", "awesome", "thanks", "thank", "good"]):
            self.current_mood = "happy"
        elif "?" in text:
            self.current_mood = "curious"
        elif any(word in text_lower for word in ["problem", "error", "issue", "help"]):
            self.current_mood = "thinking"
        elif any(word in text_lower for word in ["wow", "amazing", "cool"]):
            self.current_mood = "excited"
    
    def generate_response(self, user_input: str) -> str:
        """Generate chatbot response using advanced cognitive systems"""
        user_input_lower = user_input.lower()
        
        # Check if this is a coding/AI related query
        coding_keywords = ["code", "python", "programming", "function", "class", "algorithm", 
                          "neural", "network", "machine learning", "ai", "capsule", "roca",
                          "convert", "architecture", "generate", "write", "debug", "error"]
        
        is_coding_query = any(keyword in user_input_lower for keyword in coding_keywords)
        
        # Use AI coding assistant for coding queries
        if self.ai_coding_assistant and is_coding_query:
            try:
                response = self.ai_coding_assistant.chat(user_input)
                return response + " 💻"
            except Exception as e:
                print(f"AI coding assistant error: {e}")
                # Fall back to other methods
        
        # Use autonomous brain for general responses
        if self.autonomous_brain:
            # Create perception from user input
            perception = ai_fast_mod.Perception()
            perception.user_activity = "chatting"
            perception.user_emotional_cues = []
            
            # Analyze user input for emotional cues
            if any(word in user_input_lower for word in ["great", "awesome", "thanks", "good"]):
                perception.user_emotional_cues.append("positive")
            if "?" in user_input:
                perception.user_emotional_cues.append("curious")
            if any(word in user_input_lower for word in ["problem", "error", "issue", "help"]):
                perception.user_emotional_cues.append("frustrated")
            
            # Simulate capsule activity based on input
            if "capsule" in user_input_lower:
                perception.capsule_activity = {
                    'new_creations': 1,
                    'active_capsules': random.randint(1, 5),
                    'recent_activity': 'medium'
                }
            
            # Update brain perception
            self.autonomous_brain.short_term_memory.add_perception(perception)
            
            # Generate thought
            thought = self.autonomous_brain.think_cycle()
            
            if thought and thought.content:
                # Use thought content as response, but format it nicely
                response = thought.content
                # Add some personality based on thought type
                if thought.type == ai_fast_mod.ThoughtType.QUESTION:
                    response += " 🤔"
                elif thought.type == ai_fast_mod.ThoughtType.IDEA:
                    response += " 💡"
                elif thought.type == ai_fast_mod.ThoughtType.REFLECTION:
                    response += " 🧠"
                
                # Enhance with personality system if available
                if self.personality_system:
                    try:
                        personality_enhancement = self.personality_system.generate_personality_response(
                            response, self.current_mood, user_input
                        )
                        response = personality_enhancement
                    except Exception as e:
                        print(f"Personality system error: {e}")
                
                # Store in hierarchical memory
                if self.hierarchical_memory:
                    try:
                        # Convert text to simple embedding (placeholder)
                        input_embedding = np.random.randn(128)  # Placeholder embedding
                        self.hierarchical_memory.process_sequence([input_embedding])
                    except Exception as e:
                        print(f"Hierarchical memory error: {e}")
                
                # Update intrinsic motivation
                if self.intrinsic_motivation:
                    try:
                        # Calculate novelty and competence
                        novelty = self._calculate_novelty(user_input)
                        competence = self._calculate_competence(response)
                        self.intrinsic_motivation.update_motivation(novelty, competence)
                    except Exception as e:
                        print(f"Intrinsic motivation error: {e}")
                
                return response
        
        # Fallback to original responses if brain not available or no thought generated
        user_input_lower = user_input.lower()
        
        # Contextual responses
        if "hello" in user_input_lower or "hi " in user_input_lower:
            responses = [
                "Hello! How can I assist you today? 😊",
                "Hi there! Ready to explore some AI concepts? 🚀",
                "Greetings! What would you like to discuss? 💭"
            ]
            return random.choice(responses)
        
        elif "help" in user_input_lower:
            return "I can help you with: AI concepts, capsule networks, orbital visualization, and more! Try asking about specific topics or add capsules to the orbital view. 🎯"
        
        elif "capsule" in user_input_lower:
            return "Capsule networks use capsules (groups of neurons) to recognize patterns while preserving spatial relationships! Each capsule in the orbital view represents a concept. 🧠"
        
        elif "orbital" in user_input_lower or "visualization" in user_input_lower:
            return "The orbital visualization shows knowledge capsules orbiting a central nucleus! Size = certainty, distance = abstraction, color = concept type. It's powered by numpy for smooth animations! 🌌"
        
        elif "numpy" in user_input_lower:
            return "This app uses NumPy for efficient calculations! Capsule positions, physics, and animations are all optimized with vectorized operations. 🚀"
        
        elif "what can you do" in user_input_lower:
            return "I can: 1) Chat about AI topics 🤖, 2) Visualize concepts in 3D orbital view 🌠, 3) Learn from our conversation 📚, 4) Manage knowledge capsules 🧠, 5) Speak responses 🔊"
        
        elif "thank" in user_input_lower:
            return "You're welcome! Feel free to ask more questions or explore the orbital visualization. I'm here to help! 😊"
        
        elif "?" in user_input:
            topic = user_input.split('?')[0].strip()
            return f"That's an interesting question about '{topic}'! In capsule networks, this relates to hierarchical representation learning. Would you like me to elaborate? 🤔"
        
        # Default creative response
        topics = ["neural networks", "machine learning", "deep learning", "AI ethics", "reinforcement learning"]
        topic = random.choice(topics)
        return f"I understand you're mentioning '{user_input[:30]}...'. This relates to {topic}! In capsule networks, we handle concepts through recursive routing. Want to add this as a capsule? 🎯"
    
    def _calculate_novelty(self, user_input: str) -> float:
        """Calculate novelty of user input based on context memory"""
        if not self.context_memory:
            return 1.0
        
        # Simple novelty calculation based on word overlap
        input_words = set(user_input.lower().split())
        total_overlap = 0
        
        for context in self.context_memory[-5:]:  # Check last 5 messages
            context_words = set(context.lower().split())
            overlap = len(input_words.intersection(context_words))
            total_overlap += overlap / max(len(input_words), len(context_words))
        
        average_overlap = total_overlap / min(5, len(self.context_memory))
        novelty = 1.0 - average_overlap  # Higher overlap = lower novelty
        
        return max(0.0, min(1.0, novelty))
    
    def _calculate_competence(self, response: str) -> float:
        """Calculate competence based on response quality indicators"""
        # Simple heuristics for response quality
        score = 0.5  # Base competence
        
        # Length indicator (not too short, not too long)
        length = len(response)
        if 20 <= length <= 200:
            score += 0.2
        elif length < 10:
            score -= 0.2
        
        # Contains helpful indicators
        helpful_words = ["help", "explain", "understand", "learn", "create", "build"]
        if any(word in response.lower() for word in helpful_words):
            score += 0.1
        
        # Contains emojis (engagement indicator)
        if any(char in response for char in ["😊", "🤔", "💡", "🧠", "🎯"]):
            score += 0.1
        
        # Question in response (engagement)
        if "?" in response:
            score += 0.1
        
        return max(0.0, min(1.0, score))

# ============================================================================
# SMILEY FACE AVATAR WIDGET
# ============================================================================

class SmileyAvatar(QWidget):
    """Interactive smiley face avatar that responds to chatbot mood"""
    
    mood_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mood = "happy"
        self.eye_blink = False
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink)
        self.blink_timer.start(3000)
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(50)
        
        # Set fixed size
        self.setFixedSize(120, 120)
        
        # Colors for different moods
        self.mood_colors = {
            "happy": QColor(255, 220, 100),
            "thinking": QColor(150, 200, 255),
            "curious": QColor(200, 150, 255),
            "excited": QColor(255, 150, 150)
        }
    
    def set_mood(self, mood: str):
        """Set the avatar mood"""
        if mood in self.mood_colors:
            self.mood = mood
            self.mood_changed.emit(mood)
            self.update()
    
    def toggle_blink(self):
        """Toggle eye blink animation"""
        self.eye_blink = not self.eye_blink
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get mood color
        base_color = self.mood_colors.get(self.mood, QColor(255, 220, 100))
        
        # Draw background gradient
        gradient = QRadialGradient(60, 60, 60)
        gradient.setColorAt(0, base_color.lighter(150))
        gradient.setColorAt(1, base_color.darker(120))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(base_color.darker(150), 2))
        painter.drawEllipse(10, 10, 100, 100)
        
        # Draw eyes
        eye_color = QColor(50, 50, 50)
        painter.setBrush(QBrush(eye_color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Left eye (blink animation)
        if not self.eye_blink:
            painter.drawEllipse(35, 40, 15, 25)
        else:
            painter.drawRect(35, 50, 15, 3)
        
        # Right eye
        painter.drawEllipse(70, 40, 15, 25)
        
        # Draw mouth based on mood
        self._draw_mouth(painter, base_color)
        
        # Draw highlights
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawEllipse(20, 20, 80, 80)
        
        # Draw mood indicator
        mood_text = self.mood.upper()
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 115, 120, 20, Qt.AlignmentFlag.AlignCenter, mood_text)
    
    def _draw_mouth(self, painter: QPainter, base_color: QColor):
        """Draw mouth based on current mood"""
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.setPen(QPen(base_color.darker(200), 2))
        
        if self.mood == "happy":
            # Smiling mouth
            painter.drawArc(35, 55, 50, 30, 0, 180 * 16)
        elif self.mood == "thinking":
            # Straight mouth
            painter.drawLine(40, 70, 80, 70)
        elif self.mood == "curious":
            # Small "o" mouth
            painter.drawEllipse(50, 65, 20, 15)
        elif self.mood == "excited":
            # Wide smile
            painter.drawArc(30, 50, 60, 40, 0, 180 * 16)

# ============================================================================
# NUMPY ENHANCED ORBITAL WIDGET
# ============================================================================

class NumpyOrbitalWidget(QWidget):
    """Orbital visualization optimized with numpy"""
    
    capsule_selected = pyqtSignal(object)
    capsule_hovered = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.capsules: List[NumpyCapsule] = []
        self.selected_capsule = None
        self.hovered_capsule = None
        
        # Visualization parameters
        self.zoom_level = np.float32(1.0)
        self.rotation_angle = np.float32(0.0)
        self.rotation_speed = np.float32(0.5)
        self.auto_rotate = True
        self.show_info = True
        
        # Physics parameters
        self.physics_enabled = True
        self.attraction_strength = np.float32(0.1)
        self.repulsion_strength = np.float32(0.2)
        self.damping = np.float32(0.95)
        
        # Colors
        self.bg_color = QColor(20, 20, 30)
        self.panel_color = QColor(30, 30, 40, 200)
        self.highlight_color = QColor(100, 200, 255)
        
        # Starfield data
        self.stars = self._generate_starfield(500)
        
        # Performance optimization
        self.last_update_time = time.time()
        self.update_interval = 1/60.0
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_physics)
        self.timer.start(int(1000/60))  # 60 FPS
        
        # Initialize with sample capsules
        self._populate_initial_capsules()
        
        # Set minimum size
        self.setMinimumSize(400, 400)

        # Numba/parallel options
        self.use_numba = NUMBA_AVAILABLE
        self.use_parallel = True

        # Barnes-Hut and numba thresholds (tunable)
        self.bh_threshold = 100
        self.numba_pairwise_limit = 2000
        self.bh_theta = 0.5

        # Reusable numeric buffers to reduce per-frame allocations
        self._buf_capacity = 0
        self._buf_positions = None
        self._buf_velocities = None
        self._buf_masses = None
        self._buf_accelerations = None

    def _ensure_numeric_buffers(self, n):
        """Ensure internal buffers have capacity for n particles."""
        if n <= self._buf_capacity and self._buf_positions is not None:
            return

        new_cap = max(n, 16)
        if self._buf_capacity > 0:
            new_cap = max(new_cap, self._buf_capacity * 2)

        self._buf_positions = np.zeros((new_cap, 2), dtype=np.float64)
        self._buf_velocities = np.zeros((new_cap, 2), dtype=np.float64)
        self._buf_masses = np.zeros((new_cap,), dtype=np.float64)
        self._buf_accelerations = np.zeros((new_cap, 2), dtype=np.float64)
        self._buf_capacity = new_cap
        # prediction overlay helpers (methods defined later may not be bound in some edits)
    
    def show_predictions(self, positions: np.ndarray, ttl_ms: int = 5000):
        """Show predicted positions overlay for `ttl_ms` milliseconds."""
        try:
            if positions is None:
                self.predicted_positions = None
                return
            self.predicted_positions = np.array(positions, dtype=np.float64)
            try:
                self._pred_clear_timer.stop()
            except Exception:
                pass
            try:
                self._pred_clear_timer.start(int(ttl_ms))
            except Exception:
                pass
            self.update()
        except Exception:
            pass

    def clear_predictions(self):
        try:
            self.predicted_positions = None
            try:
                self._pred_clear_timer.stop()
            except Exception:
                pass
            self.update()
        except Exception:
            pass
        # attach genome sequencer - prefer threadripper_core implementation
        try:
            try:
                from threadripper_core.CapsuleGenomeSequencer import CapsuleGenomeSequencer as TRCapsuleGenomeSequencer
                self.genome_sequencer = TRCapsuleGenomeSequencer()
            except Exception:
                # fall back to local definition if present
                self.genome_sequencer = CapsuleGenomeSequencer()
        except Exception:
            self.genome_sequencer = None

        # Capsule store (persistence) - prefer NUMA-optimized / evolutionary store
        try:
            try:
                from threadripper_core.NUMA_Aware_Capsule_Distribution import NUMAOptimizedCapsuleStore
            except Exception:
                NUMAOptimizedCapsuleStore = None

            try:
                # prefer the EvolutionaryCapsuleStore if present
                from SELF_EXPANSION_MECHANISMS import EvolutionaryCapsuleStore
            except Exception:
                EvolutionaryCapsuleStore = None

            # Only create a default store if one hasn't been provided by the
            # surrounding application (e.g., NumpyROCAWindow may inject one).
            if not hasattr(self, 'store') or getattr(self, 'store', None) is None:
                threadripper_mode = os.environ.get('THREADRIPPER_MODE', '0').lower() in ('1', 'true', 'yes')
                created = False

                # prefer NUMA-optimized store when available and appropriate
                if NUMAOptimizedCapsuleStore is not None:
                    try:
                        import numa as _numa_check
                        numa_nodes = getattr(_numa_check, 'get_max_node', lambda: 0)()
                        multicore_numa = (numa_nodes + 1) > 1
                    except Exception:
                        multicore_numa = False

                    if multicore_numa or threadripper_mode:
                        try:
                            self.store = NUMAOptimizedCapsuleStore()
                            created = True
                        except Exception:
                            created = False

                if not created:
                    if EvolutionaryCapsuleStore is not None:
                        try:
                            self.store = EvolutionaryCapsuleStore()
                        except Exception:
                            try:
                                from core.capsule_store import CapsuleStore
                                self.store = CapsuleStore()
                            except Exception:
                                from threadripper_core.NUMA_Aware_Capsule_Distribution import _InMemoryStore
                                self.store = _InMemoryStore()
                    else:
                        try:
                            from core.capsule_store import CapsuleStore
                            self.store = CapsuleStore()
                        except Exception:
                            from threadripper_core.NUMA_Aware_Capsule_Distribution import _InMemoryStore
                            self.store = _InMemoryStore()

            # attempt to load default store if present
            default_path = os.path.join(os.getcwd(), 'capsules_store.json')
            if os.path.exists(default_path) and hasattr(self.store, 'load'):
                try:
                    self.store.load(default_path)
                    # populate widget from store
                    for cap in self.store.list_all():
                        try:
                            self.add_capsule_object(cap)
                        except Exception:
                            continue
                except Exception:
                    pass

            # attempt to load universal capsules if supported
            default_univ = os.path.join(os.getcwd(), 'universal_capsules.json')
            if os.path.exists(default_univ) and hasattr(self.store, 'load_universal'):
                try:
                    self.store.load_universal(default_univ)
                except Exception:
                    pass
        except Exception:
            self.store = None
        # predicted positions overlay
        self.predicted_positions: Optional[np.ndarray] = None
        self._pred_clear_timer = QTimer()
        self._pred_clear_timer.setSingleShot(True)
        self._pred_clear_timer.timeout.connect(self.clear_predictions)
    
    def _generate_starfield(self, num_stars: int) -> np.ndarray:
        """Generate starfield positions and brightness"""
        stars = np.zeros((num_stars, 3), dtype=np.float32)
        stars[:, :2] = np.random.rand(num_stars, 2)  # x, y positions (0-1)
        stars[:, 2] = np.random.rand(num_stars) * 0.8 + 0.2  # brightness
        return stars
    
    def _populate_initial_capsules(self):
        """Create initial capsules with numpy operations"""
        concepts = np.array([
            "Neural Networks", "Capsule Theory", "Attention Mechanism",
            "Recursive Routing", "Memory Networks", "Transfer Learning",
            "Reinforcement Learning", "Generative Models", "Symbolic AI",
            "Backpropagation", "Convolution", "Embeddings"
        ])
        
        kinds = np.array(["theory", "method", "hypothesis", "concept", "data"])
        
        for i, concept in enumerate(concepts):
            kind = np.random.choice(kinds)
            certainty = np.random.uniform(0.4, 1.0)
            
            capsule = NumpyCapsule(
                content=concept,
                kind=kind,
                certainty=np.float32(certainty),
                orbit_radius=np.float32(np.random.uniform(0.8, 2.5)),
                angle=np.float32(np.random.uniform(0, 2 * np.pi))
            )
            self.capsules.append(capsule)
    
    def add_capsule(self, content: str, kind: str = "concept", certainty: float = 0.7) -> NumpyCapsule:
        """Add a new capsule using numpy randomization"""
        capsule = NumpyCapsule(
            content=content,
            kind=kind,
            certainty=np.float32(certainty),
            orbit_radius=np.float32(np.random.uniform(1.0, 2.5)),
            angle=np.float32(np.random.uniform(0, 2 * np.pi))
        )
        self.capsules.append(capsule)
        try:
            if self.store is not None:
                self.store.add(capsule)
        except Exception:
            pass
        return capsule
    
    def add_capsule_object(self, capsule: NumpyCapsule):
        """Add an existing capsule object to the orbital display"""
        self.capsules.append(capsule)
        try:
            if self.store is not None:
                self.store.add(capsule)
        except Exception:
            pass

    def save_store(self, path: Optional[str] = None):
        """Save the capsule store to disk. Defaults to `capsules_store.json` in CWD."""
        if self.store is None:
            return
        if path is None:
            path = os.path.join(os.getcwd(), 'capsules_store.json')
        try:
            self.store.save(path)
        except Exception:
            pass

    def load_store(self, path: Optional[str] = None):
        """Load capsules from disk and populate the widget (clears current)."""
        if self.store is None:
            return
        if path is None:
            path = os.path.join(os.getcwd(), 'capsules_store.json')
        try:
            self.store.load(path)
            # replace current capsules with store contents
            self.capsules = []
            for cap in self.store.list_all():
                self.capsules.append(cap)
        except Exception:
            pass
    
    def update_physics(self):
        """Update capsule physics using numpy vectorized operations"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        dt = np.float32(current_time - self.last_update_time)
        self.last_update_time = current_time
        
        if self.auto_rotate:
            self.rotation_angle += self.rotation_speed * dt * 0.5
        
        # Update capsule angles
        if self.capsules:
            # Vectorized angle update
            angles = np.array([c.angle for c in self.capsules], dtype=np.float32)
            certainties = np.array([c.certainty for c in self.capsules], dtype=np.float32)
            angles += 0.002 * (0.5 + certainties)
            angles %= (2 * np.pi)
            
            # Apply updated angles
            for i, capsule in enumerate(self.capsules):
                capsule.angle = angles[i]
        
        # Apply physics if enabled
        if self.physics_enabled and len(self.capsules) > 1:
            self._apply_capsule_physics(dt)
        
        self.update()
    
    def _apply_capsule_physics(self, dt: np.float32):
        """Apply attraction/repulsion physics between capsules"""
        n = len(self.capsules)
        if n < 2:
            return
        # Use Barnes-Hut quadtree for large N to approximate forces
        # Threshold for switching to Barnes-Hut
        BH_THRESHOLD = 100

        # reuse internal buffers to avoid allocations
        self._ensure_numeric_buffers(n)
        positions = self._buf_positions
        velocities = self._buf_velocities
        masses = self._buf_masses
        accelerations = self._buf_accelerations

        # fill buffers
        for i in range(n):
            p = self.capsules[i].position
            v = self.capsules[i].velocity
            positions[i, 0] = float(p[0])
            positions[i, 1] = float(p[1])
            velocities[i, 0] = float(v[0])
            velocities[i, 1] = float(v[1])
            masses[i] = max(1.0, float(self.capsules[i].certainty * 10.0))

        # zero only the active slice of accelerations
        accelerations[:n, :] = 0.0

        # Option: use numba-accelerated pairwise computation for moderate N
        NUMBA_PAIRWISE_LIMIT = getattr(self, 'numba_pairwise_limit', 2000)
        if NUMBA_AVAILABLE and self.use_numba and n <= NUMBA_PAIRWISE_LIMIT:
            # Use numba pairwise acceleration
            G = float(self.attraction_strength * 100.0)
            softening = 1.0
            try:
                acc = _numba_pairwise_accelerations(positions, masses, G, softening)
                accelerations = acc
            except Exception:
                # Fall back to Barnes-Hut if numba call fails
                pass

        BH_THRESHOLD = getattr(self, 'bh_threshold', 100)
        if accelerations.sum() == 0 and n >= BH_THRESHOLD:
            # Build quadtree (prefer Numba native builder to avoid Python object overhead)
            theta = getattr(self, 'bh_theta', 0.5)
            G = float(self.attraction_strength * 100.0)
            softening = 1.0

            if NUMBA_AVAILABLE and self.use_numba:
                try:
                    (node_cx, node_cy, node_h, node_mass, node_comx, node_comy,
                     node_pidx, child0, child1, child2, child3) = _numba_build_bh_tree(positions, masses)

                    acc = _numba_bh_forces(node_cx, node_cy, node_h, node_mass, node_comx, node_comy,
                                           node_pidx, child0, child1, child2, child3,
                                           positions, masses, float(theta), float(G), float(softening))
                    accelerations = acc
                except Exception:
                    # Fallback to Python quadtree construction if numba builder fails
                    minx, miny = positions.min(axis=0)
                    maxx, maxy = positions.max(axis=0)
                    cx = 0.5 * (minx + maxx)
                    cy = 0.5 * (miny + maxy)
                    size = max(maxx - minx, maxy - miny) * 1.1 + 1e-3

                    root = _QuadNode(cx, cy, size / 2.0)
                    for i in range(n):
                        x, y = float(positions[i,0]), float(positions[i,1])
                        root.insert(i, x, y, float(masses[i]))

                    for i in range(n):
                        x, y = float(positions[i,0]), float(positions[i,1])
                        fx, fy = root.compute_force_on(i, x, y, theta, G, softening)
                        accelerations[i,0] = fx / masses[i]
                        accelerations[i,1] = fy / masses[i]
            else:
                # Python quadtree build & traversal
                minx, miny = positions.min(axis=0)
                maxx, maxy = positions.max(axis=0)
                cx = 0.5 * (minx + maxx)
                cy = 0.5 * (miny + maxy)
                size = max(maxx - minx, maxy - miny) * 1.1 + 1e-3

                root = _QuadNode(cx, cy, size / 2.0)
                for i in range(n):
                    x, y = float(positions[i,0]), float(positions[i,1])
                    root.insert(i, x, y, float(masses[i]))

                for i in range(n):
                    x, y = float(positions[i,0]), float(positions[i,1])
                    fx, fy = root.compute_force_on(i, x, y, theta, G, softening)
                    accelerations[i,0] = fx / masses[i]
                    accelerations[i,1] = fy / masses[i]
        else:
            # Direct O(n^2) computation for small N
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[j] - positions[i]
                    dist_sq = np.dot(dx, dx) + 1e-6
                    dist = np.sqrt(dist_sq)
                    dir_vec = dx / dist
                    # Attractive/repulsive balance
                    if dist > 100.0:
                        force = float(self.attraction_strength) / dist_sq
                    else:
                        force = -float(self.repulsion_strength) / dist_sq
                    fvec = dir_vec * force
                    accelerations[i] += fvec / masses[i]
                    accelerations[j] -= fvec / masses[j]

        # Integrate velocities and positions (semi-implicit Euler)
        velocities += accelerations * float(dt)
        velocities *= float(self.damping)
        positions += velocities * float(dt)

        # Write back to capsules (use buffers)
        for i in range(n):
            capsule = self.capsules[i]
            # set velocity and position from buffers
            capsule.velocity = np.array(velocities[i], dtype=np.float32)
            capsule.position = np.array(positions[i], dtype=np.float32)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.bg_color)
        
        w = self.width()
        h = self.height()
        center = np.array([w / 2, h / 2], dtype=np.float32)
        
        # Draw starfield
        self._draw_starfield(painter, w, h)
        
        # Draw orbit paths
        if self.show_info and self.capsules:
            self._draw_orbit_paths(painter, center)
        
        # Update capsule positions
        for capsule in self.capsules:
            capsule.update_position(center, self.zoom_level, self.rotation_angle)
        
        # Draw capsules
        for capsule in self.capsules:
            self._draw_capsule(painter, capsule)

        # Draw predicted positions overlay (hollow markers)
        if getattr(self, 'predicted_positions', None) is not None:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2, Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for p in self.predicted_positions:
                try:
                    x = float(p[0])
                    y = float(p[1])
                    painter.drawEllipse(int(x - 6), int(y - 6), 12, 12)
                except Exception:
                    continue
        
        # Draw central nucleus
        self._draw_central_nucleus(painter, center[0], center[1])
        
        # Draw info panel
        if self.show_info:
            self._draw_info_panel(painter)
    
    def _draw_starfield(self, painter: QPainter, w: int, h: int):
        """Draw animated starfield"""
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Animate stars with subtle movement
        time_factor = np.sin(time.time() * 0.5) * 0.5 + 0.5
        
        for star in self.stars:
            # Calculate position with parallax effect
            x = star[0] * w + np.sin(time.time() * 0.3 + star[0]) * 10
            y = star[1] * h + np.cos(time.time() * 0.3 + star[1]) * 10
            
            # Calculate brightness with twinkling
            brightness = int(star[2] * 255 * (0.8 + 0.2 * np.sin(time.time() * 2 + star[0] * 10)))
            
            color = QColor(brightness, brightness, brightness, int(brightness * 0.8))
            painter.setBrush(QBrush(color))
            
            size = 1 + star[2] * 2
            painter.drawEllipse(int(x), int(y), int(size), int(size))
    
    def _draw_orbit_paths(self, painter: QPainter, center: np.ndarray):
        """Draw orbit paths for capsules"""
        painter.setPen(QPen(QColor(60, 60, 80, 100), 1))
        painter.setOpacity(0.3)
        
        # Get unique orbit radii
        radii = np.array([c.orbit_radius for c in self.capsules], dtype=np.float32)
        unique_radii = np.unique(np.round(radii * 10) / 10)
        
        for radius in unique_radii:
            orbit_radius = radius * 50 * self.zoom_level
            painter.drawEllipse(
                int(center[0] - orbit_radius),
                int(center[1] - orbit_radius),
                int(orbit_radius * 2),
                int(orbit_radius * 2)
            )
        
        painter.setOpacity(1.0)
    
    def _draw_capsule(self, painter: QPainter, capsule: NumpyCapsule):
        """Draw a single capsule"""
        x, y = capsule.position
        color = QColor(
            int(capsule.color[0] * 255),
            int(capsule.color[1] * 255),
            int(capsule.color[2] * 255)
        )
        size = int(capsule.size)
        
        # Draw glow effect
        for i in range(3, 0, -1):
            glow_color = QColor(color)
            glow_color.setAlpha(int(40 / (i + 1)))
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.PenStyle.NoPen)
            glow_radius = size + i * 3
            painter.drawEllipse(int(x - glow_radius/2), int(y - glow_radius/2), glow_radius, glow_radius)
        
        # Draw capsule body with gradient
        gradient = QRadialGradient(x, y, size/2)
        gradient.setColorAt(0, color.lighter(150))
        gradient.setColorAt(1, color.darker(120))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawEllipse(int(x - size/2), int(y - size/2), size, size)
        
        # Draw label if zoomed in or selected/hovered
        if self.zoom_level > 0.8 or capsule in [self.selected_capsule, self.hovered_capsule]:
            painter.setPen(QPen(QColor(255, 255, 255)))
            font = painter.font()
            font.setPointSize(max(8, int(10 * self.zoom_level)))
            painter.setFont(font)
            label = capsule.character
            text_rect = painter.fontMetrics().boundingRect(label)
            painter.drawText(int(x - text_rect.width()/2), int(y + size/2 + 15), label)
        
        # Highlight selected capsule
        if capsule == self.selected_capsule:
            painter.setPen(QPen(self.highlight_color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(x - size/2 - 5), int(y - size/2 - 5), size + 10, size + 10)
        
        # Highlight hovered capsule
        elif capsule == self.hovered_capsule:
            painter.setPen(QPen(QColor(200, 200, 150, 150), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(x - size/2 - 3), int(y - size/2 - 3), size + 6, size + 6)
    
    def _draw_central_nucleus(self, painter: QPainter, cx: float, cy: float):
        """Draw the central identity nucleus"""
        # Draw glow
        for i in range(30, 0, -5):
            glow_color = QColor(255, 200, 0)
            glow_color.setAlpha(int(100 * (30 - i) / 30))
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(cx - i), int(cy - i), i * 2, i * 2)
        
        # Draw nucleus with gradient
        gradient = QRadialGradient(cx, cy, 15)
        gradient.setColorAt(0, QColor(255, 240, 150))
        gradient.setColorAt(1, QColor(255, 200, 50))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 220, 100), 2))
        painter.drawEllipse(int(cx - 15), int(cy - 15), 30, 30)
        
        # Draw core
        painter.setBrush(QBrush(QColor(255, 255, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
        
        # Draw label
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(int(cx + 20), int(cy - 5), "ROCA")
        painter.drawText(int(cx + 20), int(cy + 10), "Core")
    
    def _draw_info_panel(self, painter: QPainter):
        """Draw info panel with statistics"""
        panel_w = 120
        panel_h = 90
        
        # Background
        painter.fillRect(10, 10, panel_w, panel_h, self.panel_color)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawRect(10, 10, panel_w, panel_h)
        
        # Text
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(15, 25, "ORBITAL VIEW")
        
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        
        y = 40
        lines = [
            f"Capsules: {len(self.capsules)}",
            f"Zoom: {self.zoom_level:.1f}x",
            f"Physics: {'ON' if self.physics_enabled else 'OFF'}",
            f"FPS: {int(1/self.update_interval)}"
        ]
        
        for line in lines:
            painter.drawText(15, y, line)
            y += 12


# Numba-accelerated pairwise force computation (falls back to pure Python if numba unavailable)
@njit(parallel=True)
def _numba_pairwise_accelerations(positions, masses, G, softening):
    n = positions.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in prange(n):
        ax = 0.0
        ay = 0.0
        xi = positions[i, 0]
        yi = positions[i, 1]
        mi = masses[i]
        for j in range(n):
            if i == j:
                continue
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dist_sq = dx * dx + dy * dy + 1e-6
            invr = 1.0 / np.sqrt(dist_sq)
            invr3 = invr / dist_sq
            f = G * masses[j] * invr3
            ax += f * dx
            ay += f * dy
        acc[i, 0] = ax / mi
        acc[i, 1] = ay / mi
    return acc

def _py_pairwise_accelerations(positions, masses, G, softening):
    # Numpy fallback (vectorized-ish)
    n = positions.shape[0]
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        dx = positions[:, 0] - positions[i, 0]
        dy = positions[:, 1] - positions[i, 1]
        dist_sq = dx * dx + dy * dy + 1e-6
        mask = np.arange(n) != i
        invr = 1.0 / np.sqrt(dist_sq[mask])
        invr3 = invr / dist_sq[mask]
        f = G * masses[mask] * invr3
        acc_vec = np.vstack((dx[mask] * f, dy[mask] * f)).T
        s = acc_vec.sum(axis=0)
        acc[i] = s / masses[i]
    return acc


def _serialize_quadtree_for_numba(root):
    """Serialize the _QuadNode tree into numpy arrays for a numba traversal.

    Returns arrays: node_cx, node_cy, node_h, node_mass, node_comx, node_comy,
    node_pidx, child0, child1, child2, child3
    """
    nodes = []
    mapping = {}
    # BFS collect nodes
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node in mapping:
            continue
        mapping[node] = len(nodes)
        nodes.append(node)
        for ch in node.children:
            if ch is not None:
                queue.append(ch)

    n_nodes = len(nodes)
    node_cx = np.zeros(n_nodes, dtype=np.float64)
    node_cy = np.zeros(n_nodes, dtype=np.float64)
    node_h = np.zeros(n_nodes, dtype=np.float64)
    node_mass = np.zeros(n_nodes, dtype=np.float64)
    node_comx = np.zeros(n_nodes, dtype=np.float64)
    node_comy = np.zeros(n_nodes, dtype=np.float64)
    node_pidx = np.full(n_nodes, -1, dtype=np.int64)
    child0 = np.full(n_nodes, -1, dtype=np.int64)
    child1 = np.full(n_nodes, -1, dtype=np.int64)
    child2 = np.full(n_nodes, -1, dtype=np.int64)
    child3 = np.full(n_nodes, -1, dtype=np.int64)

    for i, node in enumerate(nodes):
        node_cx[i] = float(node.cx)
        node_cy[i] = float(node.cy)
        node_h[i] = float(node.h)
        node_mass[i] = float(node.mass)
        node_comx[i] = float(node.com_x)
        node_comy[i] = float(node.com_y)
        if node.particle is not None:
            node_pidx[i] = int(node.particle[0])
        for k, ch in enumerate(node.children):
            if ch is not None:
                child_idx = mapping.get(ch, -1)
                if k == 0:
                    child0[i] = child_idx
                elif k == 1:
                    child1[i] = child_idx
                elif k == 2:
                    child2[i] = child_idx
                elif k == 3:
                    child3[i] = child_idx

    return (node_cx, node_cy, node_h, node_mass, node_comx, node_comy,
            node_pidx, child0, child1, child2, child3)


if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _numba_bh_forces(node_cx, node_cy, node_h, node_mass, node_comx, node_comy,
                         node_pidx, child0, child1, child2, child3,
                         positions, masses, theta, G, softening):
        n = positions.shape[0]
        acc = np.zeros((n, 2), dtype=np.float64)

        root_idx = 0
        # maximum stack depth (safe upper bound)
        STACK_MAX = 1024

        for i in prange(n):
            x = positions[i, 0]
            y = positions[i, 1]
            fx = 0.0
            fy = 0.0

            stack = np.empty(STACK_MAX, dtype=np.int64)
            sp = 0
            stack[sp] = root_idx
            sp += 1

            while sp > 0:
                sp -= 1
                node = stack[sp]
                if node_mass[node] == 0.0:
                    continue

                # If this node is a single particle identical to i and has no children, skip
                pidx = node_pidx[node]
                if pidx == i and child0[node] == -1 and child1[node] == -1 and child2[node] == -1 and child3[node] == -1:
                    continue

                dx = node_comx[node] - x
                dy = node_comy[node] - y
                dist_sq = dx * dx + dy * dy + 1e-12
                dist = np.sqrt(dist_sq)
                s = node_h[node] * 2.0

                # Accept node as a single body or descend
                if s / dist < theta or (child0[node] == -1 and child1[node] == -1 and child2[node] == -1 and child3[node] == -1):
                    invr = 1.0 / np.sqrt(dist_sq)
                    invr3 = invr / dist_sq
                    f = G * node_mass[node] * invr3
                    fx += f * dx
                    fy += f * dy
                else:
                    c0 = child0[node]
                    c1 = child1[node]
                    c2 = child2[node]
                    c3 = child3[node]
                    if c0 != -1:
                        stack[sp] = c0
                        sp += 1
                    if c1 != -1:
                        stack[sp] = c1
                        sp += 1
                    if c2 != -1:
                        stack[sp] = c2
                        sp += 1
                    if c3 != -1:
                        stack[sp] = c3
                        sp += 1

            acc[i, 0] = fx / masses[i]
            acc[i, 1] = fy / masses[i]

        return acc

else:
    # numba not available - provide a simple alias to raise or fallback
    def _numba_bh_forces(*args, **kwargs):
        raise RuntimeError('Numba not available')


if NUMBA_AVAILABLE:
    @njit
    def _numba_build_bh_tree(positions, masses):
        # positions: (n,2), masses: (n,)
        n = positions.shape[0]
        max_nodes = max(16, 4 * n)

        node_cx = np.zeros(max_nodes, dtype=np.float64)
        node_cy = np.zeros(max_nodes, dtype=np.float64)
        node_h = np.zeros(max_nodes, dtype=np.float64)
        node_mass = np.zeros(max_nodes, dtype=np.float64)
        node_comx = np.zeros(max_nodes, dtype=np.float64)
        node_comy = np.zeros(max_nodes, dtype=np.float64)
        node_pidx = np.full(max_nodes, -1, dtype=np.int64)
        child0 = np.full(max_nodes, -1, dtype=np.int64)
        child1 = np.full(max_nodes, -1, dtype=np.int64)
        child2 = np.full(max_nodes, -1, dtype=np.int64)
        child3 = np.full(max_nodes, -1, dtype=np.int64)

        # particle index buffer
        pbuf = np.empty(n, dtype=np.int64)
        for i in range(n):
            pbuf[i] = i

        # compute global bounds
        minx = positions[0,0]
        maxx = positions[0,0]
        miny = positions[0,1]
        maxy = positions[0,1]
        for i in range(1, n):
            px = positions[i,0]
            py = positions[i,1]
            if px < minx:
                minx = px
            if px > maxx:
                maxx = px
            if py < miny:
                miny = py
            if py > maxy:
                maxy = py

        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        size = max(maxx - minx, maxy - miny) * 1.1 + 1e-6

        # node segment arrays
        seg_start = np.full(max_nodes, -1, dtype=np.int64)
        seg_end = np.full(max_nodes, -1, dtype=np.int64)

        # init root
        node_cx[0] = cx
        node_cy[0] = cy
        node_h[0] = size / 2.0
        seg_start[0] = 0
        seg_end[0] = n
        nodes_used = 1
        proc = 0

        temp_idx = np.empty(n, dtype=np.int64)
        labels = np.empty(n, dtype=np.int64)

        while proc < nodes_used:
            node = proc
            proc += 1
            s = seg_start[node]
            e = seg_end[node]
            if s >= e:
                continue
            count = e - s

            if count == 1:
                pid = pbuf[s]
                node_pidx[node] = pid
                # set mass and center
                m = masses[pid]
                node_mass[node] = m
                node_comx[node] = positions[pid,0]
                node_comy[node] = positions[pid,1]
                continue

            # compute center for this node and assign h
            minx = positions[pbuf[s],0]
            maxx = positions[pbuf[s],0]
            miny = positions[pbuf[s],1]
            maxy = positions[pbuf[s],1]
            for k in range(s+1, e):
                pid = pbuf[k]
                px = positions[pid,0]
                py = positions[pid,1]
                if px < minx:
                    minx = px
                if px > maxx:
                    maxx = px
                if py < miny:
                    miny = py
                if py > maxy:
                    maxy = py

            ncx = 0.5 * (minx + maxx)
            ncy = 0.5 * (miny + maxy)
            nh = max(maxx - minx, maxy - miny) * 0.5 + 1e-12
            node_cx[node] = ncx
            node_cy[node] = ncy
            node_h[node] = nh

            # compute mass/com for node
            total_m = 0.0
            comx = 0.0
            comy = 0.0
            for k in range(s, e):
                pid = pbuf[k]
                m = masses[pid]
                total_m += m
                comx += positions[pid,0] * m
                comy += positions[pid,1] * m
            if total_m > 0.0:
                node_mass[node] = total_m
                node_comx[node] = comx / total_m
                node_comy[node] = comy / total_m

            # partition particles into 4 quadrants relative to ncx,ncy
            tpos = 0
            for k in range(s, e):
                pid = pbuf[k]
                px = positions[pid,0]
                py = positions[pid,1]
                q = 0
                if px >= ncx:
                    if py < ncy:
                        q = 1
                    else:
                        q = 3
                else:
                    if py < ncy:
                        q = 0
                    else:
                        q = 2
                temp_idx[tpos] = pid
                labels[tpos] = q
                tpos += 1

            # count per quadrant and copy back
            counts0 = 0
            counts1 = 0
            counts2 = 0
            counts3 = 0
            for k in range(tpos):
                q = labels[k]
                if q == 0:
                    counts0 += 1
                elif q == 1:
                    counts1 += 1
                elif q == 2:
                    counts2 += 1
                else:
                    counts3 += 1

            idx = s
            # write q0
            for k in range(tpos):
                if labels[k] == 0:
                    pbuf[idx] = temp_idx[k]
                    idx += 1
            p_q0_start = s
            p_q0_end = s + counts0

            # q1
            for k in range(tpos):
                if labels[k] == 1:
                    pbuf[idx] = temp_idx[k]
                    idx += 1
            p_q1_start = p_q0_end
            p_q1_end = p_q1_start + counts1

            # q2
            for k in range(tpos):
                if labels[k] == 2:
                    pbuf[idx] = temp_idx[k]
                    idx += 1
            p_q2_start = p_q1_end
            p_q2_end = p_q2_start + counts2

            # q3
            for k in range(tpos):
                if labels[k] == 3:
                    pbuf[idx] = temp_idx[k]
                    idx += 1
            p_q3_start = p_q2_end
            p_q3_end = p_q3_start + counts3

            # create children for non-empty quadrants
            if counts0 > 0:
                child0[node] = nodes_used
                seg_start[nodes_used] = p_q0_start
                seg_end[nodes_used] = p_q0_end
                nodes_used += 1
            if counts1 > 0:
                child1[node] = nodes_used
                seg_start[nodes_used] = p_q1_start
                seg_end[nodes_used] = p_q1_end
                nodes_used += 1
            if counts2 > 0:
                child2[node] = nodes_used
                seg_start[nodes_used] = p_q2_start
                seg_end[nodes_used] = p_q2_end
                nodes_used += 1
            if counts3 > 0:
                child3[node] = nodes_used
                seg_start[nodes_used] = p_q3_start
                seg_end[nodes_used] = p_q3_end
                nodes_used += 1

        # trim arrays to nodes_used
        node_cx = node_cx[:nodes_used].copy()
        node_cy = node_cy[:nodes_used].copy()
        node_h = node_h[:nodes_used].copy()
        node_mass = node_mass[:nodes_used].copy()
        node_comx = node_comx[:nodes_used].copy()
        node_comy = node_comy[:nodes_used].copy()
        node_pidx = node_pidx[:nodes_used].copy()
        child0 = child0[:nodes_used].copy()
        child1 = child1[:nodes_used].copy()
        child2 = child2[:nodes_used].copy()
        child3 = child3[:nodes_used].copy()

        return (node_cx, node_cy, node_h, node_mass, node_comx, node_comy,
                node_pidx, child0, child1, child2, child3)

else:
    def _numba_build_bh_tree(*args, **kwargs):
        raise RuntimeError('Numba not available')
    
    def mousePressEvent(self, event):
        """Handle mouse clicks for capsule selection"""
        if event.button() == Qt.MouseButton.LeftButton:
            mx = event.position().x()
            my = event.position().y()
            
            # Find clicked capsule (using numpy for distance calculation)
            click_pos = np.array([mx, my], dtype=np.float32)
            
            for capsule in self.capsules:
                dist = np.linalg.norm(click_pos - capsule.position)
                if dist < capsule.size + 5:  # Click radius
                    self.selected_capsule = capsule
                    self.capsule_selected.emit(capsule)
                    self.update()
                    break
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for hover effects"""
        mx = event.position().x()
        my = event.position().y()
        click_pos = np.array([mx, my], dtype=np.float32)
        
        self.hovered_capsule = None
        
        for capsule in self.capsules:
            dist = np.linalg.norm(click_pos - capsule.position)
            if dist < capsule.size + 10:  # Hover radius
                self.hovered_capsule = capsule
                self.capsule_hovered.emit(capsule)
                break
        
        self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        if delta != 0:
            factor = 1.2 ** (delta / 240)
            self.zoom_level = np.float32(max(0.2, min(3.0, self.zoom_level * factor)))
            self.update()

        


class _QuadNode:
    """Internal quadtree node for Barnes-Hut force computation."""
    def __init__(self, cx, cy, half_size):
        self.cx = cx
        self.cy = cy
        self.h = half_size
        self.mass = 0.0
        self.com_x = 0.0
        self.com_y = 0.0
        self.particle = None  # (idx, x, y, m)
        self.children = [None, None, None, None]

    def contains(self, x, y):
        return (x >= self.cx - self.h and x <= self.cx + self.h and
                y >= self.cy - self.h and y <= self.cy + self.h)

    def _child_index(self, x, y):
        # 0: NW, 1: NE, 2: SW, 3: SE
        left = x < self.cx
        top = y < self.cy
        if left and top:
            return 0
        if not left and top:
            return 1
        if left and not top:
            return 2
        return 3

    def subdivide(self):
        hs = self.h / 2.0
        self.children[0] = _QuadNode(self.cx - hs, self.cy - hs, hs)
        self.children[1] = _QuadNode(self.cx + hs, self.cy - hs, hs)
        self.children[2] = _QuadNode(self.cx - hs, self.cy + hs, hs)
        self.children[3] = _QuadNode(self.cx + hs, self.cy + hs, hs)

    def insert(self, idx, x, y, m):
        if not self.contains(x, y):
            return False

        # update center-of-mass and mass
        total_mass = self.mass + m
        if total_mass > 0:
            self.com_x = (self.com_x * self.mass + x * m) / total_mass
            self.com_y = (self.com_y * self.mass + y * m) / total_mass
        self.mass = total_mass

        if self.particle is None and all(ch is None for ch in self.children):
            self.particle = (idx, x, y, m)
            return True

        if any(ch is None for ch in self.children) and self.particle is not None:
            if self.children[0] is None:
                self.subdivide()
            old_idx, ox, oy, om = self.particle
            self.particle = None
            ci = self._child_index(ox, oy)
            self.children[ci].insert(old_idx, ox, oy, om)

        if self.children[0] is None:
            self.subdivide()
        ci = self._child_index(x, y)
        return self.children[ci].insert(idx, x, y, m)

    def compute_force_on(self, idx, x, y, theta, G, softening):
        fx = 0.0
        fy = 0.0

        if self.mass == 0.0:
            return 0.0, 0.0

        dx = self.com_x - x
        dy = self.com_y - y
        dist = math.hypot(dx, dy) + 1e-12
        s = self.h * 2.0

        if self.particle is not None:
            pidx, px, py, pm = self.particle
            if pidx == idx:
                return 0.0, 0.0

        if s / dist < theta or all(ch is None for ch in self.children):
            invr3 = 1.0 / ((dist*dist + softening*softening) * dist)
            f = G * self.mass * invr3
            fx += f * dx
            fy += f * dy
            return fx, fy

        for ch in self.children:
            if ch is not None:
                cfx, cfy = ch.compute_force_on(idx, x, y, theta, G, softening)
                fx += cfx
                fy += cfy

        return fx, fy

# ============================================================================
# CHAT WIDGET WITH SMILEY AVATAR
# ============================================================================

class ChatWidget(QWidget):
    """Chat interface with smiley avatar"""
    
    message_sent = pyqtSignal(str)
    
    def __init__(self, chatbot: ChatBot, parent=None):
        super().__init__(parent)
        self.chatbot = chatbot
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Create header with avatar and title
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Smiley avatar
        self.avatar = SmileyAvatar()
        header_layout.addWidget(self.avatar)
        
        # Title
        title_label = QLabel("ROCA Chat Assistant")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18pt;
                font-weight: bold;
                padding: 10px;
            }
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(header_widget)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: rgb(25, 25, 35);
                color: white;
                border: 1px solid rgb(50, 50, 60);
                border-radius: 8px;
                font-family: Consolas;
                font-size: 11pt;
                padding: 10px;
            }
        """)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 5, 0, 0)
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                background-color: rgb(40, 40, 50);
                color: white;
                border: 1px solid rgb(60, 60, 80);
                border-radius: 5px;
                padding: 8px;
                font-size: 11pt;
                selection-background-color: rgb(100, 150, 255);
            }
            QLineEdit:focus {
                border: 2px solid rgb(100, 150, 255);
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        send_button = QPushButton("Send")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: rgb(0, 150, 255);
                color: white;
                border: none;
                padding: 8px 20px;
                font-weight: bold;
                border-radius: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: rgb(0, 170, 255);
            }
            QPushButton:pressed {
                background-color: rgb(0, 130, 230);
            }
        """)
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)
        
        layout.addWidget(input_widget)
        
        # Connect avatar mood to chatbot
        self.chatbot.add_message("Chat initialized!", "System")
        self.update_chat_display()
    
    def send_message(self):
        """Send message from input"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Clear input
        self.message_input.clear()
        
        # Add user message
        self.chatbot.add_message(message, "You", is_user=True)
        
        # Update display and emit signal; let window route to brains/assistant
        self.update_chat_display()
        self.message_sent.emit(message)
    
    def update_chat_display(self):
        """Update chat display with all messages"""
        self.chat_display.clear()
        
        html = """
        <style>
            body {
                background-color: #1a1a2a;
                color: white;
                font-family: Consolas, monospace;
                margin: 0;
                padding: 10px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 10px;
                max-width: 80%;
            }
            .user-message {
                background-color: #2a3a2a;
                margin-left: 20%;
                border-left: 3px solid #8f8;
            }
            .bot-message {
                background-color: #2a2a3a;
                margin-right: 20%;
                border-left: 3px solid #6cf;
            }
            .system-message {
                background-color: #3a2a2a;
                margin: 5px 10%;
                border-left: 3px solid #f66;
                font-style: italic;
            }
            .sender {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .time {
                color: #aaa;
                font-size: 9pt;
                float: right;
            }
        </style>
        """
        
        for msg in self.chatbot.messages:
            css_class = "user-message" if msg.is_user else "bot-message"
            if msg.sender == "System":
                css_class = "system-message"
            
            time_str = msg.timestamp.strftime("%H:%M:%S")
            html += f"""
            <div class="message {css_class}">
                <div class="sender">{msg.sender} <span class="time">{time_str}</span></div>
                <div>{msg.text}</div>
            </div>
            """
        
        self.chat_display.setHtml(html)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

# ============================================================================
# MAIN WINDOW
# ============================================================================

class NumpyROCAWindow(QMainWindow):
    """Main window with numpy optimizations"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize systems
        self.settings = NumpySettings()
        self.chatbot = ChatBot()
        self.brain = None
        
        # Setup UI
        self.setWindowTitle("ROCA - NumPy Enhanced AI Assistant")
        self.setGeometry(100, 100, 1600, 900)

        # Set icon
        self.setWindowIcon(self.create_numpy_icon())

        # Initialize UI
        self.init_ui()

        # Enable drag and drop for JSON files
        self.setAcceptDrops(True)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("ROCA Ready - NumPy optimized orbital visualization active! Drag & drop JSON files to load configurations!")
        
        # Router controller (optional)
        try:
            # create a HybridRouter instance if available and inject it
            try:
                router_instance = HybridRouter() if HybridRouter is not None else None
            except Exception:
                router_instance = None

            self.router_controller = RouterController(router=router_instance)
            if self.router_controller.is_available():
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  Router: available")
            else:
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  Router: fallback")
        except Exception:
            self.router_controller = RouterController()

        # Time capsule controller (optional)
        try:
            self.time_capsule_controller = TimeCapsuleController()
            if self.time_capsule_controller.is_available():
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  TimeCapsule: available")
            else:
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  TimeCapsule: offline")
        except Exception:
            self.time_capsule_controller = TimeCapsuleController()

        # Learning orbital visualizer (optional)
        try:
            self.learning_visualizer = LearningOrbitalVisualizerController()
            if self.learning_visualizer.is_available():
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  LearningVis: available")
            else:
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  LearningVis: offline")
        except Exception:
            self.learning_visualizer = LearningOrbitalVisualizerController()
        
        # Embedded learning visualizer pane (prefer shader OpenGL widget, fallback Matplotlib canvas)
        try:
            self.learning_dock = QtWidgets.QDockWidget("Learning Visualizer", self)
            # Prefer high-fidelity shader widget when available
            try:
                if ShaderVisualizerWidget is not None and USE_GL:
                    try:
                        self.shader_visualizer = ShaderVisualizerWidget(self)
                        self.learning_dock.setWidget(self.shader_visualizer)
                    except Exception:
                        self.shader_visualizer = None
                else:
                    self.shader_visualizer = None
            except Exception:
                self.shader_visualizer = None

            # If shader widget not available, use the matplotlib fallback pane
            if getattr(self, 'shader_visualizer', None) is None:
                try:
                    self.learning_pane = LearningVisualizerPane(self)
                    self.learning_dock.setWidget(self.learning_pane)
                except Exception:
                    self.learning_pane = None
            else:
                self.learning_pane = None

            # Use PyQt6 enum style and guard the dock visibility call
            try:
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.learning_dock)
            except Exception:
                self.learning_dock = None
                # start hidden by default (only if the dock was created)
                if getattr(self, 'learning_dock', None) is not None:
                    try:
                        self.learning_dock.setVisible(False)
                    except Exception:
                        pass

            # Try to attach external visualizer controller to the embedded pane (if supported)
            try:
                target_widget = getattr(self, 'shader_visualizer', None) or getattr(self, 'learning_pane', None)
                if getattr(self, 'learning_visualizer', None) is not None and target_widget is not None:
                    try:
                        self.learning_visualizer.attach_widget(target_widget)
                    except Exception:
                        pass
            except Exception:
                pass

            # Timer for auto-updating the learning visualizer (disabled by default)
            try:
                self._learning_vis_auto_update = False
                self._learning_vis_timer = QTimer(self)
                self._learning_vis_timer.setInterval(1000)  # 1s default
                self._learning_vis_timer.timeout.connect(self.refresh_learning_visualizer)
            except Exception:
                self._learning_vis_timer = None
        except Exception:
            self.learning_pane = None
            self.learning_dock = None
            self.shader_visualizer = None
        # Performance tuning controller
        try:
            self.performance_tuner = PerformanceTuningController()
            if self.performance_tuner.is_available():
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  PerfTuner: available")
            else:
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  PerfTuner: offline")
        except Exception:
            self.performance_tuner = PerformanceTuningController()
        # Whether to apply tuner recommendations to app parameters
        self._apply_perf_recommendations = False
        
        # Voice controller (if available)
        self.voice = None
        self.mic_on = False
        try:
            if VoiceController is not None and self.settings.get_numpy('voice_enabled', True):
                # Force offline-only voice controller to avoid network access
                self.voice = VoiceController(on_text=self._on_voice_text, offline_only=True)
        except Exception:
            self.voice = None

        # Instantiate AutonomousBrain if available
        try:
            if autonomous_brain_mod is not None:
                try:
                    self.brain = autonomous_brain_mod.AutonomousBrain()
                except Exception:
                    # Fallback: try factory or alternate name
                    self.brain = None
        except Exception:
            self.brain = None
        
        # Instantiate coder assistant if available (prefer enhanced)
        self.coder_assistant = None
        try:
            if enhanced_ai_assistant_mod is not None:
                try:
                    self.coder_assistant = enhanced_ai_assistant_mod.EnhancedAICodingAssistant()
                except Exception:
                    self.coder_assistant = None
            if self.coder_assistant is None and ai_coding_assistant_mod is not None:
                try:
                    self.coder_assistant = ai_coding_assistant_mod.AICodingAssistant()
                except Exception:
                    self.coder_assistant = None
        except Exception:
            self.coder_assistant = None
        # Initialize session paths and attempt to load previous session
        try:
            self._init_session_paths()
            self.load_session()
        except Exception:
            pass

    def _init_session_paths(self):
        """Initialize session directory and file paths for persistence."""
        try:
            self.session_dir = os.path.join(os.getcwd(), 'roca_session')
            os.makedirs(self.session_dir, exist_ok=True)
            self.capsule_store_path = os.path.join(self.session_dir, 'capsules_store.json')
            self.universal_store_path = os.path.join(self.session_dir, 'universal_capsules.json')
        except Exception:
            # Fallback to cwd files
            self.session_dir = os.getcwd()
            self.capsule_store_path = os.path.join(self.session_dir, 'capsules_store.json')
            self.universal_store_path = os.path.join(self.session_dir, 'universal_capsules.json')

    def load_session(self):
        """Load persisted session data: capsules and universal capsules."""
        # Load capsule store
        try:
            if getattr(self, 'orbital_widget', None) is not None and getattr(self.orbital_widget, 'store', None) is not None:
                store = self.orbital_widget.store
                if os.path.exists(self.capsule_store_path) and hasattr(store, 'load'):
                    try:
                        store.load(self.capsule_store_path)
                        # populate widget from store
                        for cap in store.list_all():
                            try:
                                self.orbital_widget.add_capsule_object(cap)
                            except Exception:
                                continue
                        try:
                            self.status_bar.showMessage('Session loaded from ' + self.capsule_store_path)
                        except Exception:
                            pass
                    except Exception:
                        pass

                # Load universal capsules if supported
                if hasattr(store, 'load_universal') and os.path.exists(self.universal_store_path):
                    try:
                        store.load_universal(self.universal_store_path)
                        try:
                            self.status_bar.showMessage((self.status_bar.currentMessage() or '') + '  |  Universal capsules loaded')
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

    def save_session(self):
        """Save current session: capsules and universal capsules."""
        try:
            if getattr(self, 'orbital_widget', None) is not None and getattr(self.orbital_widget, 'store', None) is not None:
                store = self.orbital_widget.store
                # save capsule store
                try:
                    if hasattr(store, 'save'):
                        store.save(self.capsule_store_path)
                except Exception:
                    pass

                # save universal capsules
                try:
                    if hasattr(store, 'save_universal'):
                        store.save_universal(self.universal_store_path)
                except Exception:
                    pass

                try:
                    self.status_bar.showMessage('Session saved')
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, event):
        """Override close to persist session before exit."""
        try:
            self.save_session()
        except Exception:
            pass
        try:
            super().closeEvent(event)
        except Exception:
            event.accept()
    
    def create_numpy_icon(self):
        """Create NumPy-themed application icon"""
        pixmap = QtGui.QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Draw NumPy-style icon
        painter.setBrush(QBrush(QColor(100, 200, 255, 200)))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(12, 12, 40, 40)
        
        # Draw matrix representation
        painter.setBrush(QBrush(QColor(255, 200, 100, 180)))
        painter.drawRect(20, 20, 8, 8)
        painter.drawRect(32, 20, 8, 8)
        painter.drawRect(20, 32, 8, 8)
        painter.drawRect(32, 32, 8, 8)
        
        painter.setBrush(QBrush(QColor(200, 100, 255, 180)))
        painter.drawEllipse(28, 28, 8, 8)
        
        painter.end()
        return QIcon(pixmap)
    
    def init_ui(self):
        """Initialize enhanced UI"""
        # --- Session menu: Save/Load hooks for capsule persistence ---
        try:
            menubar = self.menuBar()
            session_menu = menubar.addMenu("Session")

            save_action = QtGui.QAction("Save Session", self)
            save_action.setShortcut(QKeySequence("Ctrl+S"))
            save_action.triggered.connect(lambda: self.save_session())
            session_menu.addAction(save_action)

            load_action = QtGui.QAction("Load Session", self)
            load_action.setShortcut(QKeySequence("Ctrl+L"))
            load_action.triggered.connect(lambda: self.load_session())
            session_menu.addAction(load_action)
        except Exception:
            pass
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Chat with avatar
        self.chat_widget = ChatWidget(self.chatbot)
        self.chat_widget.message_sent.connect(self.on_chat_message)
        main_splitter.addWidget(self.chat_widget)
        
        # Right side: Orbital visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Orbital widget
        self.orbital_widget = NumpyOrbitalWidget()
        self.orbital_widget.capsule_selected.connect(self.on_capsule_selected)
        right_layout.addWidget(self.orbital_widget)
        # Prefer NUMA-optimized or Evolutionary capsule store for the orbital widget
        try:
            try:
                from threadripper_core.NUMA_Aware_Capsule_Distribution import NUMAOptimizedCapsuleStore
            except Exception:
                NUMAOptimizedCapsuleStore = None

            try:
                from threadripper_core.SELF_EXPANSION_MECHANISMS import EvolutionaryCapsuleStore
            except Exception:
                try:
                    from Cognition.SELF_EXPANSION_MECHANISMS import EvolutionaryCapsuleStore
                except Exception:
                    EvolutionaryCapsuleStore = None

            router_obj = None
            try:
                router_obj = getattr(self.router_controller, 'router', None)
            except Exception:
                router_obj = None

            chosen = None
            threadripper_mode = os.environ.get('THREADRIPPER_MODE', '0').lower() in ('1', 'true', 'yes')

            # Try NUMA-optimized first when appropriate
            if NUMAOptimizedCapsuleStore is not None:
                try:
                    import numa as _numa_check
                    numa_nodes = getattr(_numa_check, 'get_max_node', lambda: 0)()
                    multicore_numa = (numa_nodes + 1) > 1
                except Exception:
                    multicore_numa = False

                if multicore_numa or threadripper_mode:
                    try:
                        self.orbital_widget.store = NUMAOptimizedCapsuleStore()
                        chosen = 'numa'
                    except Exception:
                        chosen = None

            # Fall back to EvolutionaryCapsuleStore
            if chosen is None and EvolutionaryCapsuleStore is not None:
                try:
                    self.orbital_widget.store = EvolutionaryCapsuleStore(router=router_obj)
                    chosen = 'evolutionary'
                except Exception:
                    try:
                        self.orbital_widget.store = EvolutionaryCapsuleStore()
                        chosen = 'evolutionary'
                    except Exception:
                        chosen = None

            # Last-resort: legacy CapsuleStore or in-memory
            if chosen is None:
                try:
                    from core.capsule_store import CapsuleStore
                    self.orbital_widget.store = CapsuleStore()
                except Exception:
                    try:
                        from threadripper_core.NUMA_Aware_Capsule_Distribution import _InMemoryStore
                        self.orbital_widget.store = _InMemoryStore()
                    except Exception:
                        # best effort: default to None
                        self.orbital_widget.store = getattr(self.orbital_widget, 'store', None)

            # attempt to load default store files into new store
            try:
                default_path = os.path.join(os.getcwd(), 'capsules_store.json')
                if os.path.exists(default_path) and hasattr(self.orbital_widget.store, 'load'):
                    self.orbital_widget.store.load(default_path)
                    for cap in self.orbital_widget.store.list_all():
                        try:
                            self.orbital_widget.add_capsule_object(cap)
                        except Exception:
                            continue
            except Exception:
                pass

            try:
                default_univ = os.path.join(os.getcwd(), 'universal_capsules.json')
                if os.path.exists(default_univ) and hasattr(self.orbital_widget.store, 'load_universal'):
                    self.orbital_widget.store.load_universal(default_univ)
            except Exception:
                pass

            # Ensure the store has an expansion_orchestrator that can consult the router.
            try:
                try:
                    from SELF_EXPANSION_MECHANISMS import ExpansionOrchestrator
                except Exception:
                    try:
                        from Cognition.SELF_EXPANSION_MECHANISMS import ExpansionOrchestrator
                    except Exception:
                        ExpansionOrchestrator = None

                if ExpansionOrchestrator is None:
                    class ExpansionOrchestrator:
                        def __init__(self, router=None):
                            self.router = router
                        def execute_expansion(self, capsule, trigger, context):
                            try:
                                if self.router is not None:
                                    try:
                                        guidance = self.router.route({'target': 'expansion', 'context': context})
                                        if isinstance(guidance, dict) and 'path' in guidance:
                                            trigger['router_path'] = guidance.get('path')
                                    except Exception:
                                        pass
                                try:
                                    capsule.generated_children = getattr(capsule, 'generated_children', [])
                                    capsule.generated_children.append(f"expansion:{trigger.get('action')}")
                                except Exception:
                                    pass
                            except Exception:
                                pass

                # attach orchestrator to the store
                try:
                    if not hasattr(self.orbital_widget, 'store') or getattr(self.orbital_widget, 'store', None) is None:
                        try:
                            from core.capsule_store import CapsuleStore
                            self.orbital_widget.store = CapsuleStore()
                        except Exception:
                            from threadripper_core.NUMA_Aware_Capsule_Distribution import _InMemoryStore
                            self.orbital_widget.store = _InMemoryStore()
                    router_obj = getattr(self.router_controller, 'router', None)
                    try:
                        self.orbital_widget.store.expansion_orchestrator = ExpansionOrchestrator(router=router_obj)
                    except Exception:
                        try:
                            self.orbital_widget.store.expansion_orchestrator = ExpansionOrchestrator()
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            # Instantiate the threadripper self-expansion engine if available
            try:
                try:
                    from threadripper_core.SELF_EXPANSION_MECHANISMS import CosmicExpansionEngine
                except Exception:
                    try:
                        from SELF_EXPANSION_MECHANISMS import CosmicExpansionEngine
                    except Exception:
                        CosmicExpansionEngine = None

                if CosmicExpansionEngine is not None:
                    try:
                        router_obj = getattr(self.router_controller, 'router', None)
                        visualizer_obj = getattr(self, 'learning_visualizer', None) or getattr(self, 'shader_visualizer', None) or getattr(self, 'learning_pane', None)
                        # Bridge visualizer events back into the main window for UI notifications
                        class _VisualizerBridge:
                            def __init__(self, app, vis):
                                self._app = app
                                self._vis = vis

                            def show_creation_context(self, capsule, suggestions):
                                if self._vis is not None and hasattr(self._vis, 'show_creation_context'):
                                    try:
                                        self._vis.show_creation_context(capsule, suggestions)
                                    except Exception:
                                        pass

                            def show_expansion_event(self, capsule):
                                # First, forward to the wrapped visualizer
                                if self._vis is not None and hasattr(self._vis, 'show_expansion_event'):
                                    try:
                                        self._vis.show_expansion_event(capsule)
                                    except Exception:
                                        pass
                                # Then notify the main UI
                                try:
                                    self._app.on_expansion_event(capsule)
                                except Exception:
                                    # Last resort: print to stdout
                                    print('Expansion event:', getattr(capsule, 'id', getattr(capsule, 'get', lambda k, d=None: str(capsule))('id', str(capsule))))

                        visualizer_obj = _VisualizerBridge(self, visualizer_obj)
                        store_obj = getattr(self.orbital_widget, 'store', None)
                        self.cosmic_expansion_engine = CosmicExpansionEngine(store_obj, router_obj, visualizer_obj)
                    except Exception:
                        self.cosmic_expansion_engine = None
                else:
                    self.cosmic_expansion_engine = None
            except Exception:
                self.cosmic_expansion_engine = None


        # Enhanced controls
        # End of try block for self-expansion engine setup
        # (Add except/finally as required by Python syntax)
        # No-op except to close the try
        except Exception:
            pass
        controls_group = QGroupBox("Orbital Controls")
        controls_layout = QGridLayout()
        
        # Row 0: Rotation
        controls_layout.addWidget(QLabel("Rotation:"), 0, 0)
        rotation_slider = QSlider(Qt.Orientation.Horizontal)
        rotation_slider.setRange(0, 100)
        rotation_slider.setValue(50)
        rotation_slider.valueChanged.connect(lambda v: self.orbital_widget.set_rotation_speed(np.float32(v/10.0)))
        controls_layout.addWidget(rotation_slider, 0, 1)
        
        # Row 1: Auto-rotate and physics
        auto_rotate_check = QCheckBox("Auto Rotate")
        auto_rotate_check.setChecked(True)
        auto_rotate_check.stateChanged.connect(lambda s: self.orbital_widget.set_auto_rotate(s == Qt.CheckState.Checked.value))
        controls_layout.addWidget(auto_rotate_check, 1, 0)
        
        physics_check = QCheckBox("Physics")
        physics_check.setChecked(True)
        physics_check.stateChanged.connect(lambda s: self.orbital_widget.set_physics_enabled(s == Qt.CheckState.Checked.value))
        controls_layout.addWidget(physics_check, 1, 1)
        
        # Row 2: Capsule operations
        add_capsule_btn = QPushButton("+ Add Capsule")
        add_capsule_btn.clicked.connect(self.add_capsule_from_chat)
        controls_layout.addWidget(add_capsule_btn, 2, 0)
        
        clear_capsules_btn = QPushButton("Clear All")
        clear_capsules_btn.clicked.connect(self.clear_capsules)
        controls_layout.addWidget(clear_capsules_btn, 2, 1)
        
        controls_group.setLayout(controls_layout)
        controls_group.setMaximumHeight(120)
        right_layout.addWidget(controls_group)

        # Expose method to get evolutionary store if available
        def get_evolutionary_store():
            try:
                # prefer explicit EvolutionaryCapsuleStore type if present
                from SELF_EXPANSION_MECHANISMS import EvolutionaryCapsuleStore
                if isinstance(getattr(self, 'store', None), EvolutionaryCapsuleStore):
                    return self.store
            except Exception:
                pass
            # fallback: return store if it has evolutionary features
            return getattr(self, 'store', None)

        self.get_evolutionary_store = get_evolutionary_store
        
        # Stats panel
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("Capsules: 0 | FPS: 0")
        self.stats_label.setStyleSheet("color: white; font-family: monospace;")
        stats_layout.addWidget(self.stats_label)
        
        stats_group.setLayout(stats_layout)
        stats_group.setMaximumHeight(60)
        right_layout.addWidget(stats_group)
        
        main_splitter.addWidget(right_widget)
        
        # Set splitter sizes
        main_splitter.setSizes([600, 1000])
        
        main_layout.addWidget(main_splitter)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Update timer for stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)
    
    def create_menu_bar(self):
        """Create enhanced menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Session", self)
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        new_action.triggered.connect(self.new_session)
        file_menu.addAction(new_action)
        
        export_action = QAction("Export Visualization", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_visualization)
        file_menu.addAction(export_action)
        
        load_json_action = QAction("Load JSON File", self)
        load_json_action.setShortcut(QKeySequence("Ctrl+L"))
        load_json_action.triggered.connect(self.load_json_dialog)
        file_menu.addAction(load_json_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        toggle_physics_action = QAction("Toggle Physics", self)
        toggle_physics_action.triggered.connect(self.toggle_physics)
        view_menu.addAction(toggle_physics_action)

        # Auto-update Learning Visualizer (checkable)
        self.auto_learning_vis_action = QAction("Auto-update Learning Visualizer", self)
        self.auto_learning_vis_action.setCheckable(True)
        self.auto_learning_vis_action.setChecked(False)
        self.auto_learning_vis_action.toggled.connect(self.toggle_learning_visualizer_auto)
        view_menu.addAction(self.auto_learning_vis_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        analyze_action = QAction("Analyze Capsules", self)
        analyze_action.triggered.connect(self.analyze_capsules)
        tools_menu.addAction(analyze_action)

        cosmic_action = QAction("Cosmic Creation Flow", self)
        cosmic_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        cosmic_action.triggered.connect(self.start_cosmic_creation)
        tools_menu.addAction(cosmic_action)

        style_action = QAction("Style Transfer", self)
        style_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        style_action.triggered.connect(self.start_style_transfer)
        tools_menu.addAction(style_action)

        predictive_action = QAction("Predictive Reasoning", self)
        predictive_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
        predictive_action.triggered.connect(self.start_predictive_reasoning)
        tools_menu.addAction(predictive_action)

        # Router status
        router_action = QAction("Router Status", self)
        router_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        router_action.triggered.connect(self.show_router_status)
        tools_menu.addAction(router_action)

        # Time Capsule
        timecapsule_action = QAction("Time Capsule", self)
        timecapsule_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        timecapsule_action.triggered.connect(self.start_time_capsule)
        tools_menu.addAction(timecapsule_action)

        # Learning Orbital Visualizer
        learning_vis_action = QAction("Learning Orbital Visualizer", self)
        learning_vis_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        learning_vis_action.triggered.connect(self.start_learning_visualizer)
        tools_menu.addAction(learning_vis_action)

        # Performance tuning
        perf_action = QAction("Performance Tuning", self)
        perf_action.setShortcut(QKeySequence("Ctrl+Shift+U"))
        perf_action.triggered.connect(self.start_performance_tuning)
        tools_menu.addAction(perf_action)
        
        # Add NumPy info menu
        numpy_menu = menubar.addMenu("NumPy")
        
        info_action = QAction("NumPy Info", self)
        info_action.triggered.connect(self.show_numpy_info)
        numpy_menu.addAction(info_action)

    def start_cosmic_creation(self):
        """Prompt the user and run the cosmic creation flow."""
        try:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox
            text, ok = QInputDialog.getText(self, "Cosmic Creation", "Describe the creation (text or paste data):")
            if not ok or not text:
                return

            # Ask the router (if present) for any contextual guidance before creation
            router_suggestion = None
            try:
                if getattr(self, 'router_controller', None) is not None:
                    req = {'query': str(text), 'context': {'capsule_count': len(self.orbital_widget.capsules)}}
                    router_suggestion = self.router_controller.route(req)
            except Exception:
                router_suggestion = None

            # Prefer DI-safe workflow from threadripper_core if available
            suggestions = []
            cap = None
            try:
                try:
                    from threadripper_core.Cosmic_Creation_Workflow import cosmic_creation_workflow
                except Exception:
                    try:
                        from Cosmic_Creation_Workflow import cosmic_creation_workflow
                    except Exception:
                        cosmic_creation_workflow = None

                if cosmic_creation_workflow is not None:
                    # Build best-effort dependency set from this window
                    canvas = getattr(self, 'orbital_widget', None)
                    capsule_store = getattr(canvas, 'store', None) if canvas is not None else None
                    growth_engine = getattr(self, 'growth_engine', None)
                    voice_twin = getattr(self, 'voice', None)
                    expansion_engine = getattr(self, 'cosmic_expansion_engine', None) or (getattr(capsule_store, 'expansion_orchestrator', None) if capsule_store is not None else None)
                    router = None
                    try:
                        router = getattr(self.router_controller, 'router', None)
                    except Exception:
                        router = None
                    visualizer = getattr(self, 'learning_visualizer', None) or getattr(self, 'shader_visualizer', None) or getattr(self, 'learning_pane', None)

                    workflow_result = cosmic_creation_workflow(canvas=canvas,
                                                              capsule_store=capsule_store,
                                                              growth_engine=growth_engine,
                                                              voice_twin=voice_twin,
                                                              expansion_engine=expansion_engine,
                                                              router=router,
                                                              visualizer=visualizer,
                                                              user_input=str(text))
                    cap = workflow_result.get('created_capsule')
                    suggestions = workflow_result.get('suggestions', [])
                    notes = workflow_result.get('notes', [])
                    if notes:
                        for n in notes:
                            self.chatbot.add_message(str(n), 'System')
                else:
                    # Fallback to existing class-based flow
                    flow = CosmicCreationFlow(self.orbital_widget, store=getattr(self.orbital_widget, 'store', None),
                                               sequencer=getattr(self.orbital_widget, 'genome_sequencer', None))
                    result = flow.run(str(text))
                    cap = result.get('capsule')
                    suggestions = result.get('suggestions', [])
            except Exception as e:
                # fallback and inform
                try:
                    self.chatbot.add_message(f'Cosmic creation failed: {e}', 'System')
                except Exception:
                    pass
                return
            # Merge router suggestions into the suggestions list if present
            try:
                if isinstance(router_suggestion, dict):
                    if 'path' in router_suggestion and router_suggestion['path']:
                        suggestions = suggestions + [str(x) for x in router_suggestion.get('path', [])]
                    elif 'suggestions' in router_suggestion and router_suggestion['suggestions']:
                        suggestions = suggestions + [str(x) for x in router_suggestion.get('suggestions', [])]
            except Exception:
                pass

            msg = f"Created capsule: {cap.content[:60]}\nSuggestions: {', '.join(suggestions)}"
            self.chatbot.add_message(msg, 'System')
            self.chat_widget.update_chat_display()
            QMessageBox.information(self, 'Cosmic Creation', msg)
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Cosmic Creation', f'Failed: {e}')
            except Exception:
                print('Cosmic Creation failed:', e)

    def start_style_transfer(self):
        """Prompt for a style name and apply style transfer to selected or all capsules."""
        try:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox
            style, ok = QInputDialog.getText(self, "Style Transfer", "Enter target style name (e.g. 'vibrant', 'pastel'):")
            if not ok or not style:
                return

            controller = StyleTransferController()
            if self.orbital_widget.selected_capsule is not None:
                controller.apply_style_to_capsule(self.orbital_widget.selected_capsule, style)
                msg = f"Applied style '{style}' to selected capsule."
            else:
                controller.apply_style_to_all(self.orbital_widget.capsules, style)
                msg = f"Applied style '{style}' to all capsules."

            self.chatbot.add_message(msg, 'System')
            self.chat_widget.update_chat_display()
            QMessageBox.information(self, 'Style Transfer', msg)
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Style Transfer', f'Failed: {e}')
            except Exception:
                print('Style Transfer failed:', e)

    def start_time_capsule(self):
        """Create and save a time-capsule snapshot of the current capsules."""
        try:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox
            name, ok = QInputDialog.getText(self, "Time Capsule", "Enter a name for this time capsule:")
            if not ok or not name:
                return
            desc, _ = QInputDialog.getText(self, "Time Capsule", "Optional description:")

            # Optionally run a high-performance simulation to produce predicted updates
            sim_results = None
            try:
                if getattr(self, 'time_capsule_controller', None) is not None and self.time_capsule_controller.is_available():
                    sim_results = self.time_capsule_controller.simulate(getattr(self.orbital_widget, 'capsules', []))
            except Exception:
                sim_results = None

            # Serialize current capsule states (and attach simulation results if any)
            caps = []
            for idx, c in enumerate(getattr(self.orbital_widget, 'capsules', [])):
                try:
                    cap_dict = {
                        'position': np.array(c.position, dtype=np.float64).tolist(),
                        'velocity': np.array(c.velocity, dtype=np.float64).tolist(),
                        'color': np.array(getattr(c, 'color', [1.0,1.0,1.0])).tolist(),
                        'size': float(getattr(c, 'size', 1.0)),
                        'metadata': getattr(c, 'metadata', {})
                    }
                    # If simulation returned an array-like of updates, try to attach per-capsule result
                    if sim_results is not None:
                        try:
                            # support numpy arrays, lists, or list-of-tuples
                            arr = np.asarray(sim_results)
                            if arr.ndim == 2 and arr.shape[0] == len(getattr(self.orbital_widget, 'capsules', [])):
                                # assume each row is [x,y] or [x,y,z]
                                row = arr[idx]
                                cap_dict['simulated_position'] = [float(row[0]), float(row[1])]
                        except Exception:
                            pass

                    caps.append(cap_dict)
                except Exception:
                    continue

            data = {'capsules': caps, 'count': len(caps)}

            # If simulation produced final positions, show predictions overlay
            try:
                if sim_results is not None:
                    arr = np.asarray(sim_results)
                    if arr.ndim == 2 and arr.shape[0] == len(caps):
                        # convert to Nx2 float64 array for visualization
                        final = np.array(arr[:, :2], dtype=np.float64)
                        try:
                            # show overlay in orbital widget
                            if getattr(self, 'orbital_widget', None) is not None and hasattr(self.orbital_widget, 'show_predictions'):
                                self.orbital_widget.show_predictions(final)
                        except Exception:
                            pass
            except Exception:
                pass

            path = ''
            try:
                if getattr(self, 'time_capsule_controller', None) is not None:
                    path = self.time_capsule_controller.save_time_capsule(name, desc, data)
            except Exception:
                path = ''

            if path:
                msg = f"Time capsule saved: {path}"
            else:
                msg = "Failed to save time capsule."

            self.chatbot.add_message(msg, 'System')
            self.chat_widget.update_chat_display()
            QMessageBox.information(self, 'Time Capsule', msg)
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Time Capsule', f'Failed: {e}')
            except Exception:
                print('Time Capsule failed:', e)

    def start_performance_tuning(self):
        """Open a small dialog to run performance tuning and enable auto-tuning."""
        try:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Performance Tuning')
            layout = QVBoxLayout()

            info_label = QLabel('Performance Tuning Controls')
            layout.addWidget(info_label)

            apply_btn = QPushButton('Apply Optimizations')
            layout.addWidget(apply_btn)

            auto_chk = QCheckBox('Enable auto-tune (adjust based on metrics)')
            auto_chk.setChecked(False)
            layout.addWidget(auto_chk)

            result_text = QTextEdit()
            result_text.setReadOnly(True)
            result_text.setFixedHeight(160)
            layout.addWidget(result_text)

            def do_apply():
                try:
                    res = {}
                    if getattr(self, 'performance_tuner', None) is not None:
                        res = self.performance_tuner.optimize_system()
                    result_text.setPlainText(json.dumps(res, indent=2))
                except Exception as e:
                    result_text.setPlainText(f'Error: {e}')

            def toggle_auto(state: bool):
                try:
                    if getattr(self, 'performance_tuner', None) is None:
                        return
                    if state:
                        # start timer
                        if not hasattr(self, '_perf_tune_timer') or self._perf_tune_timer is None:
                            self._perf_tune_timer = QTimer(self)
                            self._perf_tune_timer.setInterval(2000)
                            self._perf_tune_timer.timeout.connect(self._run_auto_tune)
                        self._perf_tune_timer.start()
                    else:
                        if hasattr(self, '_perf_tune_timer') and self._perf_tune_timer is not None:
                            self._perf_tune_timer.stop()
                except Exception:
                    pass

            apply_btn.clicked.connect(do_apply)
            auto_chk.toggled.connect(toggle_auto)
            # Checkbox to apply tuner recommendations to app parameters
            apply_chk = QCheckBox('Apply tuner recommendations to app parameters')
            apply_chk.setChecked(False)
            layout.addWidget(apply_chk)

            def apply_recommendations_toggled(state: bool):
                try:
                    self._apply_perf_recommendations = bool(state)
                except Exception:
                    pass

            apply_chk.toggled.connect(apply_recommendations_toggled)

            dlg.setLayout(layout)
            dlg.exec()
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Performance Tuning', f'Failed: {e}')
            except Exception:
                print('Performance tuning failed:', e)

    def _run_auto_tune(self):
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent
            }
            if getattr(self, 'performance_tuner', None) is not None:
                res = self.performance_tuner.adjust_based_on_metrics(metrics)
                # optional: log or show small toast in status
                self.status_bar.showMessage(f"PerfTune adjusted: CPU {metrics['cpu_percent']}% MEM {metrics['memory_percent']}%")
                # Apply recommendations to app parameters if requested
                try:
                    if getattr(self, '_apply_perf_recommendations', False):
                        # Heuristic mapping from tuner cache sizes to app parameters
                        cache_sizes = res.get('cache_sizes', {}) if isinstance(res, dict) else {}
                        cap_mb = cache_sizes.get('capsule_cache_mb', None)
                        if cap_mb is not None:
                            if cap_mb >= 1024:
                                self.orbital_widget.numba_pairwise_limit = 5000
                                self.orbital_widget.bh_threshold = 300
                                self.orbital_widget.update_interval = max(1/120.0, float(self.orbital_widget.update_interval) * 0.8)
                                self.orbital_widget.use_numba = True and NUMBA_AVAILABLE
                            elif cap_mb >= 512:
                                self.orbital_widget.numba_pairwise_limit = 3000
                                self.orbital_widget.bh_threshold = 200
                                self.orbital_widget.update_interval = max(1/90.0, float(self.orbital_widget.update_interval) * 0.9)
                                self.orbital_widget.use_numba = True and NUMBA_AVAILABLE
                            else:
                                self.orbital_widget.numba_pairwise_limit = 1500
                                self.orbital_widget.bh_threshold = 100
                                # don't speed up further
                                self.orbital_widget.use_numba = NUMBA_AVAILABLE

                        # If the tuner reports memory limits, be conservative on memory pressure
                        mem_limits = res.get('memory_limits', {}) if isinstance(res, dict) else {}
                        system_reserve = mem_limits.get('system_reserve', None)
                        if system_reserve is not None:
                            # If system reserve is small, reduce frame rate to save memory
                            total_reserved = float(system_reserve)
                            if total_reserved < 2.0:
                                # very low reserve -> slow down updates
                                self.orbital_widget.update_interval = float(self.orbital_widget.update_interval) * 1.5

                        # Inform user in status bar
                        self.status_bar.showMessage(self.status_bar.currentMessage() + f"  |  Perf applied: cap={cap_mb}")
                except Exception:
                    pass
        except Exception:
            pass

    def start_learning_visualizer(self):
        """Invoke the optional LearningOrbitalVisualizer to draw an enhanced view."""
        try:
            caps = getattr(self.orbital_widget, 'capsules', [])
            success = False
            if getattr(self, 'learning_visualizer', None) is not None:
                try:
                    success = self.learning_visualizer.draw(caps)
                except Exception:
                    success = False

            # Ensure embedded pane updates as well (fallback or alongside external vis)
            try:
                if getattr(self, 'learning_pane', None) is not None:
                    self.learning_pane.update(caps)
                    if getattr(self, 'learning_dock', None) is not None:
                        self.learning_dock.setVisible(True)
            except Exception:
                pass

            if success:
                self.chatbot.add_message('Learning Orbital Visualizer: drew enhanced view.', 'System')
                self.chat_widget.update_chat_display()
                QMessageBox.information(self, 'Learning Visualizer', 'Enhanced orbital view drawn.')
            else:
                # Inform user but still show embedded pane (fallback)
                self.chatbot.add_message('Learning Orbital Visualizer not available or failed; using embedded view.', 'System')
                self.chat_widget.update_chat_display()
                QMessageBox.information(self, 'Learning Visualizer', 'Embedded visualizer shown.')
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Learning Visualizer', f'Failed: {e}')
            except Exception:
                print('Learning visualizer failed:', e)

    def toggle_learning_visualizer_auto(self, checked: bool):
        """Enable or disable periodic auto-updates of the learning visualizer pane."""
        try:
            self._learning_vis_auto_update = bool(checked)
            if self._learning_vis_timer is None:
                return
            if self._learning_vis_auto_update:
                # Start timer
                self._learning_vis_timer.start()
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  LearningVis: auto-update ON")
            else:
                # Stop timer
                self._learning_vis_timer.stop()
                self.status_bar.showMessage(self.status_bar.currentMessage() + "  |  LearningVis: auto-update OFF")
        except Exception:
            pass

    def refresh_learning_visualizer(self):
        """Called by the auto-update timer to refresh the learning visualizer pane."""
        try:
            caps = getattr(self.orbital_widget, 'capsules', [])
            # Prefer external visualizer if available
            drawn = False
            try:
                if getattr(self, 'learning_visualizer', None) is not None:
                    drawn = self.learning_visualizer.draw(caps)
            except Exception:
                drawn = False

            # Always update embedded pane as fallback or alongside external
            try:
                # Prefer shader visualizer if present
                if getattr(self, 'shader_visualizer', None) is not None:
                    try:
                        self.shader_visualizer.update_capsules(caps)
                        if getattr(self, 'learning_dock', None) is not None:
                            self.learning_dock.setVisible(True)
                    except Exception:
                        pass
                elif getattr(self, 'learning_pane', None) is not None:
                    try:
                        self.learning_pane.update(caps)
                        if getattr(self, 'learning_dock', None) is not None:
                            self.learning_dock.setVisible(True)
                    except Exception:
                        pass
            except Exception:
                pass

            return drawn
        except Exception:
            return False

    def start_predictive_reasoning(self):
        """Prompt for prediction horizon and show predicted positions overlay."""
        try:
            from PyQt6.QtWidgets import QInputDialog, QMessageBox
            text, ok = QInputDialog.getInt(self, "Predictive Reasoning", "Predict how many frames ahead?", value=60, min=1, max=10000)
            if not ok:
                return
            steps = int(text)
            dt = 1.0 / 60.0

            reasoner = PredictiveReasoner()

            # If a heavier predictive creativity engine is available, ask it for suggestions.
            used_creative = False
            try:
                if getattr(reasoner, 'engine', None) is not None:
                    user_history = getattr(self.chatbot, 'context_memory', None)
                    current_context = {
                        'capsules': [
                            {
                                'position': np.array(c.position, dtype=np.float64).tolist(),
                                'velocity': np.array(c.velocity, dtype=np.float64).tolist()
                            }
                            for c in self.orbital_widget.capsules
                        ]
                    }
                    res = reasoner.engine.predict_creative_needs(user_history, current_context)
                    if isinstance(res, dict):
                        pre = res.get('pre_generated_capsules')
                        if pre and isinstance(pre, list):
                            pts = []
                            for item in pre:
                                if isinstance(item, dict) and 'position' in item:
                                    pts.append(np.array(item['position'], dtype=np.float64))
                            if len(pts) > 0:
                                final = np.stack(pts, axis=0)
                                self.orbital_widget.show_predictions(final)
                                msg = f"Creative engine returned {len(pts)} pre-generated capsules."
                                used_creative = True
            except Exception:
                used_creative = False

            if not used_creative:
                paths = reasoner.predict_paths(self.orbital_widget.capsules, dt=dt, steps=steps)
                # show final predicted positions and also keep the path (optional)
                final = paths[:, -1, :]
                self.orbital_widget.show_predictions(final)

                msg = f"Predicted {len(final)} capsule positions {steps} frames ahead."
            self.chatbot.add_message(msg, 'System')
            self.chat_widget.update_chat_display()
            QMessageBox.information(self, 'Predictive Reasoning', msg)
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Predictive Reasoning', f'Failed: {e}')
            except Exception:
                print('Predictive reasoning failed:', e)
    
    def on_expansion_event(self, capsule):
        """Handle expansion events from the self-expansion engine.

        Shows a brief message in the status bar and records a system message
        in the chatbot view if available.
        """
        try:
            cid = None
            try:
                cid = capsule.get('id') if isinstance(capsule, dict) else getattr(capsule, 'id', None)
            except Exception:
                cid = str(capsule)

            msg = f"Expansion produced: {cid}"
            try:
                self.status_bar.showMessage(msg)
            except Exception:
                pass

            try:
                if getattr(self, 'chatbot', None) is not None:
                    self.chatbot.add_message(msg, 'System')
            except Exception:
                print(msg)
        except Exception:
            # swallow errors to avoid disturbing UI
            pass
    
    def on_chat_message(self, message: str):
        """Handle chat messages"""
        # Handle slash commands first
        if message.startswith("/"):
            cmd = message.strip().lower()
            if cmd.startswith("/mic"):
                if "on" in cmd:
                    started = False
                    if self.voice:
                        try:
                            started = self.voice.start()
                        except Exception:
                            started = False
                    if started:
                        self.mic_on = True
                        self.chatbot.add_message("Microphone listening started.", "System")
                        self.chat_widget.update_chat_display()
                        self.status_bar.showMessage("Mic: ON")
                    else:
                        self.chatbot.add_message("Voice system not available.", "System")
                        self.chat_widget.update_chat_display()
                    return
                elif "off" in cmd:
                    try:
                        if self.voice:
                            self.voice.stop()
                    except Exception:
                        pass
                    self.mic_on = False
                    self.chatbot.add_message("Microphone stopped.", "System")
                    self.chat_widget.update_chat_display()
                    self.status_bar.showMessage("Mic: OFF")
                    return
            # Unknown slash command — ignore for now
            # New: route command -> call optional router
            if cmd.startswith("/route"):
                try:
                    payload = message[len("/route"):].strip()
                    req = {'query': payload or cmd, 'context': {'capsules': len(self.orbital_widget.capsules)}}
                    if getattr(self, 'router_controller', None) is not None:
                        res = self.router_controller.route(req)
                        # Summarize routing result in chat
                        summary = ''
                        if isinstance(res, dict):
                            if 'path' in res:
                                summary = f"Route path: {res.get('path')}"
                            elif res.get('fallback'):
                                summary = "Router fallback used."
                            elif res.get('error'):
                                summary = "Router error during routing."
                            else:
                                summary = str(res)
                        else:
                            summary = str(res)

                        self.chatbot.add_message(f"/route -> {summary}", 'System')
                        self.chat_widget.update_chat_display()
                        return
                except Exception as e:
                    self.chatbot.add_message(f"/route failed: {e}", 'System')
                    self.chat_widget.update_chat_display()
                    return

            return

    def show_router_status(self):
        """Display the router availability and basic info."""
        try:
            if getattr(self, 'router_controller', None) is None:
                QMessageBox.information(self, 'Router Status', 'Router controller not initialized.')
                return

            if self.router_controller.is_available():
                QMessageBox.information(self, 'Router Status', 'HybridRouter is available and active.')
            else:
                QMessageBox.information(self, 'Router Status', 'HybridRouter not available — using fallback router.')
        except Exception as e:
            try:
                QMessageBox.warning(self, 'Router Status', f'Error checking router: {e}')
            except Exception:
                print('Router status check failed:', e)

        # Extract potential capsule content from message
        if len(message.split()) >= 2:
            # Simple keyword extraction
            keywords = ["capsule", "concept", "theory", "method", "idea"]
            if any(keyword in message.lower() for keyword in keywords):
                self.add_capsule_from_message(message)

        # If message looks like a code request, prefer coder assistant
        code_keywords = ["code", "implement", "fix", "bug", "debug", "function", "class", "script"]
        try:
            if self.coder_assistant is not None and any(k in message.lower() for k in code_keywords):
                try:
                    if hasattr(self.coder_assistant, '_process_user_input'):
                        resp = self.coder_assistant._process_user_input(message)
                    elif hasattr(self.coder_assistant, 'process_input'):
                        resp = self.coder_assistant.process_input(message)
                    else:
                        resp = None
                except Exception:
                    resp = None

                if resp:
                    self.chatbot.add_message(resp, "ROCA")
                    self.chat_widget.update_chat_display()
                    return
        except Exception:
            pass

        # Route to coder assistant for code-related queries if available
        try:
            message_lower = message.lower()
            handled = False
            if getattr(self, 'coder_assistant', None) is not None:
                if message_lower.startswith('/assistant') or 'code' in message_lower or 'capsule' in message_lower:
                    try:
                        resp = self.coder_assistant._process_user_input(message)
                        if resp:
                            self.chatbot.add_message(resp, "AICodingAssistant")
                            self.chat_widget.update_chat_display()
                            handled = True
                    except Exception:
                        handled = False

            # If not handled by coder assistant, forward to autonomous brain
            if not handled and getattr(self, 'brain', None) is not None:
                try:
                    response = None
                    if hasattr(self.brain, 'roca_reason'):
                        try:
                            response = self.brain.roca_reason(message)
                        except Exception:
                            response = None
                    if not response and hasattr(self.brain, 'get_tribunal_response'):
                        try:
                            response = self.brain.get_tribunal_response(message)
                        except Exception:
                            response = None

                    if response:
                        # Add brain response to chat
                        self.chatbot.add_message(response, "ROCA")
                        self.chat_widget.update_chat_display()
                except Exception:
                    pass
        except Exception:
            pass
    
    def add_capsule_from_chat(self):
        """Add capsule based on recent chat"""
        if self.chatbot.messages:
            last_message = self.chatbot.messages[-1].text
            if len(last_message) > 10:  # Avoid very short messages
                capsule = self.orbital_widget.add_capsule(last_message[:50])
                self.chatbot.add_message(f"Added capsule: '{capsule.content}' to orbital view!", "System")
                self.chat_widget.update_chat_display()
    
    def add_capsule_from_message(self, message: str):
        """Add capsule from message content"""
        # Simple content extraction
        content = message[:40] + "..." if len(message) > 40 else message
        capsule = self.orbital_widget.add_capsule(content)
        self.chatbot.add_message(f"Created capsule: '{capsule.content}'", "System")
        self.chat_widget.update_chat_display()
    
    def on_capsule_selected(self, capsule):
        """Handle capsule selection"""
        self.chatbot.add_message(f"Selected capsule: {capsule.content} ({capsule.kind}, certainty: {capsule.certainty:.2f})", "System")
        self.chat_widget.update_chat_display()
        self.status_bar.showMessage(f"Selected: {capsule.content}")
    
    def clear_capsules(self):
        """Clear all capsules"""
        reply = QMessageBox.question(
            self, 'Clear Capsules',
            'Clear all capsules from orbital view?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.orbital_widget.capsules = []
            self.orbital_widget.selected_capsule = None
            self.orbital_widget.hovered_capsule = None
            self.orbital_widget.update()
            self.chatbot.add_message("Cleared all capsules from orbital view.", "System")
            self.chat_widget.update_chat_display()

    def _on_voice_text(self, text: str):
        """Callback from voice system when speech is recognized."""
        try:
            if not text:
                return
            # Inject recognized text into the chat input and send
            self.chat_widget.message_input.setText(text)
            # Use the chat widget's send_message to ensure consistent flow
            self.chat_widget.send_message()
        except Exception:
            pass
    
    def toggle_physics(self):
        """Toggle physics simulation"""
        self.orbital_widget.physics_enabled = not self.orbital_widget.physics_enabled
        status = "enabled" if self.orbital_widget.physics_enabled else "disabled"
        self.chatbot.add_message(f"Physics {status}", "System")
        self.chat_widget.update_chat_display()
    
    def new_session(self):
        """Start new session"""
        reply = QMessageBox.question(
            self, 'New Session',
            'Start new session? This will clear chat and capsules.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.chatbot.messages = []
            self.clear_capsules()
            self.chatbot.add_message("New session started! How can I help you today? 😊", "ROCA")
            self.chat_widget.update_chat_display()
    
    def export_visualization(self):
        """Export visualization data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                data = {
                    'export_date': datetime.now().isoformat(),
                    'capsules': [c.to_numpy_dict() for c in self.orbital_widget.capsules],
                    'chat_messages': len(self.chatbot.messages),
                    'numpy_version': np.__version__
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self.chatbot.add_message(f"Exported visualization data to {os.path.basename(file_path)}", "System")
                self.chat_widget.update_chat_display()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def load_json_dialog(self):
        """Open file dialog to load JSON files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load JSON Files", "Json/", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_paths:
            loaded_files = []
            for file_path in file_paths:
                try:
                    success = self.load_json_file(file_path)
                    if success:
                        loaded_files.append(os.path.basename(file_path))
                except Exception as e:
                    QMessageBox.warning(self, "Load Error", 
                                      f"Error loading {os.path.basename(file_path)}:\n{str(e)}")
            
            if loaded_files:
                self.chatbot.add_message(f"Successfully loaded JSON files: {', '.join(loaded_files)}", "System")
                self.chat_widget.update_chat_display()
                QMessageBox.information(self, "Success", f"Loaded {len(loaded_files)} JSON file(s)")
            else:
                QMessageBox.warning(self, "No Files Loaded", "No valid JSON files were loaded.")
    
    def analyze_capsules(self):
        """Analyze capsule statistics using numpy"""
        if not self.orbital_widget.capsules:
            self.chatbot.add_message("No capsules to analyze", "System")
            return
        
        # Use numpy for analysis
        certainties = np.array([c.certainty for c in self.orbital_widget.capsules], dtype=np.float32)
        radii = np.array([c.orbit_radius for c in self.orbital_widget.capsules], dtype=np.float32)
        
        analysis = f"""
        Capsule Analysis (NumPy Powered):
        --------------------------------
        Total capsules: {len(self.orbital_widget.capsules)}
        Average certainty: {np.mean(certainties):.3f}
        Certainty std: {np.std(certainties):.3f}
        Average orbit radius: {np.mean(radii):.3f}
        Min/Max certainty: {np.min(certainties):.3f} / {np.max(certainties):.3f}
        """
        
        self.chatbot.add_message(analysis, "System")
        self.chat_widget.update_chat_display()
    
    def show_numpy_info(self):
        """Show NumPy information"""
        info = f"""
        NumPy Information:
        -----------------
        NumPy version: {np.__version__}
        Array operations: Vectorized
        Physics: {'Enabled' if self.orbital_widget.physics_enabled else 'Disabled'}
        Capsule count: {len(self.orbital_widget.capsules)}
        Performance: Optimized with numpy arrays
        """
        
        QMessageBox.information(self, "NumPy Info", info)
    
    def update_stats(self):
        """Update statistics display"""
        fps = int(1 / self.orbital_widget.update_interval) if self.orbital_widget.update_interval > 0 else 0
        stats = f"Capsules: {len(self.orbital_widget.capsules)} | FPS: {fps} | Physics: {'ON' if self.orbital_widget.physics_enabled else 'OFF'}"
        self.stats_label.setText(stats)
    
    # Drag and Drop Methods for JSON Files
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().endswith('.json') for url in urls):
                event.acceptProposedAction()
                self.status_bar.showMessage("Drop JSON files to load configurations!")
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events for JSON files"""
        urls = event.mimeData().urls()
        loaded_files = []
        created_capsules = []
        
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.endswith('.json'):
                try:
                    result = self.load_json_file(file_path)
                    if result:
                        loaded_files.append(os.path.basename(file_path))
                        if result.get('capsules_created', 0) > 0:
                            created_capsules.extend(result.get('capsule_names', []))
                except Exception as e:
                    self.chatbot.add_message(f"Error loading {os.path.basename(file_path)}: {str(e)}", "System")
        
        if loaded_files:
            self.chatbot.add_message(f"Successfully loaded JSON files: {', '.join(loaded_files)}", "System")
            if created_capsules:
                capsule_count = len(created_capsules)
                capsule_names = created_capsules[:3]  # Show first 3
                names_str = ', '.join(capsule_names)
                if capsule_count > 3:
                    names_str += f" and {capsule_count - 3} more"
                self.chatbot.add_message(f"🧠 Created {capsule_count} orbital capsule(s): {names_str}", "System")
            self.chat_widget.update_chat_display()
            self.status_bar.showMessage(f"Loaded {len(loaded_files)} JSON file(s), {len(created_capsules)} capsule(s) created")
        else:
            self.status_bar.showMessage("No valid JSON files loaded")
    
    def load_json_file(self, file_path: str) -> dict:
        """Load and process a JSON file based on its content type
        
        Returns:
            dict: {'success': bool, 'capsules_created': int, 'capsule_names': list, 'type': str}
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = os.path.basename(file_path).lower()
            result = {'success': False, 'capsules_created': 0, 'capsule_names': [], 'type': 'unknown'}
            
            # Handle different JSON file types
            if 'chat_history' in filename or filename.startswith('chat'):
                result['success'] = self.load_chat_history(data)
                result['type'] = 'chat_history'
            elif 'knowledge_base' in filename or filename.startswith('knowledge'):
                result['success'] = self.load_knowledge_base(data)
                result['type'] = 'knowledge_base'
            elif 'config' in filename or filename.startswith('settings'):
                result['success'] = self.load_config(data)
                result['type'] = 'config'
            elif 'capsule' in filename or 'orbital' in filename:
                capsule_result = self.load_capsule_data(data)
                result['success'] = capsule_result['success']
                result['capsules_created'] = capsule_result['capsules_created']
                result['capsule_names'] = capsule_result['capsule_names']
                result['type'] = 'capsule_data'
            elif filename == 'user_profile.json' or filename.startswith('user_profile'):
                result['success'] = self.load_user_profile(data)
                result['type'] = 'user_profile'
            elif filename == 'style_profiles.json' or filename.startswith('style'):
                style_result = self.load_style_profiles(data)
                result['success'] = style_result['success']
                result['capsules_created'] = style_result['capsules_created']
                result['capsule_names'] = style_result['capsule_names']
                result['type'] = 'style_profiles'
            elif 'dataset' in filename or 'fine_tune' in filename:
                training_result = self.load_training_data(data)
                result['success'] = training_result['success']
                result['capsules_created'] = training_result['capsules_created']
                result['capsule_names'] = training_result['capsule_names']
                result['type'] = 'training_data'
            else:
                # Generic JSON loading - add as capsule
                generic_result = self.load_generic_json(data, filename)
                result['success'] = generic_result['success']
                result['capsules_created'] = generic_result['capsules_created']
                result['capsule_names'] = generic_result['capsule_names']
                result['type'] = 'generic_json'
                
            return result
                
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {file_path}: {e}")
            return {'success': False, 'capsules_created': 0, 'capsule_names': [], 'type': 'invalid_json', 'error': str(e)}
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {'success': False, 'capsules_created': 0, 'capsule_names': [], 'type': 'error', 'error': str(e)}
    
    def load_chat_history(self, data: dict) -> dict:
        """Load chat history from JSON
        
        Returns:
            dict: {'success': bool, 'messages_loaded': int}
        """
        try:
            messages_loaded = 0
            
            if isinstance(data, list):
                # List of messages
                for msg in data[-20:]:  # Load last 20 messages
                    sender = msg.get('sender', 'Unknown')
                    text = msg.get('text', msg.get('message', ''))
                    is_user = msg.get('is_user', sender.lower() == 'user')
                    self.chatbot.add_message(text, sender, is_user)
                    messages_loaded += 1
            elif isinstance(data, dict) and 'messages' in data:
                # Structured chat history
                for msg in data['messages'][-20:]:
                    sender = msg.get('sender', 'Unknown')
                    text = msg.get('text', msg.get('message', ''))
                    is_user = msg.get('is_user', sender.lower() == 'user')
                    self.chatbot.add_message(text, sender, is_user)
                    messages_loaded += 1
            
            return {
                'success': True,
                'messages_loaded': messages_loaded
            }
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return {
                'success': False,
                'messages_loaded': 0,
                'error': str(e)
            }
    
    def load_knowledge_base(self, data: dict) -> dict:
        """Load knowledge base from JSON
        
        Returns:
            dict: {'success': bool, 'capsules_created': int, 'capsule_names': list}
        """
        try:
            capsules_created = 0
            capsule_names = []
            
            if self.coder_assistant and hasattr(self.coder_assistant, 'load_knowledge_base'):
                self.coder_assistant.load_knowledge_base(data)
                self.chatbot.add_message("Knowledge base loaded and integrated", "System")
                return {
                    'success': True,
                    'capsules_created': 0,
                    'capsule_names': []
                }
            else:
                # Fallback: add knowledge as capsules
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            capsule_name = f"Knowledge: {key}"
                            self.add_capsule_from_text(f"{capsule_name}: {value}")
                            capsules_created += 1
                            capsule_names.append(capsule_name)
                
                return {
                    'success': True,
                    'capsules_created': capsules_created,
                    'capsule_names': capsule_names
                }
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {
                'success': False,
                'capsules_created': 0,
                'capsule_names': [],
                'error': str(e)
            }
    
    def load_config(self, data: dict) -> dict:
        """Load configuration from JSON
        
        Returns:
            dict: {'success': bool, 'settings_updated': int}
        """
        try:
            settings_updated = 0
            
            # Update settings
            for key, value in data.items():
                if hasattr(self.settings, f'set_numpy'):
                    self.settings.set_numpy(key, value)
                    settings_updated += 1
                elif hasattr(self.settings, 'setValue'):
                    self.settings.setValue(key, value)
                    settings_updated += 1
            
            self.chatbot.add_message("Configuration loaded and applied", "System")
            return {
                'success': True,
                'settings_updated': settings_updated
            }
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                'success': False,
                'settings_updated': 0,
                'error': str(e)
            }
    
    def load_capsule_data(self, data: dict) -> dict:
        """Load capsule data from JSON
        
        Returns:
            dict: {'success': bool, 'capsules_created': int, 'capsule_names': list}
        """
        try:
            capsules_created = 0
            capsule_names = []
            
            if isinstance(data, list):
                # List of capsules
                for capsule_data in data:
                    name = self.create_capsule_from_data(capsule_data)
                    if name:
                        capsules_created += 1
                        capsule_names.append(name)
            elif isinstance(data, dict) and 'capsules' in data:
                # Structured capsule data
                for capsule_data in data['capsules']:
                    name = self.create_capsule_from_data(capsule_data)
                    if name:
                        capsules_created += 1
                        capsule_names.append(name)
            else:
                # Single capsule
                name = self.create_capsule_from_data(data)
                if name:
                    capsules_created += 1
                    capsule_names.append(name)
            
            return {
                'success': True,
                'capsules_created': capsules_created,
                'capsule_names': capsule_names
            }
        except Exception as e:
            print(f"Error loading capsule data: {e}")
            return {
                'success': False,
                'capsules_created': 0,
                'capsule_names': [],
                'error': str(e)
            }
    
    def load_user_profile(self, data: dict) -> dict:
        """Load user profile from JSON
        
        Returns:
            dict: {'success': bool, 'personality_loaded': bool, 'preferences_loaded': bool}
        """
        try:
            personality_loaded = False
            preferences_loaded = False
            
            # Update chatbot personality based on profile
            if 'personality' in data and self.chatbot.personality_system:
                self.chatbot.personality_system.load_profile(data['personality'])
                personality_loaded = True
            
            if 'preferences' in data:
                self.chatbot.add_message(f"User preferences loaded: {data['preferences']}", "System")
                preferences_loaded = True
            
            return {
                'success': True,
                'personality_loaded': personality_loaded,
                'preferences_loaded': preferences_loaded
            }
        except Exception as e:
            print(f"Error loading user profile: {e}")
            return {
                'success': False,
                'personality_loaded': False,
                'preferences_loaded': False,
                'error': str(e)
            }
    
    def load_style_profiles(self, data: dict) -> dict:
        """Load style profiles from JSON
        
        Returns:
            dict: {'success': bool, 'capsules_created': int, 'capsule_names': list}
        """
        try:
            capsules_created = 0
            capsule_names = []
            
            # Add style capsules
            if isinstance(data, dict):
                for style_name, style_data in data.items():
                    capsule_name = f"Style: {style_name}"
                    self.add_capsule_from_text(f"{capsule_name} - {str(style_data)}")
                    capsules_created += 1
                    capsule_names.append(capsule_name)
            
            return {
                'success': True,
                'capsules_created': capsules_created,
                'capsule_names': capsule_names
            }
        except Exception as e:
            print(f"Error loading style profiles: {e}")
            return {
                'success': False,
                'capsules_created': 0,
                'capsule_names': [],
                'error': str(e)
            }
    
    def load_training_data(self, data: dict) -> dict:
        """Load training data from JSON
        
        Returns:
            dict: {'success': bool, 'capsules_created': int, 'capsule_names': list}
        """
        try:
            capsules_created = 0
            capsule_names = []
            
            # Add training examples as capsules
            if isinstance(data, list):
                for item in data[:10]:  # Limit to 10 items
                    if isinstance(item, dict):
                        text = item.get('text', item.get('input', str(item)))
                        capsule_name = f"Training: {text[:30]}..."
                        self.add_capsule_from_text(f"{capsule_name}: {text}")
                        capsules_created += 1
                        capsule_names.append(capsule_name)
            
            self.chatbot.add_message("Training data loaded as capsules", "System")
            return {
                'success': True,
                'capsules_created': capsules_created,
                'capsule_names': capsule_names
            }
        except Exception as e:
            print(f"Error loading training data: {e}")
            return {
                'success': False,
                'capsules_created': 0,
                'capsule_names': [],
                'error': str(e)
            }
    
    def load_generic_json(self, data: dict, filename: str) -> dict:
        """Load generic JSON as capsule
        
        Returns:
            dict: {'success': bool, 'capsules_created': int, 'capsule_names': list}
        """
        try:
            # Convert JSON to readable text
            json_text = json.dumps(data, indent=2)
            capsule_name = f"JSON Data ({filename})"
            self.add_capsule_from_text(f"{capsule_name}: {json_text[:500]}...")
            return {
                'success': True,
                'capsules_created': 1,
                'capsule_names': [capsule_name]
            }
        except Exception as e:
            print(f"Error loading generic JSON: {e}")
            return {
                'success': False,
                'capsules_created': 0,
                'capsule_names': [],
                'error': str(e)
            }
    
    def create_capsule_from_data(self, capsule_data: dict) -> str:
        """Create a capsule from JSON data
        
        Returns:
            str: The name of the created capsule, or empty string if failed
        """
        try:
            # Extract capsule properties
            name = capsule_data.get('name', capsule_data.get('id', 'Unknown'))
            content = capsule_data.get('content', capsule_data.get('text', str(capsule_data)))
            
            # Create capsule
            self.add_capsule_from_text(f"{name}: {content}")
            return name
        except Exception as e:
            print(f"Error creating capsule from data: {e}")
            return ""
    
    def add_capsule_from_text(self, text: str):
        """Add a capsule from text content"""
        try:
            # Create a new capsule
            capsule = NumpyCapsule(content=text)
            capsule.name = text[:50] + "..." if len(text) > 50 else text
            capsule.certainty = 0.8
            
            # Add to orbital widget
            self.orbital_widget.add_capsule_object(capsule)
            
        except Exception as e:
            print(f"Error adding capsule from text: {e}")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ROCA_NUMPY")
    app.setOrganizationName("ROCA Project")
    
    # Set dark theme
    app.setStyle('Fusion')
    
    # Create and show main window
    window = NumpyROCAWindow()
    window.show()
    
    # Start application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()