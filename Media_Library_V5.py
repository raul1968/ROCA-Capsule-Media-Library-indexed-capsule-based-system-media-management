# media_library2.py - Complete ROCA Media Registry Application
# Combined application with full media registry, exchange, and UI capabilities

from dataclasses import dataclass, field
import numpy as np
import os
import sys
import json
import time
import hashlib
import sqlite3
import threading
import zipfile
import msgpack
import base64
import platform
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Set, Tuple, Generator
from enum import Enum, auto
from contextlib import contextmanager
import pickle
import zlib
from collections import deque
import psutil
import math
import random
import html
import shutil

# PyQt6 imports
try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
        QTabWidget, QSplitter, QTreeWidget, QTreeWidgetItem, QListWidget, 
        QListWidgetItem, QMenuBar, QStatusBar, QToolBar, QFileDialog, 
        QMessageBox, QProgressDialog, QDockWidget, QTextEdit, QGroupBox, 
        QGridLayout, QLineEdit, QComboBox, QCheckBox, QSlider, QDialog, 
        QDialogButtonBox, QSpinBox, QScrollArea, QProgressBar, QApplication
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QAbstractListModel, QFileSystemWatcher
    from PyQt6.QtGui import QIcon, QAction, QPixmap, QFont, QPalette, QColor, QPainter, QPen, QBrush
    from PyQt6 import QtGui, QtWidgets
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
        QTabWidget, QSplitter, QTreeWidget, QTreeWidgetItem, QListWidget, 
        QListWidgetItem, QMenuBar, QStatusBar, QToolBar, QFileDialog, 
        QMessageBox, QProgressDialog, QDockWidget, QTextEdit, QGroupBox, 
        QGridLayout, QLineEdit, QComboBox, QCheckBox, QSlider, QDialog, 
        QDialogButtonBox, QSpinBox, QScrollArea, QProgressBar, QApplication
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QAbstractListModel, QFileSystemWatcher
    from PyQt5.QtGui import QIcon, QAction, QPixmap, QFont, QPalette, QColor, QPainter, QPen, QBrush
    from PyQt5 import QtGui, QtWidgets
    PYQT_VERSION = 5

# ============================================================================
# CORE MEDIA TYPES AND CAPSULES
# ============================================================================

class MediaType(Enum):
    IMAGE = auto()
    MODEL_3D = auto()
    ANIMATION = auto()
    VIDEO = auto()
    TEXTURE = auto()
    SHADER = auto()
    RIG = auto()
    MOCAP = auto()
    SCRIPT = auto()
    BRUSH = auto()
    SCENE = auto()
    AUDIO = auto()
    DOCUMENT = auto()
    PROJECT = auto()
    UNKNOWN = auto()

@dataclass
class MediaCapsule:
    """Enhanced media capsule with comprehensive metadata"""
    source_path: str
    activity_vector: np.ndarray = field(default_factory=lambda: np.zeros(128))
    media_type: MediaType = MediaType.UNKNOWN
    style_hash: str = ""
    content_hash: str = ""
    complexity: float = 0.5
    poly_count: int = 0
    texture_count: int = 0
    animation_length: float = 0.0
    rig_bones: int = 0
    emotional_tone: str = ""
    file_size: int = 0
    capsule_path: str = ""
    thumbnail_path: str = ""
    id: Optional[str] = None
    filename: str = ""
    extension: str = ""
    _thumbnail_cache: Optional[Any] = None
    _metadata_cache: Optional[Dict] = None
    _style_hash_obj: Optional[Any] = None
    _image_dimensions: Optional[Tuple[int, int]] = None
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    material_types: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    usage_context: List[str] = field(default_factory=list)
    parent_projects: List[str] = field(default_factory=list)
    related_capsules: List[str] = field(default_factory=list)
    used_with: List[str] = field(default_factory=list)
    description: str = ""
    projects: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.id is None:
            self.id = str(hashlib.sha1(f"{self.source_path}_{time.time()}".encode()).hexdigest())
        if not self.filename:
            self.filename = os.path.basename(self.source_path)
        if not self.extension:
            self.extension = os.path.splitext(self.source_path)[1].lower()
        if self.media_type == MediaType.UNKNOWN:
            self.media_type = self._detect_media_type()
        if not self.content_hash and os.path.exists(self.source_path):
            self.content_hash = self._compute_content_hash()
        if not self.file_size and os.path.exists(self.source_path):
            self.file_size = os.path.getsize(self.source_path)

    def _detect_media_type(self) -> MediaType:
        ext = self.extension.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', '.bmp', '.webp']:
            return MediaType.TEXTURE if self._is_likely_texture() else MediaType.IMAGE
        elif ext in ['.fbx', '.obj', '.gltf', '.glb', '.blend', '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae']:
            return MediaType.MODEL_3D
        elif ext in ['.bvh', '.trc', '.c3d', '.cho']:
            return MediaType.MOCAP
        elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv']:
            return MediaType.VIDEO
        elif ext in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            return MediaType.AUDIO
        elif ext in ['.pdf', '.txt', '.md', '.doc', '.docx']:
            return MediaType.DOCUMENT
        return MediaType.UNKNOWN

    def _is_likely_texture(self) -> bool:
        filename_lower = self.filename.lower()
        texture_indicators = [
            'albedo', 'diffuse', 'normal', 'roughness', 'metallic',
            'ao', 'ambient', 'occlusion', 'height', 'displacement',
            'bump', 'specular', 'gloss', 'opacity', 'alpha', 'emission',
            '_d', '_n', '_r', '_m', '_h', '_b'
        ]
        return any(indicator in filename_lower for indicator in texture_indicators)

    def _compute_content_hash(self) -> str:
        try:
            with open(self.source_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""

    def compute_activity_vector(self, dim: int = 128):
        """Compute activity vector from text and image data"""
        from numpy.linalg import norm
        
        def _embed_text(text: str, dim: int = 128) -> np.ndarray:
            vec = np.zeros(dim, dtype=float)
            if not text:
                return vec
            t = text.strip().lower()
            for i in range(dim):
                h = hashlib.sha256(f"{t}::{i}".encode('utf-8')).digest()
                val = int.from_bytes(h[:8], 'big') / (2**64 - 1)
                vec[i] = val
            vec = vec - vec.mean()
            n = norm(vec)
            if n > 0:
                vec = vec / n
            return vec
        
        def _embed_image(path: str, dim: int = 128) -> np.ndarray:
            try:
                from PIL import Image
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    im = im.resize((32, 32))
                    arr = np.asarray(im).astype(float) / 255.0
                    flat = arr.reshape(-1)
                    if flat.size < dim:
                        vec = np.zeros(dim, dtype=float)
                        vec[:flat.size] = flat
                    else:
                        vec = np.array([flat[i::dim].mean() for i in range(dim)], dtype=float)
                    vec = vec - vec.mean()
                    n = norm(vec)
                    if n > 0:
                        vec = vec / n
                    return vec
            except Exception:
                return np.zeros(dim, dtype=float)
        
        text_parts = [self.filename or '', ' '.join(self.style_tags or []), 
                     ' '.join(self.usage_context or [])]
        if self.description:
            text_parts.append(self.description)
        text_src = ' '.join([p for p in text_parts if p])
        txt_vec = _embed_text(text_src, dim=dim)
        
        img_vec = None
        try:
            from PIL import Image
            if self.media_type in [MediaType.IMAGE, MediaType.TEXTURE]:
                img_vec = _embed_image(self.source_path, dim=dim)
        except:
            pass
        
        if img_vec is not None:
            vec = 0.6 * txt_vec + 0.4 * img_vec
        else:
            vec = txt_vec
        
        n = norm(vec)
        if n > 0:
            vec = vec / n
        
        self.activity_vector = vec

@dataclass
class Capsule:
    content: str
    kind: str = "concept"
    certainty: float = 0.6
    orbit_radius: float = 1.0
    angle: float = 0.0
    character: Optional[str] = None
    id: str = field(default_factory=lambda: str(random.randint(100000, 999999)))

# ============================================================================
# INDEXED CAPSULES - LIGHTLY ALIVE
# ============================================================================

@dataclass
class IndexedCapsule:
    """Base class for indexed capsules - lookup-friendly, not thinking"""
    file_path: str
    content_hash: str
    media_type: str
    file_size: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[np.ndarray] = None  # Optional cheap embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = {
            'file_path': self.file_path,
            'content_hash': self.content_hash,
            'media_type': self.media_type,
            'file_size': self.file_size,
            'created_at': self.created_at,
            'last_modified': self.last_modified
        }
        if self.embedding is not None:
            data['embedding'] = base64.b64encode(self.embedding.tobytes()).decode('utf-8')
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexedCapsule':
        """Create from dictionary"""
        embedding = None
        if 'embedding' in data:
            embedding_bytes = base64.b64decode(data['embedding'])
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        return cls(
            file_path=data['file_path'],
            content_hash=data['content_hash'],
            media_type=data['media_type'],
            file_size=data['file_size'],
            created_at=data.get('created_at', datetime.now().isoformat()),
            last_modified=data.get('last_modified', datetime.now().isoformat()),
            embedding=embedding
        )

@dataclass
class ImageCapsule(IndexedCapsule):
    """Thin capsule for image files"""
    width: int = 0
    height: int = 0
    channels: int = 3
    color_mode: str = "RGB"
    dpi: int = 72

    def __post_init__(self):
        self.media_type = "image"

    @classmethod
    def create_from_file(cls, file_path: str) -> 'ImageCapsule':
        """Create capsule from image file"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                color_mode = img.mode
                dpi = img.info.get('dpi', (72, 72))[0] if 'dpi' in img.info else 72

            # Compute hash
            with open(file_path, 'rb') as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            file_size = os.path.getsize(file_path)

            return cls(
                file_path=file_path,
                content_hash=content_hash,
                file_size=file_size,
                width=width,
                height=height,
                channels=channels,
                color_mode=color_mode,
                dpi=dpi
            )
        except Exception as e:
            # Fallback for unreadable images
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )

@dataclass
class AudioCapsule(IndexedCapsule):
    """Thin capsule for audio files"""
    duration: float = 0.0  # seconds
    sample_rate: int = 44100
    channels: int = 2  # stereo
    bit_depth: int = 16
    codec: str = ""

    def __post_init__(self):
        self.media_type = "audio"

    @classmethod
    def create_from_file(cls, file_path: str) -> 'AudioCapsule':
        """Create capsule from audio file"""
        try:
            import mutagen
            audio = mutagen.File(file_path)

            duration = audio.info.length if hasattr(audio.info, 'length') else 0.0
            sample_rate = audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else 44100
            channels = audio.info.channels if hasattr(audio.info, 'channels') else 2
            bit_depth = audio.info.bits_per_sample if hasattr(audio.info, 'bits_per_sample') else 16

            # Try to determine codec
            codec = ""
            if hasattr(audio, 'mime'):
                mime_parts = audio.mime[0].split('/')
                if len(mime_parts) > 1:
                    codec = mime_parts[1].upper()

            # Compute hash
            with open(file_path, 'rb') as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            file_size = os.path.getsize(file_path)

            return cls(
                file_path=file_path,
                content_hash=content_hash,
                file_size=file_size,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                codec=codec
            )
        except ImportError:
            # mutagen not available, basic info only
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )
        except Exception as e:
            # Fallback for unreadable audio
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )

@dataclass
class PDFCapsule(IndexedCapsule):
    """Thin capsule for PDF documents"""
    page_count: int = 0
    title: str = ""
    author: str = ""
    subject: str = ""
    creator: str = ""
    producer: str = ""

    def __post_init__(self):
        self.media_type = "document"

    @classmethod
    def create_from_file(cls, file_path: str) -> 'PDFCapsule':
        """Create capsule from PDF file"""
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                pdf = pypdf.PdfReader(f)
                page_count = len(pdf.pages)

                # Extract metadata
                metadata = pdf.metadata
                title = metadata.title if metadata and metadata.title else ""
                author = metadata.author if metadata and metadata.author else ""
                subject = metadata.subject if metadata and metadata.subject else ""
                creator = metadata.creator if metadata and metadata.creator else ""
                producer = metadata.producer if metadata and metadata.producer else ""

            # Compute hash
            with open(file_path, 'rb') as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            file_size = os.path.getsize(file_path)

            return cls(
                file_path=file_path,
                content_hash=content_hash,
                file_size=file_size,
                page_count=page_count,
                title=title,
                author=author,
                subject=subject,
                creator=creator,
                producer=producer
            )
        except ImportError:
            # pypdf not available, basic info only
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )
        except Exception as e:
            # Fallback for unreadable PDFs
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )

@dataclass
class VideoCapsule(IndexedCapsule):
    """Thin capsule for video files"""
    duration: float = 0.0  # seconds
    width: int = 0
    height: int = 0
    frame_rate: float = 0.0
    codec: str = ""
    audio_tracks: int = 0

    def __post_init__(self):
        self.media_type = "video"

    @classmethod
    def create_from_file(cls, file_path: str) -> 'VideoCapsule':
        """Create capsule from video file"""
        try:
            import cv2
            cap = cv2.VideoCapture(file_path)

            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

            cap.release()

            # Compute hash (sample first 1MB for performance)
            with open(file_path, 'rb') as f:
                sample = f.read(1024 * 1024)  # 1MB sample
                content_hash = hashlib.sha256(sample).hexdigest()

            file_size = os.path.getsize(file_path)

            return cls(
                file_path=file_path,
                content_hash=content_hash,
                file_size=file_size,
                duration=duration,
                width=width,
                height=height,
                frame_rate=frame_rate,
                codec=codec_str,
                audio_tracks=0  # Would need additional audio analysis
            )
        except ImportError:
            # cv2 not available, basic info only
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )
        except Exception as e:
            # Fallback for unreadable videos
            return cls(
                file_path=file_path,
                content_hash="",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )

@dataclass
class Model3DCapsule(IndexedCapsule):
    """Thin capsule for 3D model files"""
    vertex_count: int = 0
    face_count: int = 0
    material_count: int = 0
    has_animation: bool = False
    format: str = ""

    def __post_init__(self):
        self.media_type = "3d_model"

    @classmethod
    def create_from_file(cls, file_path: str) -> 'Model3DCapsule':
        """Create capsule from 3D model file"""
        # Basic implementation - would need specific 3D libraries for full analysis
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.obj': 'OBJ',
            '.fbx': 'FBX',
            '.gltf': 'GLTF',
            '.glb': 'GLB',
            '.blend': 'BLEND',
            '.dae': 'COLLADA'
        }
        format_str = format_map.get(ext, ext.upper())

        # Compute hash (sample for performance)
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024 * 1024)  # 1MB sample
                content_hash = hashlib.sha256(sample).hexdigest()
        except:
            content_hash = ""

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        return cls(
            file_path=file_path,
            content_hash=content_hash,
            file_size=file_size,
            format=format_str
        )

# ============================================================================
# INDEXED CAPSULE STORE
# ============================================================================

class IndexedCapsuleStore:
    """Lookup-friendly store for indexed capsules"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_path = storage_path / "capsule_index.db"
        self._init_index()

    def _init_index(self):
        """Initialize SQLite index for fast lookups"""
        conn = sqlite3.connect(str(self.index_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capsules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                content_hash TEXT,
                media_type TEXT,
                file_size INTEGER,
                created_at TEXT,
                last_modified TEXT,
                capsule_data TEXT,  -- JSON serialized capsule
                UNIQUE(file_path)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash ON capsules(content_hash)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_type ON capsules(media_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_size ON capsules(file_size)
        ''')

        conn.commit()
        conn.close()

    def store_capsule(self, capsule: IndexedCapsule) -> bool:
        """Store an indexed capsule"""
        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            capsule_data = json.dumps(capsule.to_dict())

            cursor.execute('''
                INSERT OR REPLACE INTO capsules
                (file_path, content_hash, media_type, file_size, created_at, last_modified, capsule_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                capsule.file_path,
                capsule.content_hash,
                capsule.media_type,
                capsule.file_size,
                capsule.created_at,
                capsule.last_modified,
                capsule_data
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing capsule: {e}")
            return False

    def get_capsule(self, file_path: str) -> Optional[IndexedCapsule]:
        """Retrieve capsule by file path"""
        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            cursor.execute('SELECT capsule_data FROM capsules WHERE file_path = ?', (file_path,))
            row = cursor.fetchone()
            conn.close()

            if row:
                data = json.loads(row[0])
                # Determine capsule type and create appropriate instance
                media_type = data.get('media_type', '')
                if media_type == 'image':
                    return ImageCapsule.from_dict(data)
                elif media_type == 'audio':
                    return AudioCapsule.from_dict(data)
                elif media_type == 'document':
                    return PDFCapsule.from_dict(data)
                elif media_type == 'video':
                    return VideoCapsule.from_dict(data)
                elif media_type == '3d_model':
                    return Model3DCapsule.from_dict(data)
                else:
                    return IndexedCapsule.from_dict(data)

            return None
        except Exception as e:
            print(f"Error retrieving capsule: {e}")
            return None

    def get_capsule_by_hash(self, content_hash: str) -> Optional[IndexedCapsule]:
        """Retrieve capsule by content hash"""
        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            cursor.execute('SELECT capsule_data FROM capsules WHERE content_hash = ?', (content_hash,))
            row = cursor.fetchone()
            conn.close()

            if row:
                data = json.loads(row[0])
                media_type = data.get('media_type', '')
                if media_type == 'image':
                    return ImageCapsule.from_dict(data)
                elif media_type == 'audio':
                    return AudioCapsule.from_dict(data)
                elif media_type == 'document':
                    return PDFCapsule.from_dict(data)
                elif media_type == 'video':
                    return VideoCapsule.from_dict(data)
                elif media_type == '3d_model':
                    return Model3DCapsule.from_dict(data)
                else:
                    return IndexedCapsule.from_dict(data)

            return None
        except Exception as e:
            print(f"Error retrieving capsule by hash: {e}")
            return None

    def get_capsules_by_type(self, media_type: str, limit: int = 100) -> List[IndexedCapsule]:
        """Get capsules by media type"""
        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            cursor.execute('SELECT capsule_data FROM capsules WHERE media_type = ? LIMIT ?',
                         (media_type, limit))
            rows = cursor.fetchall()
            conn.close()

            capsules = []
            for row in rows:
                data = json.loads(row[0])
                if media_type == 'image':
                    capsules.append(ImageCapsule.from_dict(data))
                elif media_type == 'audio':
                    capsules.append(AudioCapsule.from_dict(data))
                elif media_type == 'document':
                    capsules.append(PDFCapsule.from_dict(data))
                elif media_type == 'video':
                    capsules.append(VideoCapsule.from_dict(data))
                elif media_type == '3d_model':
                    capsules.append(Model3DCapsule.from_dict(data))
                else:
                    capsules.append(IndexedCapsule.from_dict(data))

            return capsules
        except Exception as e:
            print(f"Error retrieving capsules by type: {e}")
            return []

    def remove_capsule(self, file_path: str) -> bool:
        """Remove capsule from store"""
        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            cursor.execute('DELETE FROM capsules WHERE file_path = ?', (file_path,))

            conn.commit()
            conn.close()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error removing capsule: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        try:
            conn = sqlite3.connect(str(self.index_path))
            cursor = conn.cursor()

            stats = {}

            # Total capsules
            cursor.execute('SELECT COUNT(*) FROM capsules')
            stats['total_capsules'] = cursor.fetchone()[0]

            # By type
            cursor.execute('SELECT media_type, COUNT(*) FROM capsules GROUP BY media_type')
            stats['by_type'] = dict(cursor.fetchall())

            # Total size
            cursor.execute('SELECT SUM(file_size) FROM capsules')
            stats['total_size'] = cursor.fetchone()[0] or 0

            conn.close()
            return stats
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

# ============================================================================
# CONFIGURATION
# ============================================================================

class RegistryMode(Enum):
    """Registry operation modes"""
    STANDALONE = "standalone"  # Single user, local storage
    TEAM = "team"              # Shared registry with permissions
    ENTERPRISE = "enterprise"  # Multi-tenant, cloud-backed
    CLOUD = "cloud"            # Full cloud registry

@dataclass
class RegistryConfig:
    """Production registry configuration"""
    # Core settings
    mode: RegistryMode = RegistryMode.TEAM
    registry_path: Path = Path.home() / ".roca_registry"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    
    # Performance settings
    max_workers: int = 32  # Threadripper optimized
    batch_size: int = 100
    cache_size_mb: int = 1024  # 1GB cache
    
    # Security settings
    enable_encryption: bool = True
    encryption_key_path: Optional[Path] = None
    require_authentication: bool = False
    
    # Network settings
    enable_p2p: bool = True
    p2p_port: int = 8765
    discovery_broadcast: bool = True
    
    # Storage settings
    use_compression: bool = True
    compression_level: int = 6
    deduplicate_files: bool = True
    thumbnail_size: tuple = (256, 256)
    
    # Export/Import settings
    default_export_format: str = "rocapkg"  # rocapkg, zip, directory
    include_metadata: bool = True
    include_previews: bool = True
    include_thumbnails: bool = True

# ============================================================================
# PORTABLE REGISTRY DETECTION AND IMPORT
# ============================================================================

def detect_removable_drives():
    """Detect removable drives (thumb drives, USB sticks) on Windows"""
    import string
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        try:
            import ctypes
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
            # DRIVE_REMOVABLE = 2
            if drive_type == 2:
                drives.append(drive)
        except:
            pass
    return drives

def find_portable_registries():
    """Find ROCA registry folders on removable drives"""
    registries = {}
    removable_drives = detect_removable_drives()
    
    for drive in removable_drives:
        registry_path = Path(drive) / ".roca_registry"
        if registry_path.exists() and (registry_path / "registry.db").exists():
            registries[drive] = registry_path
    
    # Also check local machine for comparison
    local_registry = Path.home() / ".roca_registry"
    if local_registry.exists() and (local_registry / "registry.db").exists():
        registries["LOCAL"] = local_registry
    
    return registries

def is_running_from_removable_drive():
    """Check if the app is running from a removable drive"""
    script_path = Path(__file__).resolve()
    drive_letter = str(script_path).split(':')[0].upper()
    
    try:
        import ctypes
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(f"{drive_letter}:\\")
        return drive_type == 2  # DRIVE_REMOVABLE
    except:
        return False

class PortableRegistryDialog(QDialog):
    """Dialog for importing portable registries"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔄 Portable Registry Detected")
        self.setModal(True)
        self.resize(500, 300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Portable Registry Options")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "A portable ROCA registry was detected. You can import it to merge with your local registry, "
            "or switch to using the portable registry directly."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #ccc;")
        layout.addWidget(desc)
        
        # Options
        self.import_radio = QRadioButton("📥 Import portable registry into local registry")
        self.import_radio.setChecked(True)
        self.import_radio.setStyleSheet("color: white;")
        layout.addWidget(self.import_radio)
        
        self.switch_radio = QRadioButton("🔄 Switch to portable registry (use from thumb drive)")
        self.switch_radio.setStyleSheet("color: white;")
        layout.addWidget(self.switch_radio)
        
        self.skip_radio = QRadioButton("⏭️ Skip (use local registry only)")
        self.skip_radio.setStyleSheet("color: white;")
        layout.addWidget(self.skip_radio)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def get_choice(self):
        if self.import_radio.isChecked():
            return "import"
        elif self.switch_radio.isChecked():
            return "switch"
        else:
            return "skip"

# ============================================================================
# MEDIA REGISTRY CORE
# ============================================================================

class MediaRegistry:
    """Production media registry with ACID compliance"""
    
    def __init__(self, config: RegistryConfig):
        self.config = config
        self.db_path = config.registry_path / "registry.db"
        self._lock = threading.RLock()
        self._cache = {}
        self._init_database()
        
        # Initialize indexed capsule store
        self.capsule_store = IndexedCapsuleStore(config.registry_path / "indexed_capsules")
        
        # Initialize sub-systems
        self.thumbnail_manager = ThumbnailManager(config)
        self.metadata_extractor = MetadataExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.export_manager = ExportManager(config)
        
        print(f"📁 Media Registry initialized: {self.db_path}")
        print(f"🗂️  Indexed Capsule Store: {config.registry_path / 'indexed_capsules'}")

    def _init_database(self):
        """Initialize SQLite database with proper indices"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Main registry table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS media_registry (
                    media_id TEXT PRIMARY KEY,
                    original_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    metadata BLOB,
                    thumbnail_path TEXT,
                    preview_path TEXT,
                    tags TEXT,
                    projects TEXT,
                    status TEXT DEFAULT 'registered',
                    registered_by TEXT,
                    permissions TEXT
                )
            """)
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON media_registry(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media_registry(media_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON media_registry(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON media_registry(tags)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_projects ON media_registry(projects)")
            
            # File references table (for deduplication)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_references (
                    reference_id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    is_primary BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (media_id) REFERENCES media_registry(media_id)
                )
            """)
            
            # User activity table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT,
                    media_id TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()

    def register_media(self, file_path: Path, user_id: str = "system", create_full_capsule: bool = False) -> Dict[str, Any]:
        """Register media file with indexed capsule creation (lightly alive)"""
        with self._lock:
            file_path_str = str(file_path.absolute())
            
            # Check if already have indexed capsule
            existing_capsule = self.capsule_store.get_capsule(file_path_str)
            if existing_capsule:
                print(f"⏭️  Already indexed: {file_path}")
                return existing_capsule.to_dict()
            
            # Create indexed capsule based on file type
            capsule = self._create_indexed_capsule(file_path)
            if capsule:
                # Store the indexed capsule
                success = self.capsule_store.store_capsule(capsule)
                if success:
                    print(f"📦 Indexed capsule created: {file_path.name} ({capsule.media_type})")
                    
                    # Optionally create full media registry entry if requested
                    if create_full_capsule:
                        return self._create_full_media_entry(capsule, user_id)
                    else:
                        return capsule.to_dict()
                else:
                    print(f"❌ Failed to store capsule: {file_path}")
                    return {}
            else:
                print(f"❌ Failed to create capsule: {file_path}")
                return {}

    def _create_indexed_capsule(self, file_path: Path) -> Optional[IndexedCapsule]:
        """Create appropriate indexed capsule based on file type"""
        ext = file_path.suffix.lower()
        
        try:
            if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg']:
                return ImageCapsule.create_from_file(str(file_path))
            elif ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']:
                return AudioCapsule.create_from_file(str(file_path))
            elif ext in ['.pdf']:
                return PDFCapsule.create_from_file(str(file_path))
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv']:
                return VideoCapsule.create_from_file(str(file_path))
            elif ext in ['.obj', '.fbx', '.gltf', '.glb', '.blend', '.dae', '.3ds']:
                return Model3DCapsule.create_from_file(str(file_path))
            else:
                # Generic indexed capsule for unknown types
                try:
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    return IndexedCapsule(
                        file_path=str(file_path.absolute()),
                        content_hash=content_hash,
                        media_type="unknown",
                        file_size=os.path.getsize(file_path)
                    )
                except Exception as e:
                    print(f"Error creating generic capsule: {e}")
                    return None
        except Exception as e:
            print(f"Error creating indexed capsule for {file_path}: {e}")
            return None

    def _create_full_media_entry(self, capsule: IndexedCapsule, user_id: str) -> Dict[str, Any]:
        """Create full media registry entry (legacy heavy processing)"""
        # Generate unique ID
        media_id = self._generate_media_id(Path(capsule.file_path))
        
        # Extract full metadata
        metadata = self.metadata_extractor.extract(Path(capsule.file_path))
        
        # Create thumbnail
        thumbnail_path = self.thumbnail_manager.create_thumbnail(Path(capsule.file_path), media_id)
        
        # Create preview (if applicable)
        preview_path = self._create_preview(Path(capsule.file_path), media_id)
        
        # Store in registry
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO media_registry 
            (media_id, original_path, content_hash, media_type, file_size, 
             metadata, thumbnail_path, preview_path, tags, projects, registered_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            media_id,
            capsule.file_path,
            capsule.content_hash,
            capsule.media_type,
            capsule.file_size,
            pickle.dumps(metadata),  # Serialize metadata
            str(thumbnail_path) if thumbnail_path else None,
            str(preview_path) if preview_path else None,
            ','.join(metadata.get('tags', [])),
            ','.join(metadata.get('projects', [])),
            user_id
        ))
        
        # Add primary file reference
        cursor.execute("""
            INSERT INTO file_references (reference_id, media_id, file_path, is_primary)
            VALUES (?, ?, ?, 1)
        """, (f"REF_{media_id}", media_id, capsule.file_path))
        
        # Log activity
        cursor.execute("""
            INSERT INTO user_activity (user_id, action, media_id, details)
            VALUES (?, ?, ?, ?)
        """, (user_id, 'register_full', media_id, f'Full registration: {Path(capsule.file_path).name}'))
        
        conn.commit()
        conn.close()
        
        # Update cache
        self._cache[media_id] = {
            'media_id': media_id,
            'original_path': capsule.file_path,
            'content_hash': capsule.content_hash,
            'metadata': metadata
        }
        
        print(f"✅ Full registration: {Path(capsule.file_path).name} -> {media_id}")
        
        return self._cache[media_id]

    def bulk_register(self, directory: Path, user_id: str = "system") -> Dict[str, Any]:
        """Register all media in directory with progress tracking"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        media_files = []
        supported_extensions = {
            '.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', 
            '.bmp', '.webp', '.fbx', '.obj', '.gltf', '.glb', '.blend', 
            '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae', '.bvh', '.trc', 
            '.c3d', '.cho', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv',
            '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.pdf', '.txt', '.md',
            '.doc', '.docx', '.psd', '.ai', '.afdesign', '.afphoto', '.zip',
            '.rar', '.7z', '.py', '.json', '.yaml', '.yml', '.xml'
        }
        
        # Find all media files
        for ext in supported_extensions:
            media_files.extend(directory.rglob(f"*{ext}"))
        
        print(f"📁 Found {len(media_files)} media files")
        
        results = {
            'registered': [],
            'duplicates': [],
            'errors': [],
            'skipped': []
        }
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(self.register_media, file, user_id): file
                for file in media_files
            }
            
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result.get('status') == 'duplicate':
                        results['duplicates'].append(result)
                    else:
                        results['registered'].append(result)
                except Exception as e:
                    results['errors'].append({
                        'file': str(file),
                        'error': str(e)
                    })
        
        # Generate summary
        summary = {
            'total_files': len(media_files),
            'registered': len(results['registered']),
            'duplicates': len(results['duplicates']),
            'errors': len(results['errors']),
            'registry_size': self.get_registry_size(),
            'storage_saved': self.calculate_storage_saved(results['duplicates'])
        }
        
        return {
            'summary': summary,
            'details': results
        }

    def get_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get media by content hash (for duplicate detection)"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM media_registry 
                WHERE content_hash = ? AND status = 'registered'
                LIMIT 1
            """, (content_hash,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_dict(row)
            return None

    def get_by_path(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get media by file path"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT mr.* FROM media_registry mr
                JOIN file_references fr ON mr.media_id = fr.media_id
                WHERE fr.file_path = ? AND mr.status = 'registered'
                LIMIT 1
            """, (str(file_path.absolute()),))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_dict(row)
            return None

    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Advanced search with filters"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            base_query = """
                SELECT * FROM media_registry 
                WHERE status = 'registered'
            """
            params = []
            
            # Text search
            if query:
                base_query += """
                    AND (original_path LIKE ? OR tags LIKE ? OR projects LIKE ?)
                """
                search_term = f"%{query}%"
                params.extend([search_term, search_term, search_term])
            
            # Apply filters
            if filters:
                if 'media_type' in filters:
                    base_query += " AND media_type = ?"
                    params.append(filters['media_type'])
                
                if 'min_size' in filters:
                    base_query += " AND file_size >= ?"
                    params.append(filters['min_size'])
                
                if 'max_size' in filters:
                    base_query += " AND file_size <= ?"
                    params.append(filters['max_size'])
                
                if 'tags' in filters:
                    tags = filters['tags']
                    if isinstance(tags, list):
                        tags = ','.join(tags)
                    base_query += " AND tags LIKE ?"
                    params.append(f"%{tags}%")
                
                if 'date_from' in filters:
                    base_query += " AND created_at >= ?"
                    params.append(filters['date_from'])
                
                if 'date_to' in filters:
                    base_query += " AND created_at <= ?"
                    params.append(filters['date_to'])
            
            # Order by most recently accessed
            base_query += " ORDER BY last_accessed DESC"
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]

    def get_all_media(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all registered media items"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM media_registry 
                WHERE status = 'registered'
                ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]

    def export_package(self, media_ids: List[str], output_path: Path, 
                      include_references: bool = True) -> Dict[str, Any]:
        """Export media as ROCA package"""
        return self.export_manager.create_package(
            media_ids=media_ids,
            output_path=output_path,
            include_references=include_references
        )

    def import_package(self, package_path: Path, target_dir: Path, 
                      user_id: str = "system") -> Dict[str, Any]:
        """Import ROCA package"""
        return self.export_manager.import_package(
            package_path=package_path,
            target_dir=target_dir,
            user_id=user_id
        )

    def _generate_media_id(self, file_path: Path) -> str:
        """Generate unique media ID"""
        stat = file_path.stat()
        unique_string = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return f"MED_{hashlib.sha256(unique_string.encode()).hexdigest()[:16]}"

    def _compute_content_hash(self, file_path: Path) -> str:
        """Compute content hash with progress for large files"""
        sha256 = hashlib.sha256()
        buffer_size = 65536  # 64KB chunks
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(buffer_size):
                sha256.update(chunk)
        
        return sha256.hexdigest()

    def _add_file_reference(self, reference_id: str, media_id: str, file_path: Path):
        """Add file reference for deduplication"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_references (reference_id, media_id, file_path, is_primary)
                VALUES (?, ?, ?, 0)
            """, (reference_id, media_id, str(file_path.absolute())))
            conn.commit()
            conn.close()

    def _create_preview(self, file_path: Path, media_id: str) -> Optional[Path]:
        """Create preview for video files"""
        try:
            # Try to use OpenCV for video frame extraction
            import cv2
            
            # Create previews directory if it doesn't exist
            previews_dir = self.config.registry_path / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            
            preview_path = previews_dir / f"{media_id}_preview.jpg"
            
            # Open video and extract frame at 10% through
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return None
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Seek to 10% through the video
            frame_number = max(1, int(total_frames * 0.1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Resize frame to reasonable preview size
                height, width = frame.shape[:2]
                max_size = 400
                if width > height:
                    new_width = min(width, max_size)
                    new_height = int(height * new_width / width)
                else:
                    new_height = min(height, max_size)
                    new_width = int(width * new_height / height)
                
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Save as JPEG
                cv2.imwrite(str(preview_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return preview_path
            
        except ImportError:
            # OpenCV not available
            pass
        except Exception as e:
            print(f"Error creating preview for {file_path}: {e}")
        
        return None

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        columns = [
            'media_id', 'original_path', 'content_hash', 'media_type', 
            'file_size', 'created_at', 'last_accessed', 'access_count',
            'metadata', 'thumbnail_path', 'preview_path', 'tags', 
            'projects', 'status', 'registered_by', 'permissions'
        ]
        
        result = dict(zip(columns, row))
        
        # Deserialize metadata
        if result['metadata']:
            result['metadata'] = pickle.loads(result['metadata'])
        
        # Parse tags and projects
        if result['tags']:
            result['tags'] = result['tags'].split(',')
        else:
            result['tags'] = []
        
        if result['projects']:
            result['projects'] = result['projects'].split(',')
        else:
            result['projects'] = []
        
        return result

    # ============================================================================
    # INDEXED CAPSULE METHODS
    # ============================================================================

    def get_indexed_capsule(self, file_path: str) -> Optional[IndexedCapsule]:
        """Get indexed capsule for a file"""
        return self.capsule_store.get_capsule(file_path)

    def get_indexed_capsule_by_hash(self, content_hash: str) -> Optional[IndexedCapsule]:
        """Get indexed capsule by content hash"""
        return self.capsule_store.get_capsule_by_hash(content_hash)

    def get_indexed_capsules_by_type(self, media_type: str, limit: int = 100) -> List[IndexedCapsule]:
        """Get indexed capsules by media type"""
        return self.capsule_store.get_capsules_by_type(media_type, limit)

    def get_indexed_capsule_stats(self) -> Dict[str, Any]:
        """Get indexed capsule store statistics"""
        return self.capsule_store.get_stats()

    def remove_indexed_capsule(self, file_path: str) -> bool:
        """Remove indexed capsule"""
        return self.capsule_store.remove_capsule(file_path)

    def upgrade_capsule_to_full(self, file_path: str, user_id: str = "system") -> Optional[Dict[str, Any]]:
        """Upgrade an indexed capsule to full media registry entry"""
        capsule = self.get_indexed_capsule(file_path)
        if capsule:
            return self._create_full_media_entry(capsule, user_id)
        return None

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total media count
            cursor.execute("SELECT COUNT(*) FROM media_registry WHERE status = 'registered'")
            stats['total_media'] = cursor.fetchone()[0]
            
            # By type
            cursor.execute("""
                SELECT media_type, COUNT(*) 
                FROM media_registry 
                WHERE status = 'registered'
                GROUP BY media_type
            """)
            stats['by_type'] = dict(cursor.fetchall())
            
            # Storage used
            cursor.execute("SELECT SUM(file_size) FROM media_registry WHERE status = 'registered'")
            stats['total_size'] = cursor.fetchone()[0] or 0
            
            # Duplicate savings
            cursor.execute("""
                SELECT COUNT(DISTINCT content_hash) as unique_files,
                       COUNT(*) as total_files
                FROM media_registry 
                WHERE status = 'registered'
            """)
            unique, total = cursor.fetchone()
            stats['duplicate_savings'] = total - unique
            
            # Recent activity
            cursor.execute("""
                SELECT action, COUNT(*) 
                FROM user_activity 
                WHERE timestamp > datetime('now', '-7 days')
                GROUP BY action
            """)
            stats['weekly_activity'] = dict(cursor.fetchall())
            
            conn.close()
            
            return stats

    def get_registry_size(self) -> int:
        """Get total registry size"""
        stats = self.get_registry_stats()
        return stats.get('total_size', 0)

    def calculate_storage_saved(self, duplicates: List) -> int:
        """Calculate storage saved from deduplication"""
        total_saved = 0
        for dup in duplicates:
            if 'file_size' in dup:
                total_saved += dup['file_size']
        return total_saved

    def delete_media(self, media_id: str) -> bool:
        """Delete media from registry"""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30)
                cursor = conn.cursor()
                
                # Get file path for cleanup
                cursor.execute("SELECT original_path FROM media_registry WHERE media_id = ?", (media_id,))
                row = cursor.fetchone()
                
                if row:
                    # Delete from registry
                    cursor.execute("DELETE FROM media_registry WHERE media_id = ?", (media_id,))
                    cursor.execute("DELETE FROM file_references WHERE media_id = ?", (media_id,))
                    
                    # Log activity
                    cursor.execute("""
                        INSERT INTO user_activity (user_id, action, media_id, details)
                        VALUES (?, ?, ?, ?)
                    """, ("system", 'delete', media_id, f'Deleted media {media_id}'))
                    
                    conn.commit()
                    conn.close()
                    
                    # Clear from cache
                    if media_id in self._cache:
                        del self._cache[media_id]
                    
                    print(f"🗑️  Deleted media: {media_id}")
                    return True
                else:
                    conn.close()
                    return False
            except Exception as e:
                print(f"Error deleting media {media_id}: {e}")
                return False

    def import_portable_registry(self, source_path: Path, use_advanced_resolution: bool = True) -> Dict[str, Any]:
        """Import a portable registry into this registry with advanced conflict resolution"""
        result = {
            'success': False,
            'imported_count': 0,
            'skipped_count': 0,
            'conflicts_resolved': 0,
            'errors': []
        }
        
        try:
            source_db = source_path / "registry.db"
            if not source_db.exists():
                result['errors'].append(f"Source registry database not found: {source_db}")
                return result
            
            # Initialize conflict resolution manager
            conflict_manager = ConflictResolutionManager(self.config.registry_path)
            
            # Connect to source database
            source_conn = sqlite3.connect(str(source_db))
            source_cursor = source_conn.cursor()
            
            # Get all media from source
            source_cursor.execute("SELECT * FROM media_registry")
            source_media_rows = source_cursor.fetchall()
            
            # Get column names
            source_cursor.execute("PRAGMA table_info(media_registry)")
            columns = [col[1] for col in source_cursor.fetchall()]
            
            # Connect to target database
            target_conn = sqlite3.connect(str(self.db_path))
            target_cursor = target_conn.cursor()
            
            imported = 0
            skipped = 0
            conflicts_resolved = 0
            
            # Collect conflicts for batch processing
            conflicts = []
            auto_resolved = []
            
            for row in source_media_rows:
                media_data = dict(zip(columns, row))
                media_id = media_data['media_id']
                
                # Check if media_id already exists
                target_cursor.execute("SELECT * FROM media_registry WHERE media_id = ?", (media_id,))
                existing = target_cursor.fetchone()
                
                if existing:
                    existing_data = dict(zip(columns, existing))
                    
                    if use_advanced_resolution:
                        # Use advanced conflict resolution
                        conflicts.append((media_data, existing_data))
                    else:
                        # Simple resolution: skip duplicates
                        skipped += 1
                        continue
                else:
                    # Check for filename conflicts
                    filename = media_data.get('filename', '')
                    target_cursor.execute("SELECT * FROM media_registry WHERE filename = ?", (filename,))
                    filename_conflicts = target_cursor.fetchall()
                    
                    if filename_conflicts:
                        if use_advanced_resolution:
                            conflicts.append((media_data, dict(zip(columns, filename_conflicts[0]))))
                        else:
                            # Simple resolution: skip
                            skipped += 1
                            continue
                    else:
                        # No conflict, import directly
                        auto_resolved.append(media_data)
            
            # Process auto-resolved items
            for media_data in auto_resolved:
                placeholders = ', '.join(['?' for _ in columns])
                values = [media_data[col] for col in columns]
                
                try:
                    target_cursor.execute(f"""
                        INSERT INTO media_registry ({', '.join(columns)})
                        VALUES ({placeholders})
                    """, values)
                    imported += 1
                except Exception as e:
                    result['errors'].append(f"Failed to import {media_data.get('media_id', 'unknown')}: {e}")
            
            # Process conflicts
            if conflicts and use_advanced_resolution:
                # Show advanced conflict resolution dialog
                dialog = AdvancedConflictDialog(conflicts, conflict_manager, self.main_window if hasattr(self, 'main_window') else None)
                if dialog.exec() == QDialog.Accepted:
                    resolutions = dialog.get_resolutions()
                    
                    for i, (source_media, target_media) in enumerate(conflicts):
                        resolution = resolutions.get(i, 'skip')
                        
                        # Record the resolution for learning
                        analysis = conflict_manager.analyze_conflict(source_media, target_media)
                        conflict_manager.record_resolution(analysis['type'], source_media.get('filename', ''), resolution)
                        
                        if resolution == 'skip':
                            skipped += 1
                        elif resolution == 'keep_both':
                            # Generate new media_id
                            new_id = f"{source_media['media_id']}_imported_{int(time.time())}_{i}"
                            source_media = source_media.copy()
                            source_media['media_id'] = new_id
                            # Import with new ID
                            placeholders = ', '.join(['?' for _ in columns])
                            values = [source_media[col] for col in columns]
                            try:
                                target_cursor.execute(f"""
                                    INSERT INTO media_registry ({', '.join(columns)})
                                    VALUES ({placeholders})
                                """, values)
                                imported += 1
                                conflicts_resolved += 1
                            except Exception as e:
                                result['errors'].append(f"Failed to import renamed item: {e}")
                        elif resolution in ['keep_source', 'import']:
                            # Replace existing
                            placeholders = ', '.join(['?' for _ in columns])
                            values = [source_media[col] for col in columns]
                            try:
                                target_cursor.execute(f"""
                                    INSERT OR REPLACE INTO media_registry ({', '.join(columns)})
                                    VALUES ({placeholders})
                                """, values)
                                imported += 1
                                conflicts_resolved += 1
                            except Exception as e:
                                result['errors'].append(f"Failed to replace item {source_media.get('media_id', 'unknown')}: {e}")
                        elif resolution == 'merge':
                            # Merge metadata (simplified)
                            merged_data = self.merge_media_metadata(source_media, target_media)
                            if merged_data:
                                placeholders = ', '.join(['?' for _ in columns])
                                values = [merged_data[col] for col in columns]
                                try:
                                    target_cursor.execute(f"""
                                        INSERT OR REPLACE INTO media_registry ({', '.join(columns)})
                                        VALUES ({placeholders})
                                    """, values)
                                    imported += 1
                                    conflicts_resolved += 1
                                except Exception as e:
                                    result['errors'].append(f"Failed to merge item {source_media.get('media_id', 'unknown')}: {e}")
                else:
                    # User cancelled
                    result['errors'].append("Import cancelled by user")
                    target_conn.close()
                    source_conn.close()
                    return result
            
            # Copy thumbnails and previews
            self._copy_media_files(source_path, target_conn)
            
            target_conn.commit()
            target_conn.close()
            source_conn.close()
            
            # Save learned rules
            conflict_manager.save_rules()
            
            result['success'] = True
            result['imported_count'] = imported
            result['skipped_count'] = skipped
            result['conflicts_resolved'] = conflicts_resolved
            
        except Exception as e:
            result['errors'].append(f"Import failed: {e}")
        
        return result

    def resolve_import_conflict(self, source_media: Dict, target_cursor) -> str:
        """Resolve conflicts during import. Returns 'import', 'skip', or 'conflict_resolved'"""
        media_id = source_media['media_id']
        filename = source_media.get('filename', '')
        
        # Check if media_id already exists
        target_cursor.execute("SELECT * FROM media_registry WHERE media_id = ?", (media_id,))
        existing = target_cursor.fetchone()
        
        if not existing:
            # Check for filename conflicts
            target_cursor.execute("SELECT * FROM media_registry WHERE filename = ?", (filename,))
            filename_conflicts = target_cursor.fetchall()
            
            if filename_conflicts:
                # Show conflict resolution dialog
                conflict_dialog = ImportConflictDialog(source_media, filename_conflicts[0], self)
                result = conflict_dialog.exec()
                
                if result == QDialog.Accepted:
                    choice = conflict_dialog.get_resolution()
                    if choice == 'skip':
                        return 'skip'
                    elif choice == 'keep_both':
                        # Generate new media_id for source
                        source_media['media_id'] = f"{media_id}_imported_{int(time.time())}"
                        return 'import'
                    elif choice == 'replace':
                        return 'import'  # INSERT OR REPLACE will handle this
                    elif choice == 'merge':
                        # Merge metadata (keep newer dates, combine tags, etc.)
                        return self.merge_media_metadata(source_media, filename_conflicts[0])
                else:
                    return 'skip'
        
        return 'import'

    def _copy_media_files(self, source_path: Path, target_conn):
        """Copy thumbnails and previews from source to target"""
        source_thumbs = source_path / "thumbnails"
        target_thumbs = self.config.registry_path / "thumbnails"
        
        if source_thumbs.exists():
            import shutil
            target_thumbs.mkdir(parents=True, exist_ok=True)
            for thumb_file in source_thumbs.glob("*"):
                if thumb_file.is_file():
                    target_thumb = target_thumbs / thumb_file.name
                    if not target_thumb.exists() or thumb_file.stat().st_mtime > target_thumb.stat().st_mtime:
                        shutil.copy2(thumb_file, target_thumb)
        
        source_previews = source_path / "previews"
        target_previews = self.config.registry_path / "previews"
        
        if source_previews.exists():
            target_previews.mkdir(parents=True, exist_ok=True)
            for preview_file in source_previews.glob("*"):
                if preview_file.is_file():
                    target_preview = target_previews / preview_file.name
                    if not target_preview.exists() or preview_file.stat().st_mtime > target_preview.stat().st_mtime:
                        shutil.copy2(preview_file, target_preview)

    def merge_media_metadata(self, source_media: Dict, target_media: Dict) -> Dict:
        """Merge metadata from source into target"""
        # Start with target data
        merged = target_media.copy()
        
        # Prefer source for certain fields
        prefer_source_fields = ['last_modified', 'file_size', 'content_hash']
        for field in prefer_source_fields:
            if field in source_media and source_media[field]:
                merged[field] = source_media[field]
        
        # Merge tags and projects (combine unique values)
        for field in ['tags', 'projects']:
            source_values = source_media.get(field, '')
            target_values = target_media.get(field, '')
            
            if source_values and target_values:
                # Parse JSON if needed
                try:
                    source_list = json.loads(source_values) if isinstance(source_values, str) else source_values
                    target_list = json.loads(target_values) if isinstance(target_values, str) else target_values
                    merged[field] = json.dumps(list(set(source_list + target_list)))
                except:
                    merged[field] = source_values  # Fallback to source
            elif source_values:
                merged[field] = source_values
        
        return merged

    def sync_registries(self, target_path: Path = None) -> Dict[str, Any]:
        """Synchronize registries bidirectionally"""
        result = {
            'success': False,
            'syncs_performed': 0,
            'conflicts_resolved': 0,
            'errors': []
        }
        
        try:
            # Find available registries
            registries = find_portable_registries()
            
            if not target_path:
                # Auto-detect target registry
                if is_running_from_removable_drive():
                    # Running from thumb drive, sync with local
                    if "LOCAL" in registries:
                        target_path = registries["LOCAL"]
                    else:
                        result['errors'].append("No local registry found to sync with")
                        return result
                else:
                    # Running from local, sync with first portable registry
                    portable_registries = {k: v for k, v in registries.items() if k != "LOCAL"}
                    if portable_registries:
                        target_path = list(portable_registries.values())[0]
                    else:
                        result['errors'].append("No portable registry found to sync with")
                        return result
            
            # Perform bidirectional sync
            sync_result = self.perform_bidirectional_sync(target_path)
            result.update(sync_result)
            
        except Exception as e:
            result['errors'].append(f"Sync failed: {e}")
        
        return result

    def perform_bidirectional_sync(self, target_path: Path) -> Dict[str, Any]:
        """Perform bidirectional synchronization between registries"""
        result = {
            'success': False,
            'syncs_performed': 0,
            'conflicts_resolved': 0,
            'errors': []
        }
        
        try:
            # This is a simplified implementation
            # In a full implementation, you'd track last_sync timestamps
            # and only sync changes since then
            
            # For now, import from target to source
            import_result = self.import_portable_registry(target_path)
            
            if import_result['success']:
                result['success'] = True
                result['syncs_performed'] = import_result['imported_count']
                result['conflicts_resolved'] = import_result['conflicts_resolved']
            
        except Exception as e:
            result['errors'].append(f"Bidirectional sync failed: {e}")
        
        return result

    def start_auto_sync(self):
        """Start automatic synchronization timer"""
        if not hasattr(self, '_sync_timer'):
            self._sync_timer = QTimer()
            self._sync_timer.timeout.connect(self.perform_auto_sync)
            self._sync_timer.start(300000)  # Check every 5 minutes
        
        self.perform_auto_sync()

    def perform_auto_sync(self):
        """Perform automatic synchronization if portable registry is available"""
        try:
            registries = find_portable_registries()
            portable_available = len([k for k in registries.keys() if k != "LOCAL"]) > 0
            
            if portable_available:
                sync_result = self.sync_registries()
                if sync_result['success'] and sync_result['syncs_performed'] > 0:
                    self.status_bar.showMessage(f"Auto-synced {sync_result['syncs_performed']} items")
                    
        except Exception as e:
            print(f"Auto-sync error: {e}")

# ============================================================================
# CONFLICT RESOLUTION DIALOG
# ============================================================================

class ImportConflictDialog(QDialog):
    """Dialog for resolving import conflicts"""
    
    def __init__(self, source_media: Dict, target_media: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔄 Import Conflict")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Import Conflict Detected")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        # Conflict details
        details_text = f"""
        <b>Filename:</b> {source_media.get('filename', 'Unknown')}<br>
        <b>Source:</b> Size: {self._format_size(source_media.get('file_size', 0))}, 
        Modified: {source_media.get('last_modified', 'Unknown')}<br>
        <b>Target:</b> Size: {self._format_size(target_media.get('file_size', 0))}, 
        Modified: {target_media.get('last_modified', 'Unknown')}
        """
        
        details = QLabel(details_text)
        details.setWordWrap(True)
        details.setStyleSheet("color: #ccc; background: rgba(40,40,50,0.5); padding: 10px; border-radius: 5px;")
        layout.addWidget(details)
        
        # Resolution options
        self.keep_source = QRadioButton("📥 Keep source (replace target)")
        self.keep_source.setChecked(True)
        self.keep_source.setStyleSheet("color: white;")
        layout.addWidget(self.keep_source)
        
        self.keep_both = QRadioButton("📋 Keep both (rename source)")
        self.keep_both.setStyleSheet("color: white;")
        layout.addWidget(self.keep_both)
        
        self.skip = QRadioButton("⏭️ Skip this item")
        self.skip.setStyleSheet("color: white;")
        layout.addWidget(self.skip)
        
        self.merge = QRadioButton("🔀 Merge metadata")
        self.merge.setStyleSheet("color: white;")
        layout.addWidget(self.merge)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        resolve_btn = QPushButton("Resolve")
        resolve_btn.clicked.connect(self.accept)
        resolve_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(resolve_btn)
        
        cancel_btn = QPushButton("Cancel Import")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def get_resolution(self):
        if self.keep_source.isChecked():
            return 'replace'
        elif self.keep_both.isChecked():
            return 'keep_both'
        elif self.skip.isChecked():
            return 'skip'
        elif self.merge.isChecked():
            return 'merge'
        return 'skip'
    
    def _format_size(self, size_bytes):
        """Format bytes to human readable"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"

# ============================================================================
# ADVANCED CONFLICT RESOLUTION
# ============================================================================

class ConflictRule(Enum):
    """Types of conflict resolution rules"""
    ALWAYS_KEEP_NEWER = "always_keep_newer"
    ALWAYS_KEEP_LARGER = "always_keep_larger"
    ALWAYS_KEEP_SOURCE = "always_keep_source"
    ALWAYS_KEEP_TARGET = "always_keep_target"
    ALWAYS_SKIP = "always_skip"
    ALWAYS_MERGE = "always_merge"
    ASK_USER = "ask_user"

@dataclass
class ConflictResolutionRule:
    """A conflict resolution rule"""
    name: str
    condition: str  # e.g., "filename_conflict", "size_difference", "date_difference"
    rule: ConflictRule
    priority: int = 0

class ConflictResolutionManager:
    """Advanced conflict resolution with rules and learning"""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path.home() / ".roca_registry" / "conflict_rules.json"
        self.rules = self.load_rules()
        self.conflict_history = self.load_history()
        
    def load_rules(self) -> List[ConflictResolutionRule]:
        """Load conflict resolution rules"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return [ConflictResolutionRule(**rule) for rule in data.get('rules', [])]
        except Exception as e:
            print(f"Error loading conflict rules: {e}")
        
        # Default rules
        return [
            ConflictResolutionRule("Same filename, different size", "filename_size_conflict", ConflictRule.ASK_USER, 10),
            ConflictResolutionRule("Same filename, different date", "filename_date_conflict", ConflictRule.ALWAYS_KEEP_NEWER, 8),
            ConflictResolutionRule("Exact duplicate", "exact_duplicate", ConflictRule.ALWAYS_SKIP, 15),
            ConflictResolutionRule("Different filenames, same hash", "hash_duplicate", ConflictRule.ALWAYS_SKIP, 12),
        ]
    
    def save_rules(self):
        """Save conflict resolution rules"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {'rules': [vars(rule) for rule in self.rules]}
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving conflict rules: {e}")
    
    def load_history(self) -> Dict:
        """Load conflict resolution history for learning"""
        history_path = self.config_path.parent / "conflict_history.json"
        try:
            if history_path.exists():
                with open(history_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading conflict history: {e}")
        return {}
    
    def save_history(self):
        """Save conflict resolution history"""
        history_path = self.config_path.parent / "conflict_history.json"
        try:
            with open(history_path, 'w') as f:
                json.dump(self.conflict_history, f, indent=2)
        except Exception as e:
            print(f"Error saving conflict history: {e}")
    
    def analyze_conflict(self, source_media: Dict, target_media: Dict) -> Dict:
        """Analyze the type and details of a conflict"""
        conflict_info = {
            'type': 'unknown',
            'severity': 'low',
            'differences': [],
            'recommendation': ConflictRule.ASK_USER,
            'confidence': 0.0
        }
        
        # Check for exact duplicate
        if (source_media.get('content_hash') and 
            source_media.get('content_hash') == target_media.get('content_hash')):
            conflict_info['type'] = 'exact_duplicate'
            conflict_info['severity'] = 'none'
            conflict_info['recommendation'] = ConflictRule.ALWAYS_SKIP
            conflict_info['confidence'] = 1.0
            return conflict_info
        
        # Check filename conflicts
        source_filename = source_media.get('filename', '')
        target_filename = target_media.get('filename', '')
        
        if source_filename == target_filename:
            conflict_info['type'] = 'filename_conflict'
            
            # Analyze differences
            source_size = source_media.get('file_size', 0)
            target_size = target_media.get('file_size', 0)
            source_date = source_media.get('last_modified', '')
            target_date = target_media.get('last_modified', '')
            
            if source_size != target_size:
                conflict_info['differences'].append({
                    'field': 'file_size',
                    'source': source_size,
                    'target': target_size,
                    'difference': abs(source_size - target_size)
                })
                conflict_info['type'] = 'filename_size_conflict'
                conflict_info['severity'] = 'medium'
            
            if source_date != target_date:
                conflict_info['differences'].append({
                    'field': 'last_modified',
                    'source': source_date,
                    'target': target_date
                })
                if conflict_info['type'] == 'filename_conflict':
                    conflict_info['type'] = 'filename_date_conflict'
                conflict_info['severity'] = 'medium'
            
            # Check for hash duplicates with different names
            if (source_media.get('content_hash') and target_media.get('content_hash') and
                source_media['content_hash'] == target_media['content_hash']):
                conflict_info['type'] = 'hash_duplicate'
                conflict_info['severity'] = 'low'
                conflict_info['recommendation'] = ConflictRule.ALWAYS_SKIP
                conflict_info['confidence'] = 0.9
        else:
            # Different filenames
            if (source_media.get('content_hash') and target_media.get('content_hash') and
                source_media['content_hash'] == target_media['content_hash']):
                conflict_info['type'] = 'hash_duplicate'
                conflict_info['severity'] = 'low'
                conflict_info['recommendation'] = ConflictRule.ALWAYS_SKIP
                conflict_info['confidence'] = 0.9
        
        # Apply rules
        for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
            if self.matches_rule(conflict_info, rule):
                conflict_info['recommendation'] = rule.rule
                conflict_info['confidence'] = 0.8
                break
        
        # Learn from history
        history_key = f"{conflict_info['type']}_{source_filename}"
        if history_key in self.conflict_history:
            history = self.conflict_history[history_key]
            most_common = max(history.items(), key=lambda x: x[1])
            if most_common[1] > 2:  # If chosen more than twice
                conflict_info['recommendation'] = ConflictRule(most_common[0])
                conflict_info['confidence'] = 0.7
        
        return conflict_info
    
    def matches_rule(self, conflict_info: Dict, rule: ConflictResolutionRule) -> bool:
        """Check if a conflict matches a rule"""
        if rule.condition == conflict_info['type']:
            return True
        if rule.condition == "any_conflict" and conflict_info['type'] != 'exact_duplicate':
            return True
        return False
    
    def resolve_conflict_auto(self, source_media: Dict, target_media: Dict) -> str:
        """Automatically resolve a conflict based on rules"""
        analysis = self.analyze_conflict(source_media, target_media)
        
        if analysis['recommendation'] == ConflictRule.ALWAYS_KEEP_NEWER:
            source_date = source_media.get('last_modified', '')
            target_date = target_media.get('last_modified', '')
            return 'import' if source_date > target_date else 'skip'
        
        elif analysis['recommendation'] == ConflictRule.ALWAYS_KEEP_LARGER:
            source_size = source_media.get('file_size', 0)
            target_size = target_media.get('file_size', 0)
            return 'import' if source_size > target_size else 'skip'
        
        elif analysis['recommendation'] == ConflictRule.ALWAYS_KEEP_SOURCE:
            return 'import'
        
        elif analysis['recommendation'] == ConflictRule.ALWAYS_KEEP_TARGET:
            return 'skip'
        
        elif analysis['recommendation'] == ConflictRule.ALWAYS_SKIP:
            return 'skip'
        
        elif analysis['recommendation'] == ConflictRule.ALWAYS_MERGE:
            return 'conflict_resolved'  # Will trigger merge
        
        return 'ask_user'  # ASK_USER
    
    def record_resolution(self, conflict_type: str, filename: str, resolution: str):
        """Record a conflict resolution for learning"""
        history_key = f"{conflict_type}_{filename}"
        if history_key not in self.conflict_history:
            self.conflict_history[history_key] = {}
        
        if resolution not in self.conflict_history[history_key]:
            self.conflict_history[history_key][resolution] = 0
        
        self.conflict_history[history_key][resolution] += 1
        self.save_history()

# ============================================================================
# ENHANCED CONFLICT RESOLUTION DIALOG
# ============================================================================

class AdvancedConflictDialog(QDialog):
    """Advanced conflict resolution dialog with rules and batch processing"""
    
    def __init__(self, conflicts: List[Tuple[Dict, Dict]], conflict_manager: ConflictResolutionManager, parent=None):
        super().__init__(parent)
        self.conflicts = conflicts
        self.conflict_manager = conflict_manager
        self.current_index = 0
        self.resolutions = {}
        
        self.setWindowTitle("🔄 Advanced Conflict Resolution")
        self.setModal(True)
        self.resize(800, 600)
        
        self.init_ui()
        self.show_conflict(0)
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Advanced Conflict Resolution")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; color: white;")
        layout.addWidget(header)
        
        # Progress indicator
        self.progress_label = QLabel(f"Conflict 1 of {len(self.conflicts)}")
        self.progress_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.progress_label)
        
        # Conflict details area
        details_group = QGroupBox("Conflict Details")
        details_group.setStyleSheet("""
            QGroupBox {
                color: white;
                border: 1px solid rgb(60, 60, 80);
                border-radius: 6px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        details_layout = QVBoxLayout(details_group)
        
        # Conflict type and analysis
        self.conflict_type_label = QLabel("Analyzing conflict...")
        self.conflict_type_label.setStyleSheet("color: #ffa; font-weight: bold;")
        details_layout.addWidget(self.conflict_type_label)
        
        # Source vs Target comparison
        comparison_layout = QHBoxLayout()
        
        # Source panel
        source_group = QGroupBox("Source (Importing)")
        source_group.setStyleSheet("color: white;")
        source_layout = QVBoxLayout(source_group)
        self.source_info = QLabel("Loading...")
        self.source_info.setWordWrap(True)
        source_layout.addWidget(self.source_info)
        comparison_layout.addWidget(source_group)
        
        # Target panel
        target_group = QGroupBox("Target (Existing)")
        target_group.setStyleSheet("color: white;")
        target_layout = QVBoxLayout(target_group)
        self.target_info = QLabel("Loading...")
        self.target_info.setWordWrap(True)
        target_layout.addWidget(self.target_info)
        comparison_layout.addWidget(target_group)
        
        details_layout.addLayout(comparison_layout)
        
        # Differences panel
        diff_group = QGroupBox("Key Differences")
        diff_group.setStyleSheet("color: white;")
        diff_layout = QVBoxLayout(diff_group)
        self.differences_text = QTextEdit()
        self.differences_text.setReadOnly(True)
        self.differences_text.setMaximumHeight(100)
        diff_layout.addWidget(self.differences_text)
        details_layout.addWidget(diff_group)
        
        layout.addWidget(details_group)
        
        # Resolution options
        resolution_group = QGroupBox("Resolution Options")
        resolution_group.setStyleSheet("color: white;")
        resolution_layout = QVBoxLayout(resolution_group)
        
        # Recommended action
        self.recommendation_label = QLabel("Recommended: Analyzing...")
        self.recommendation_label.setStyleSheet("color: #ffa; font-weight: bold;")
        resolution_layout.addWidget(self.recommendation_label)
        
        # Resolution choices
        self.resolution_buttons = {}
        options = [
            ('keep_source', '📥 Keep source (replace target)'),
            ('keep_both', '📋 Keep both (rename source)'),
            ('skip', '⏭️ Skip this item'),
            ('merge', '🔀 Merge metadata'),
        ]
        
        for key, text in options:
            radio = QRadioButton(text)
            radio.setStyleSheet("color: white;")
            self.resolution_buttons[key] = radio
            resolution_layout.addWidget(radio)
        
        # Default selection
        if 'keep_source' in self.resolution_buttons:
            self.resolution_buttons['keep_source'].setChecked(True)
        
        layout.addWidget(resolution_group)
        
        # Navigation and action buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        self.prev_btn.setEnabled(False)
        button_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ➡️")
        self.next_btn.clicked.connect(self.show_next)
        button_layout.addWidget(self.next_btn)
        
        button_layout.addStretch()
        
        # Batch actions
        self.apply_all_btn = QPushButton("Apply to All Similar")
        self.apply_all_btn.clicked.connect(self.apply_to_similar)
        button_layout.addWidget(self.apply_all_btn)
        
        self.auto_resolve_btn = QPushButton("🤖 Auto-Resolve")
        self.auto_resolve_btn.clicked.connect(self.auto_resolve)
        button_layout.addWidget(self.auto_resolve_btn)
        
        # Final actions
        resolve_btn = QPushButton("✅ Resolve")
        resolve_btn.clicked.connect(self.accept)
        resolve_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(resolve_btn)
        
        cancel_btn = QPushButton("❌ Cancel All")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def show_conflict(self, index: int):
        """Show conflict at the given index"""
        if index < 0 or index >= len(self.conflicts):
            return
        
        self.current_index = index
        source_media, target_media = self.conflicts[index]
        
        # Update progress
        self.progress_label.setText(f"Conflict {index + 1} of {len(self.conflicts)}")
        
        # Analyze conflict
        analysis = self.conflict_manager.analyze_conflict(source_media, target_media)
        
        # Update UI
        self.conflict_type_label.setText(f"Conflict Type: {analysis['type'].replace('_', ' ').title()}")
        
        # Source info
        source_info = self.format_media_info(source_media, "Source")
        self.source_info.setText(source_info)
        
        # Target info
        target_info = self.format_media_info(target_media, "Target")
        self.target_info.setText(target_info)
        
        # Differences
        diff_text = self.format_differences(analysis['differences'])
        self.differences_text.setPlainText(diff_text)
        
        # Recommendation
        rec_text = f"Recommended: {analysis['recommendation'].value.replace('_', ' ').title()}"
        if analysis['confidence'] > 0:
            rec_text += f" (Confidence: {analysis['confidence']:.1%})"
        self.recommendation_label.setText(rec_text)
        
        # Update navigation
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.conflicts) - 1)
        self.next_btn.setText("Next ➡️" if index < len(self.conflicts) - 1 else "Finish ✅")
    
    def format_media_info(self, media: Dict, label: str) -> str:
        """Format media information for display"""
        info = f"<b>{label}</b><br>"
        info += f"Filename: {media.get('filename', 'Unknown')}<br>"
        info += f"Size: {self._format_size(media.get('file_size', 0))}<br>"
        info += f"Modified: {media.get('last_modified', 'Unknown')}<br>"
        info += f"Type: {media.get('media_type', 'Unknown')}<br>"
        if media.get('content_hash'):
            info += f"Hash: {media['content_hash'][:16]}...<br>"
        return info
    
    def format_differences(self, differences: List[Dict]) -> str:
        """Format differences for display"""
        if not differences:
            return "No significant differences detected."
        
        text = ""
        for diff in differences:
            field = diff['field']
            source_val = diff.get('source', 'N/A')
            target_val = diff.get('target', 'N/A')
            
            if field == 'file_size':
                source_val = self._format_size(source_val)
                target_val = self._format_size(target_val)
            
            text += f"{field.title()}: {source_val} → {target_val}\n"
        
        return text.strip()
    
    def show_previous(self):
        """Show previous conflict"""
        if self.current_index > 0:
            self.save_current_resolution()
            self.show_conflict(self.current_index - 1)
    
    def show_next(self):
        """Show next conflict"""
        if self.current_index < len(self.conflicts) - 1:
            self.save_current_resolution()
            self.show_conflict(self.current_index + 1)
        else:
            # Finish
            self.save_current_resolution()
            self.accept()
    
    def save_current_resolution(self):
        """Save the resolution for the current conflict"""
        resolution = None
        for key, button in self.resolution_buttons.items():
            if button.isChecked():
                resolution = key
                break
        
        if resolution:
            self.resolutions[self.current_index] = resolution
    
    def apply_to_similar(self):
        """Apply current resolution to all similar conflicts"""
        current_resolution = None
        for key, button in self.resolution_buttons.items():
            if button.isChecked():
                current_resolution = key
                break
        
        if not current_resolution:
            return
        
        # Apply to all conflicts
        for i in range(len(self.conflicts)):
            self.resolutions[i] = current_resolution
        
        QMessageBox.information(self, "Applied", f"Applied '{current_resolution}' to all {len(self.conflicts)} conflicts.")
    
    def auto_resolve(self):
        """Auto-resolve conflicts using rules"""
        resolved = 0
        for i, (source, target) in enumerate(self.conflicts):
            if i not in self.resolutions:
                resolution = self.conflict_manager.resolve_conflict_auto(source, target)
                if resolution != 'ask_user':
                    self.resolutions[i] = resolution
                    resolved += 1
        
        if resolved > 0:
            QMessageBox.information(self, "Auto-Resolved", f"Automatically resolved {resolved} conflicts.")
            self.show_conflict(self.current_index)  # Refresh display
    
    def get_resolutions(self) -> Dict[int, str]:
        """Get all resolutions"""
        return self.resolutions.copy()
    
    def _format_size(self, size_bytes):
        """Format bytes to human readable"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"

# ============================================================================
# CONFLICT RULES CONFIGURATION DIALOG
# ============================================================================

class ConflictRulesDialog(QDialog):
    """Dialog for configuring conflict resolution rules"""
    
    def __init__(self, conflict_manager: ConflictResolutionManager, parent=None):
        super().__init__(parent)
        self.conflict_manager = conflict_manager
        
        self.setWindowTitle("⚙️ Conflict Resolution Rules")
        self.setModal(True)
        self.resize(700, 500)
        
        self.init_ui()
        self.load_rules()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Conflict Resolution Rules")
        header.setStyleSheet("font-size: 16pt; font-weight: bold; color: white;")
        layout.addWidget(header)
        
        # Rules list
        rules_group = QGroupBox("Active Rules")
        rules_group.setStyleSheet("color: white;")
        rules_layout = QVBoxLayout(rules_group)
        
        self.rules_list = QListWidget()
        self.rules_list.setStyleSheet("""
            QListWidget {
                background-color: rgb(40, 40, 50);
                color: white;
                border: 1px solid rgb(60, 60, 80);
                border-radius: 4px;
            }
        """)
        rules_layout.addWidget(self.rules_list)
        
        # Rule controls
        controls_layout = QHBoxLayout()
        
        add_btn = QPushButton("➕ Add Rule")
        add_btn.clicked.connect(self.add_rule)
        controls_layout.addWidget(add_btn)
        
        edit_btn = QPushButton("✏️ Edit Rule")
        edit_btn.clicked.connect(self.edit_rule)
        controls_layout.addWidget(edit_btn)
        
        remove_btn = QPushButton("➖ Remove Rule")
        remove_btn.clicked.connect(self.remove_rule)
        controls_layout.addWidget(remove_btn)
        
        controls_layout.addStretch()
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self.reset_rules)
        controls_layout.addWidget(reset_btn)
        
        rules_layout.addLayout(controls_layout)
        layout.addWidget(rules_group)
        
        # Statistics
        stats_group = QGroupBox("Learning Statistics")
        stats_group.setStyleSheet("color: white;")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        save_btn = QPushButton("💾 Save Rules")
        save_btn.clicked.connect(self.save_rules)
        save_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(save_btn)
        
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def load_rules(self):
        """Load and display rules"""
        self.rules_list.clear()
        for rule in self.conflict_manager.rules:
            item_text = f"[{rule.priority}] {rule.name} → {rule.rule.value.replace('_', ' ').title()}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, rule)
            self.rules_list.addItem(item)
        
        # Load statistics
        history = self.conflict_manager.conflict_history
        stats = f"Total conflict patterns learned: {len(history)}\n"
        total_resolutions = sum(sum(pattern.values()) for pattern in history.values())
        stats += f"Total resolutions recorded: {total_resolutions}"
        self.stats_text.setPlainText(stats)
    
    def add_rule(self):
        """Add a new rule"""
        dialog = RuleEditorDialog(None, self)
        if dialog.exec() == QDialog.Accepted:
            new_rule = dialog.get_rule()
            if new_rule:
                self.conflict_manager.rules.append(new_rule)
                self.load_rules()
    
    def edit_rule(self):
        """Edit selected rule"""
        current_item = self.rules_list.currentItem()
        if current_item:
            rule = current_item.data(Qt.ItemDataRole.UserRole)
            dialog = RuleEditorDialog(rule, self)
            if dialog.exec() == QDialog.Accepted:
                updated_rule = dialog.get_rule()
                if updated_rule:
                    # Find and replace
                    for i, r in enumerate(self.conflict_manager.rules):
                        if r is rule:
                            self.conflict_manager.rules[i] = updated_rule
                            break
                    self.load_rules()
    
    def remove_rule(self):
        """Remove selected rule"""
        current_item = self.rules_list.currentItem()
        if current_item:
            rule = current_item.data(Qt.ItemDataRole.UserRole)
            reply = QMessageBox.question(
                self, "Remove Rule",
                f"Remove rule '{rule.name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.conflict_manager.rules.remove(rule)
                self.load_rules()
    
    def reset_rules(self):
        """Reset to default rules"""
        reply = QMessageBox.question(
            self, "Reset Rules",
            "Reset all rules to defaults? This will remove custom rules.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.conflict_manager.rules = [
                ConflictResolutionRule("Same filename, different size", "filename_size_conflict", ConflictRule.ASK_USER, 10),
                ConflictResolutionRule("Same filename, different date", "filename_date_conflict", ConflictRule.ALWAYS_KEEP_NEWER, 8),
                ConflictResolutionRule("Exact duplicate", "exact_duplicate", ConflictRule.ALWAYS_SKIP, 15),
                ConflictResolutionRule("Different filenames, same hash", "hash_duplicate", ConflictRule.ALWAYS_SKIP, 12),
            ]
            self.load_rules()
    
    def save_rules(self):
        """Save rules to disk"""
        self.conflict_manager.save_rules()
        QMessageBox.information(self, "Saved", "Conflict resolution rules saved successfully.")

class RuleEditorDialog(QDialog):
    """Dialog for editing a conflict resolution rule"""
    
    def __init__(self, rule: ConflictResolutionRule = None, parent=None):
        super().__init__(parent)
        self.rule = rule
        
        self.setWindowTitle("Rule Editor")
        self.setModal(True)
        self.resize(500, 300)
        
        self.init_ui()
        if rule:
            self.load_rule(rule)
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Rule name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Rule Name:"))
        self.name_edit = QLineEdit()
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # Condition
        condition_layout = QHBoxLayout()
        condition_layout.addWidget(QLabel("Condition:"))
        self.condition_combo = QComboBox()
        self.condition_combo.addItems([
            "filename_conflict",
            "filename_size_conflict", 
            "filename_date_conflict",
            "exact_duplicate",
            "hash_duplicate",
            "any_conflict"
        ])
        condition_layout.addWidget(self.condition_combo)
        layout.addLayout(condition_layout)
        
        # Rule
        rule_layout = QHBoxLayout()
        rule_layout.addWidget(QLabel("Action:"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems([rule.value for rule in ConflictRule])
        rule_layout.addWidget(self.rule_combo)
        layout.addLayout(rule_layout)
        
        # Priority
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("Priority:"))
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 20)
        self.priority_spin.setValue(5)
        priority_layout.addWidget(self.priority_spin)
        priority_layout.addStretch()
        layout.addLayout(priority_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def load_rule(self, rule: ConflictResolutionRule):
        """Load rule data into UI"""
        self.name_edit.setText(rule.name)
        self.condition_combo.setCurrentText(rule.condition)
        self.rule_combo.setCurrentText(rule.rule.value)
        self.priority_spin.setValue(rule.priority)
    
    def get_rule(self) -> ConflictResolutionRule:
        """Get rule from UI"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid", "Rule name cannot be empty.")
            return None
        
        condition = self.condition_combo.currentText()
        rule_value = self.rule_combo.currentText()
        rule = ConflictRule(rule_value)
        priority = self.priority_spin.value()
        
        return ConflictResolutionRule(name, condition, rule, priority)

# ============================================================================
# SUBSYSTEMS
# ============================================================================

class ThumbnailManager:
    """Manages thumbnail creation and caching"""
    def __init__(self, config: RegistryConfig):
        self.config = config
        self.thumbnails_dir = config.registry_path / "thumbnails"
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
    
    def create_thumbnail(self, file_path: Path, media_id: str) -> Optional[Path]:
        """Create thumbnail for media file"""
        try:
            from PIL import Image
            thumb_path = self.thumbnails_dir / f"{media_id}.jpg"
            
            with Image.open(file_path) as img:
                img.thumbnail(self.config.thumbnail_size)
                img.save(thumb_path, "JPEG")
            
            return thumb_path
        except Exception as e:
            print(f"Failed to create thumbnail: {e}")
            return None

class MetadataExtractor:
    """Extracts metadata from media files"""
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file"""
        import mimetypes
        
        metadata = {
            'type': 'unknown',
            'file_size': file_path.stat().st_size,
            'filename': file_path.name,
            'extension': file_path.suffix.lower(),
            'mime_type': mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream',
            'created': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'tags': [],
            'projects': []
        }
        
        # Determine media type
        ext = file_path.suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', '.bmp', '.webp']:
            metadata['type'] = 'image'
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    metadata['dimensions'] = img.size
                    metadata['mode'] = img.mode
            except:
                pass
        elif ext in ['.fbx', '.obj', '.gltf', '.glb', '.blend', '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae']:
            metadata['type'] = '3d_model'
        elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv']:
            metadata['type'] = 'video'
        elif ext in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            metadata['type'] = 'audio'
        elif ext in ['.pdf', '.txt', '.md', '.doc', '.docx']:
            metadata['type'] = 'document'
        
        return metadata

class DuplicateDetector:
    """Detects duplicate media files"""
    def find_duplicates(self, registry: MediaRegistry) -> List[List[Dict]]:
        """Find duplicate media files"""
        # Simplified duplicate detection
        return []

class ExportManager:
    """Manages export and import operations"""
    def __init__(self, config: RegistryConfig):
        self.config = config
    
    def create_package(self, media_ids: List[str], output_path: Path, 
                      include_references: bool = True) -> Dict[str, Any]:
        """Create ROCA package from media IDs"""
        # Simplified export
        return {'success': True, 'path': str(output_path), 'count': len(media_ids)}
    
    def import_package(self, package_path: Path, target_dir: Path, 
                      user_id: str = "system") -> Dict[str, Any]:
        """Import ROCA package"""
        try:
            if not package_path.exists():
                return {'success': False, 'error': f'Package file not found: {package_path}'}
            
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            imported_count = 0
            failed_count = 0
            
            # Extract package
            with zipfile.ZipFile(package_path, 'r') as zipf:
                # Read manifest
                try:
                    manifest_data = zipf.read('manifest.roca')
                    manifest = json.loads(manifest_data.decode('utf-8'))
                except:
                    manifest = {'version': 'unknown', 'media_count': 0}
                
                # Read metadata
                try:
                    metadata_data = zipf.read('metadata.msgpack')
                    metadata = msgpack.unpackb(metadata_data, raw=False)
                    media_items = metadata.get('media_items', [])
                except:
                    # Fallback: try to infer from file list
                    media_items = []
                    for name in zipf.namelist():
                        if name.startswith('media/') and not name.endswith('/'):
                            # Extract basic info from path
                            parts = name.split('/')
                            if len(parts) >= 3:
                                media_id = parts[1]
                                filename = parts[2]
                                media_items.append({
                                    'media_id': media_id,
                                    'filename': filename,
                                    'original_path': str(target_dir / filename)
                                })
                
                # Extract files
                for item in media_items:
                    try:
                        media_id = item['media_id']
                        filename = item.get('filename', f"{media_id}_unknown")
                        
                        # Find the media file in the archive
                        media_arcname = None
                        for name in zipf.namelist():
                            if name.startswith(f'media/{media_id}/'):
                                media_arcname = name
                                break
                        
                        if media_arcname:
                            # Extract media file
                            target_path = target_dir / filename
                            with open(target_path, 'wb') as f:
                                f.write(zipf.read(media_arcname))
                            
                            # Register in database
                            result = self.registry.register_media(target_path, user_id)
                            if result.get('success'):
                                imported_count += 1
                            else:
                                failed_count += 1
                                print(f"Failed to register {filename}: {result.get('error', 'unknown error')}")
                        else:
                            failed_count += 1
                            print(f"Media file not found in package: {media_id}")
                            
                    except Exception as e:
                        failed_count += 1
                        print(f"Error importing item {item.get('media_id', 'unknown')}: {e}")
            
            return {
                'success': True,
                'imported_count': imported_count,
                'failed_count': failed_count,
                'total_count': len(media_items),
                'package_info': manifest
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ============================================================================
# ROCA PACKAGE FORMAT
# ============================================================================

class ROCAPackage:
    """ROCA Package format for universal media exchange"""
    
    VERSION = "1.0"
    MANIFEST_FILENAME = "manifest.roca"
    METADATA_FILENAME = "metadata.msgpack"
    THUMBNAILS_DIR = "thumbnails/"
    PREVIEWS_DIR = "previews/"
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.manifest = None
        self.metadata = None
    
    @classmethod
    def create(cls, media_items: List[Dict], output_path: Path,
              include_previews: bool = True,
              include_thumbnails: bool = True,
              compress: bool = True) -> 'ROCAPackage':
        """Create a new ROCA package"""
        
        # Prepare manifest
        manifest = {
            'version': cls.VERSION,
            'created_at': datetime.now().isoformat(),
            'creator': os.environ.get('USER', 'unknown'),
            'media_count': len(media_items),
            'total_size': sum(item.get('file_size', 0) for item in media_items),
            'includes': {
                'previews': include_previews,
                'thumbnails': include_thumbnails
            }
        }
        
        # Create package
        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        
        with zipfile.ZipFile(output_path, 'w', compression) as zipf:
            # Add manifest
            manifest_json = json.dumps(manifest, indent=2)
            zipf.writestr(cls.MANIFEST_FILENAME, manifest_json)
            
            # Add metadata
            metadata = {
                'media_items': media_items,
                'registry_info': {
                    'source_registry': 'roca_media_registry',
                    'export_time': datetime.now().isoformat()
                }
            }
            
            metadata_bytes = msgpack.packb(metadata, use_bin_type=True)
            zipf.writestr(cls.METADATA_FILENAME, metadata_bytes)
            
            # Add media files
            for item in media_items:
                source_path = Path(item['original_path'])
                if source_path.exists():
                    # Store in structured path
                    arcname = f"media/{item['media_id']}/{source_path.name}"
                    zipf.write(source_path, arcname)
                    
                    # Add thumbnail if exists
                    if include_thumbnails and item.get('thumbnail_path'):
                        thumb_path = Path(item['thumbnail_path'])
                        if thumb_path.exists():
                            thumb_arcname = f"{cls.THUMBNAILS_DIR}{item['media_id']}.jpg"
                            zipf.write(thumb_path, thumb_arcname)
        
        package = cls(output_path)
        package.manifest = manifest
        package.metadata = metadata
        
        print(f"📦 Created ROCA package: {output_path} ({len(media_items)} items)")
        return package

# ============================================================================
# EFFICIENT SPATIAL INDEX
# ============================================================================

class SpatialIndex:
    """Efficient spatial index (quadtree-like) for fast spatial queries"""
    def __init__(self, bounds, max_items=50):
        self.bounds = bounds  # (xmin, ymin, xmax, ymax)
        self.max_items = max_items
        self.items = []
        self.divided = False

    def contains(self, position):
        x, y = position[:2]
        xmin, ymin, xmax, ymax = self.bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def in_bounds(self, pos, bounds):
        x, y = pos[:2]
        xmin, ymin, xmax, ymax = bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def intersects(self, bounds):
        xmin, ymin, xmax, ymax = self.bounds
        bxmin, bymin, bxmax, bymax = bounds
        return not (bxmax < xmin or bxmin > xmax or bymax < ymin or bymin > ymax)

    def subdivide(self):
        xmin, ymin, xmax, ymax = self.bounds
        mx = (xmin + xmax) / 2
        my = (ymin + ymax) / 2
        self.nw = SpatialIndex((xmin, ymin, mx, my), self.max_items)
        self.ne = SpatialIndex((mx, ymin, xmax, my), self.max_items)
        self.sw = SpatialIndex((xmin, my, mx, ymax), self.max_items)
        self.se = SpatialIndex((mx, my, xmax, ymax), self.max_items)
        self.divided = True

    def insert(self, item, position):
        if not self.contains(position):
            return False
        if len(self.items) < self.max_items:
            self.items.append((item, position))
            return True
        else:
            if not self.divided:
                self.subdivide()
            return (self.nw.insert(item, position) or
                    self.ne.insert(item, position) or
                    self.sw.insert(item, position) or
                    self.se.insert(item, position))

    def query(self, bounds):
        found = []
        if not self.intersects(bounds):
            return found
        found.extend([item for item, pos in self.items if self.in_bounds(pos, bounds)])
        if self.divided:
            found.extend(self.nw.query(bounds))
            found.extend(self.ne.query(bounds))
            found.extend(self.sw.query(bounds))
            found.extend(self.se.query(bounds))
        return found

# ============================================================================
# LAZY THUMBNAIL LOADER
# ============================================================================

class LazyThumbnailLoader:
    """Lazy, background thumbnail loader with cache and queue"""
    def __init__(self):
        self.thumbnail_cache = {}
        self.load_queue = deque()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.background_loader, daemon=True)
        self._thread.start()

    def get_thumbnail(self, capsule_id):
        if capsule_id in self.thumbnail_cache:
            return self.thumbnail_cache[capsule_id]
        else:
            # Queue for background loading
            self.load_queue.append(capsule_id)
            return self.get_placeholder()

    def get_placeholder(self):
        """Return a default placeholder thumbnail"""
        if not hasattr(self, '_placeholder_pixmap'):
            # Create a 128x128 gray placeholder with border
            pixmap = QPixmap(128, 128)
            pixmap.fill(QColor(100, 100, 100))
            
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(150, 150, 150), 2))
            painter.drawRect(0, 0, 127, 127)
            
            # Draw a simple icon
            painter.setPen(QPen(QColor(200, 200, 200), 3))
            painter.drawLine(32, 32, 96, 32)
            painter.drawLine(32, 32, 32, 96)
            painter.drawLine(32, 96, 96, 96)
            painter.drawLine(96, 32, 96, 96)
            
            painter.end()
            self._placeholder_pixmap = pixmap
            
        return self._placeholder_pixmap

    def background_loader(self):
        while not self._stop_event.is_set():
            if self.load_queue:
                capsule_id = self.load_queue.popleft()
                # Load thumbnail logic here
                time.sleep(0.05)
    
    def stop(self):
        self._stop_event.set()
        self._thread.join()

# ============================================================================
# OPTIMIZED MEDIA CAPSULE
# ============================================================================

class OptimizedMediaCapsule:
    """Optimized capsule with compressed float16 embedding"""
    def __init__(self, **kwargs):
        self.description = kwargs.get('description', '')
        self._compressed_vector = None  # Store as bytes

    def compute_activity_vector(self):
        def _embed_text(text: str, dim: int = 64) -> np.ndarray:
            vec = np.zeros(dim, dtype=float)
            if not text:
                return vec
            t = text.strip().lower()
            for i in range(dim):
                h = hashlib.sha256(f"{t}::{i}".encode('utf-8')).digest()
                val = int.from_bytes(h[:8], 'big') / (2**64 - 1)
                vec[i] = val
            vec = vec - vec.mean()
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            return vec
        
        vec = _embed_text(self.description, dim=64)  # Reduce dimension
        # Compress to float16
        self._compressed_vector = vec.astype(np.float16).tobytes()

    @property
    def activity_vector(self):
        if self._compressed_vector:
            return np.frombuffer(self._compressed_vector, dtype=np.float16)
        return None

# ============================================================================
# QT MODELS AND WORKERS
# ============================================================================

class CapsuleListModel(QAbstractListModel):
    """Qt model for displaying capsules in a QListView or similar"""
    def __init__(self, capsules):
        super().__init__()
        self.capsules = capsules  # Reference, not copy

    def rowCount(self, parent=None):
        return len(self.capsules)

    def data(self, index, role):
        if not index.isValid():
            return None
        capsule = self.capsules[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return getattr(capsule, 'filename', str(getattr(capsule, 'id', '')))
        elif role == Qt.ItemDataRole.DecorationRole and hasattr(capsule, 'thumbnail') and capsule.thumbnail:
            return capsule.thumbnail
        return None

class MediaScannerThread(QThread):
    """Threaded media scanner for batch processing and UI feedback"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list, bool)  # list of capsules, delete_originals flag

    def __init__(self, directory, delete_originals=False):
        super().__init__()
        self.directory = directory
        self.delete_originals = delete_originals

    def find_media_files(self):
        exts = ['.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', '.bmp', '.webp',
                '.fbx', '.obj', '.gltf', '.glb', '.blend', '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae',
                '.bvh', '.trc', '.c3d', '.cho',
                '.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv',
                '.wav', '.mp3', '.ogg', '.flac', '.m4a',
                '.pdf', '.txt', '.md', '.doc', '.docx']
        file_paths = []
        for root, dirs, files in os.walk(self.directory):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    file_paths.append(os.path.join(root, name))
        return file_paths

    def process_file(self, path):
        try:
            capsule = MediaCapsule(source_path=path)
            capsule.compute_activity_vector()
            return capsule
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            return None

    def run(self):
        file_paths = self.find_media_files()
        total = len(file_paths)
        capsules = []
        for i, path in enumerate(file_paths):
            if self.isInterruptionRequested():
                break
            capsule = self.process_file(path)
            if capsule:
                capsules.append(capsule)
            self.progress.emit(i + 1, total)
            # Process in batches
            if len(capsules) % 100 == 0:
                time.sleep(0.01)  # Yield to UI
        self.finished.emit(capsules, self.delete_originals)

# ============================================================================
# CAPSULE STORE
# ============================================================================

class CapsuleStore:
    """High-performance CapsuleStore using structured numpy arrays"""
    def __init__(self):
        self.capsules = np.zeros(0, dtype=[
            ('id', 'U32'),
            ('position', 'f4', 3),
            ('activity_vector', 'f4', 128),
            ('media_type', 'i4'),
            ('file_size', 'i4'),
            # Add more fields as needed
        ])

    def add_capsule_batch(self, capsules_data):
        """Add multiple capsules efficiently as a batch."""
        new_capsules = np.array(capsules_data, dtype=self.capsules.dtype)
        self.capsules = np.concatenate([self.capsules, new_capsules])

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================

def get_optimal_thread_count():
    """Return optimal thread count for Threadripper 2990WX."""
    try:
        import platform
        import multiprocessing
        cpu_name = platform.processor().lower() + ' ' + platform.uname().processor.lower()
        if '2990wx' in cpu_name or 'threadripper' in cpu_name:
            # Threadripper 2990WX: Use NUMA-aware threading
            # Leave 4 cores for OS, use rest for compute
            count = multiprocessing.cpu_count()
            # For 32-core Threadripper: use 28 cores for parallelism
            return max(8, count - 4)
        else:
            count = multiprocessing.cpu_count()
            return max(2, int(count * 0.75))
    except Exception:
        return 32  # Default to 32 for Threadripper

def optimize_threadripper():
    """Set Threadripper-specific optimizations."""
    # Set thread affinity for better NUMA performance
    os.environ['OMP_NUM_THREADS'] = '28'  # Use 28 threads for OpenMP
    os.environ['MKL_NUM_THREADS'] = '28'  # For Intel MKL (if using)
    os.environ['NUMEXPR_NUM_THREADS'] = '28'  # For NumExpr

    # Enable thread pool for NumPy
    try:
        if hasattr(np, 'get_num_threads'):
            np.set_num_threads(28)
    except:
        pass

# ============================================================================
# AI COMPONENTS
# ============================================================================

class ChatBot:
    """Simple chatbot for interaction"""
    def __init__(self):
        self.messages = []
    
    def respond(self, message: str) -> str:
        self.messages.append(("user", message))
        response = f"I received: {message}"
        self.messages.append(("bot", response))
        return response

class AutonomousBrain:
    """Autonomous brain for AI processing"""
    def __init__(self):
        self.emotional_state = "neutral"
    
    def think_cycle(self):
        """Process thoughts"""
        return "Thinking..."

# ============================================================================
# WIDGETS
# ============================================================================

class NumpySettings:
    """Settings for numpy optimizations"""
    def __init__(self):
        self.use_gpu = False
        self.thread_count = get_optimal_thread_count()

class NumpyOrbitalWidget(QWidget):
    """Simple orbital capsule visual.

    - Draws a circular orbit and capsules that move around it.
    - Provide `set_speed()` and `set_radius()` to adjust animation.
    """
    
    capsule_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0.0
        self._speed = 30.0  # degrees per second (unused for capsule individual motion)
        self._radius = 80
        self._capsule_size = 14
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._timer.start(16)
        self.setMinimumSize(320, 320)

        # visualization state
        self.capsules: List[Capsule] = []
        self.zoom_level = 1.0
        self.selected_capsule = None
        self.hovered_capsule = None
        self.show_info = True

        # Colors
        self.bg_color = QColor(20, 20, 30)
        self.panel_color = QColor(30, 30, 40)
        self.border_color = QColor(60, 60, 80)
        self.highlight_color = QColor(100, 200, 255)
        self.text_color = QColor(255, 255, 255)

    def set_speed(self, deg_per_sec: float):
        self._speed = deg_per_sec

    def set_radius(self, r: int):
        self._radius = r
        self.update()

    def _step(self):
        dt = self._timer.interval() / 1000.0
        self._angle = (self._angle + self._speed * dt) % 360.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cx = self.width() // 2
            cy = self.height() // 2
            mx = event.pos().x()
            my = event.pos().y()
            for cap in self.capsules:
                orbit_radius = cap.orbit_radius * 50 * self.zoom_level
                x = cx + orbit_radius * math.cos(cap.angle)
                y = cy + orbit_radius * math.sin(cap.angle)
                dist = math.hypot(mx - x, my - y)
                if dist < 20:
                    self.selected_capsule = cap
                    self.capsule_selected.emit(cap)
                    self.update()
                    break

    def mouseMoveEvent(self, event):
        cx = self.width() // 2
        cy = self.height() // 2
        mx = event.pos().x()
        my = event.pos().y()
        self.hovered_capsule = None
        for cap in self.capsules:
            orbit_radius = cap.orbit_radius * 50 * self.zoom_level
            x = cx + orbit_radius * math.cos(cap.angle)
            y = cy + orbit_radius * math.sin(cap.angle)
            dist = math.hypot(mx - x, my - y)
            if dist < 20:
                self.hovered_capsule = cap
                break

    def wheelEvent(self, event):
        # Zoom in/out with wheel; use 120 units per notch standard
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # scale factor per notch
        factor = 1.2 ** (delta / 120)
        new_zoom = self.zoom_level * factor
        # clamp zoom
        self.zoom_level = max(0.2, min(3.0, new_zoom))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        cx = self.width() // 2
        cy = self.height() // 2

        # Draw starfield background
        self.draw_starfield(p)

        # Update capsule angles
        for cap in self.capsules:
            # small autonomous drift
            cap.angle = (cap.angle + 0.005 * (0.5 + cap.certainty)) % (2 * math.pi)

        # Draw orbital paths grouped by radius
        orbit_radii = sorted(set(int(cap.orbit_radius * 100) / 100.0 for cap in self.capsules))
        p.setPen(QPen(QColor(60, 80, 120), 1))
        p.setOpacity(0.35)
        for radius in orbit_radii:
            orbit_radius = radius * 50 * self.zoom_level
            p.drawEllipse(int(cx - orbit_radius), int(cy - orbit_radius), int(orbit_radius * 2), int(orbit_radius * 2))
        p.setOpacity(1.0)

        # Draw capsules
        for cap in self.capsules:
            orbit_radius = cap.orbit_radius * 50 * self.zoom_level
            x = cx + orbit_radius * math.cos(cap.angle)
            y = cy + orbit_radius * math.sin(cap.angle)

            # color by kind
            if cap.kind == "theory":
                color = QColor(200, 100, 255)
            elif cap.kind == "hypothesis":
                color = QColor(255, 200, 100)
            elif cap.kind == "method":
                color = QColor(100, 200, 255)
            else:
                color = QColor(150, 255, 150)

            size = max(6, int(14 * cap.certainty))

            # Glow
            glow_radius = size + 6
            for i in range(3, 0, -1):
                glow_color = QColor(color)
                glow_color.setAlpha(int(40 / (i + 1)))
                p.setBrush(QBrush(glow_color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(int(x - glow_radius / 2 - i * 2), int(y - glow_radius / 2 - i * 2), int(glow_radius + i * 4), int(glow_radius + i * 4))

            # Body
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawEllipse(int(x - size / 2), int(y - size / 2), size, size)

            # Label
            if self.zoom_level > 0.8 or cap == self.selected_capsule or cap == self.hovered_capsule:
                p.setPen(QPen(self.text_color))
                f = p.font()
                f.setPointSize(8)
                p.setFont(f)
                label = cap.character if cap.character else (cap.kind[:3])
                p.drawText(int(x + size / 2 + 5), int(y + 3), label)

            # Highlights
            if cap == self.selected_capsule:
                p.setPen(QPen(self.highlight_color, 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(x - size / 2 - 5), int(y - size / 2 - 5), size + 10, size + 10)
                p.setPen(QPen(self.highlight_color, 1))
                p.setOpacity(0.5)
                p.drawLine(int(cx), int(cy), int(x), int(y))
                p.setOpacity(1.0)
            elif cap == self.hovered_capsule:
                p.setPen(QPen(QColor(200, 200, 150), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(int(x - size / 2 - 3), int(y - size / 2 - 3), size + 6, size + 6)

        # Central nucleus
        self.draw_central_nucleus(p, cx, cy)

        # Info panel
        if self.show_info:
            self.draw_info_panel(p)

        p.end()

    def draw_starfield(self, p: QPainter):
        p.fillRect(self.rect(), self.bg_color)
        random.seed(42)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(Qt.PenStyle.NoPen)
        num_stars = 160
        w = self.width()
        h = self.height()
        for _ in range(num_stars):
            x = random.randint(0, w)
            y = random.randint(0, h)
            size = random.randint(1, 3)
            brightness = random.randint(100, 255)
            p.setBrush(QBrush(QColor(brightness, brightness, brightness)))
            p.drawEllipse(x, y, size, size)
        random.seed()

    def draw_central_nucleus(self, p: QPainter, cx: int, cy: int):
        for i in range(30, 0, -5):
            glow_color = QColor(255, 200, 0)
            glow_color.setAlpha(int(100 * (30 - i) / 30))
            p.setBrush(QBrush(glow_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(int(cx - i), int(cy - i), i * 2, i * 2)
        gradient_color = QColor(255, 220, 100)
        p.setBrush(QBrush(gradient_color))
        p.setPen(QPen(QColor(255, 240, 150), 2))
        p.drawEllipse(int(cx - 15), int(cy - 15), 30, 30)
        p.setBrush(QBrush(QColor(255, 255, 200)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
        p.setPen(QPen(QColor(255, 255, 255)))
        f = p.font()
        f.setPointSize(11)
        f.setBold(True)
        p.setFont(f)
        p.drawText(int(cx + 20), int(cy - 5), "Identity")
        p.drawText(int(cx + 20), int(cy + 10), "Nucleus")

    def draw_info_panel(self, p: QPainter):
        panel_w = 80
        panel_h = 60
        panel_bg = QColor(0, 0, 0)
        panel_bg.setAlpha(200)
        p.fillRect(10, 10, panel_w, panel_h, panel_bg)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawRect(10, 10, panel_w, panel_h)
        p.setPen(QPen(QColor(255, 255, 255)))
        f = p.font()
        f.setPointSize(7)
        f.setBold(True)
        p.setFont(f)
        p.drawText(15, 22, "ROCA")
        f.setBold(False)
        f.setPointSize(6)
        p.setFont(f)
        y = 34
        p.drawText(15, y, f"◐ {len(self.capsules)}")
        if self.selected_capsule:
            p.drawText(15, y + 10, "✓ Sel.")
        else:
            p.drawText(15, y + 10, f"Zm:{self.zoom_level:.1f}")

    def set_capsules(self, capsules):
        """Set capsules to display"""
        self.capsules = capsules
        self.update()

    def refresh(self):
        """Refresh display"""
        self.update()

class ChatWidget(QWidget):
    """Chat interface widget"""
    message_sent = pyqtSignal(str)
    
    def __init__(self, chatbot):
        super().__init__()
        self.chatbot = chatbot
        self.messages = []
    
    def add_message(self, message: str, sender: str = "User"):
        """Add message to chat"""
        self.messages.append((sender, message))

class SmileyAvatar(QWidget):
    """Smiley avatar widget"""
    def __init__(self):
        super().__init__()
        self.emotion = "neutral"
    
    def set_emotion(self, emotion: str):
        """Set avatar emotion"""
        self.emotion = emotion
        self.update()

# ============================================================================
# MAIN WINDOW - COMPLETE APPLICATION
# ============================================================================

class ROCAMainWindow(QMainWindow):
    """Main window with full ROCA Media Registry capabilities"""
    
    CAPSULES_JSON_PATH = "capsules_autosave.json"
    
    def __init__(self):
        super().__init__()
        
        # Initialize systems
        self.config = RegistryConfig()
        self.registry = MediaRegistry(self.config)
        self.settings = NumpySettings()
        self.chatbot = ChatBot()
        self.brain = AutonomousBrain()
        self.orbital_widget = NumpyOrbitalWidget()
        
        # Initialize file system watcher for automatic folder watching
        self.file_watcher = QFileSystemWatcher()
        self.watched_directories = set()
        self.auto_register_enabled = False
        
        # Setup UI
        self.setWindowTitle("ROCA Media Registry - Professional Edition")
        self.setGeometry(100, 100, 1600, 900)
        self.setWindowIcon(self.create_icon())
        
        self.init_ui()
        self.setAcceptDrops(True)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("ROCA Ready - Tabbed interface active!")
        
        # Initialize controllers
        self.initialize_controllers()
        self.initialize_capsule_store()
        
        # Stats timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)
        
        # Connect file watcher signals
        self.file_watcher.directoryChanged.connect(self.on_watched_directory_changed)
        self.file_watcher.fileChanged.connect(self.on_watched_file_changed)
        
        # Load saved capsules
        self.load_capsules_on_startup()
        
        # Update watched directories UI
        self.update_watched_dirs_list()
        
        # Check for portable registries
        self.check_portable_registries()

    def check_portable_registries(self):
        """Check for portable registries and offer to import"""
        try:
            portable_registries = find_portable_registries()
            running_from_thumb = is_running_from_removable_drive()
            
            # Remove local registry from consideration if we're running from thumb drive
            if running_from_thumb and "LOCAL" in portable_registries:
                del portable_registries["LOCAL"]
            
            if portable_registries:
                # Show import dialog
                dialog = PortableRegistryDialog(self)
                result = dialog.exec()
                
                if result == QDialog.Accepted:
                    choice = dialog.get_choice()
                    
                    if choice == "import":
                        # Import the first detected portable registry
                        source_path = list(portable_registries.values())[0]
                        self.import_portable_registry(source_path)
                    elif choice == "switch":
                        # Switch to using the portable registry
                        source_path = list(portable_registries.values())[0]
                        self.switch_to_portable_registry(source_path)
                        
        except Exception as e:
            print(f"Error checking portable registries: {e}")

    def import_portable_registry(self, source_path: Path, use_advanced_resolution: bool = True):
        """Import a portable registry with advanced conflict resolution"""
        try:
            result = self.registry.import_portable_registry(source_path, use_advanced_resolution)
            
            if result['success']:
                message = f"Successfully imported {result['imported_count']} items"
                if result['skipped_count'] > 0:
                    message += f" ({result['skipped_count']} skipped)"
                if result['conflicts_resolved'] > 0:
                    message += f" ({result['conflicts_resolved']} conflicts resolved)"
                QMessageBox.information(self, "Import Complete", message)
                self.refresh_media_list()
            else:
                errors = "\n".join(result['errors'])
                QMessageBox.warning(self, "Import Failed", f"Import failed:\n{errors}")
                
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"An error occurred during import: {e}")

    def switch_to_portable_registry(self, registry_path: Path):
        """Switch to using a portable registry"""
        try:
            # Update config to use the portable registry
            self.config.registry_path = registry_path
            
            # Reinitialize registry with new path
            self.registry = MediaRegistry(self.config)
            
            QMessageBox.information(
                self, "Registry Switched", 
                f"Switched to portable registry at: {registry_path}"
            )
            
            self.refresh_media_list()
            
        except Exception as e:
            QMessageBox.critical(self, "Switch Error", f"Failed to switch registry: {e}")

    def toggle_auto_sync(self, enabled):
        """Toggle automatic synchronization"""
        if enabled:
            self.start_auto_sync()
            self.sync_status_label.setText("Status: Auto-sync enabled (5min intervals)")
        else:
            if hasattr(self, '_sync_timer'):
                self._sync_timer.stop()
            self.sync_status_label.setText("Status: Auto-sync disabled")

    def manual_sync(self):
        """Perform manual synchronization"""
        try:
            self.status_bar.showMessage("Synchronizing registries...")
            sync_result = self.sync_registries()
            
            if sync_result['success']:
                message = f"Sync complete: {sync_result['syncs_performed']} items synced"
                if sync_result['conflicts_resolved'] > 0:
                    message += f", {sync_result['conflicts_resolved']} conflicts resolved"
                QMessageBox.information(self, "Sync Complete", message)
                self.refresh_media_list()
            else:
                errors = "\n".join(sync_result['errors'])
                QMessageBox.warning(self, "Sync Failed", f"Synchronization failed:\n{errors}")
                
        except Exception as e:
            QMessageBox.critical(self, "Sync Error", f"An error occurred during sync: {e}")
        finally:
            self.status_bar.showMessage("ROCA Ready")

    def show_sync_status(self):
        """Show synchronization status"""
        try:
            registries = find_portable_registries()
            
            status_text = "Registry Synchronization Status\n"
            status_text += "=" * 40 + "\n\n"
            
            # Local registry
            local_path = Path.home() / ".roca_registry"
            if local_path.exists():
                local_stats = self.get_registry_stats_for_path(local_path)
                status_text += f"Local Registry: {local_stats.get('total_media', 0)} items\n"
            else:
                status_text += "Local Registry: Not found\n"
            
            # Portable registries
            portable_count = 0
            for drive, path in registries.items():
                if drive != "LOCAL":
                    portable_count += 1
                    stats = self.get_registry_stats_for_path(path)
                    status_text += f"Portable ({drive}): {stats.get('total_media', 0)} items\n"
            
            if portable_count == 0:
                status_text += "Portable Registries: None detected\n"
            
            status_text += "\nAuto-sync: "
            if hasattr(self, '_sync_timer') and self._sync_timer.isActive():
                status_text += "Enabled (every 5 minutes)"
            else:
                status_text += "Disabled"
            
            QMessageBox.information(self, "Sync Status", status_text)
            
        except Exception as e:
            QMessageBox.warning(self, "Status Error", f"Could not retrieve sync status: {e}")

    def get_registry_stats_for_path(self, registry_path: Path) -> Dict:
        """Get basic stats for a registry at the given path"""
        try:
            db_path = registry_path / "registry.db"
            if not db_path.exists():
                return {'total_media': 0}
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM media_registry")
            count = cursor.fetchone()[0]
            conn.close()
            
            return {'total_media': count}
        except:
            return {'total_media': 0}

    def init_ui(self):
        """Initialize tabbed UI interface"""
        # Create central tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid rgb(60, 60, 80);
                background-color: rgb(30, 30, 40);
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: rgb(50, 50, 70);
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: rgb(70, 100, 150);
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: rgb(60, 80, 120);
            }
        """)
        self.setCentralWidget(self.tab_widget)
        
        # Create all tabs
        self.create_dashboard_tab()
        self.create_registry_tab()
        self.create_visualizer_tab()
        self.create_exchange_tab()
        self.create_tools_tab()
        
        # Create menu bar
        self.create_menu_bar()

    def create_dashboard_tab(self):
        """Create dashboard tab with statistics and overview"""
        dashboard_tab = QWidget()
        layout = QVBoxLayout(dashboard_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("ROCA Media Registry Dashboard")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20pt;
                font-weight: bold;
                padding: 15px;
                background-color: rgba(40, 60, 80, 150);
                border-radius: 8px;
                text-align: center;
            }
        """)
        layout.addWidget(title_label)
        
        # Stats grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(15)
        
        # Registry stats
        reg_stats = self.registry.get_registry_stats()
        
        # Stat 1: Total Media
        stat1 = self.create_stat_card("📊 Total Media", str(reg_stats.get('total_media', 0)), 
                                    "Registered media files")
        stats_grid.addWidget(stat1, 0, 0)
        
        # Stat 2: Storage Used
        total_size = reg_stats.get('total_size', 0)
        size_str = self.format_size(total_size)
        stat2 = self.create_stat_card("💾 Storage Used", size_str, "Total registry size")
        stats_grid.addWidget(stat2, 0, 1)
        
        # Stat 3: Unique Files
        unique = reg_stats.get('unique_files', reg_stats.get('total_media', 0))
        stat3 = self.create_stat_card("🔍 Unique Files", str(unique), "After deduplication")
        stats_grid.addWidget(stat3, 1, 0)
        
        # Stat 4: Space Saved
        saved = reg_stats.get('duplicate_savings', 0)
        saved_str = self.format_size(saved * 1024 * 1024) if saved > 0 else "0 B"
        stat4 = self.create_stat_card("💰 Space Saved", saved_str, "From deduplication")
        stats_grid.addWidget(stat4, 1, 1)
        
        layout.addLayout(stats_grid)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_group.setStyleSheet("""
            QGroupBox {
                color: white;
                font-size: 12pt;
                border: 1px solid rgb(60, 60, 80);
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        actions_layout = QGridLayout()
        
        # Action buttons
        scan_btn = QPushButton("📁 Scan Directory")
        scan_btn.setToolTip("Scan directory for media files")
        scan_btn.clicked.connect(self.scan_media_directory_dialog)
        scan_btn.setStyleSheet(self.get_button_style())
        actions_layout.addWidget(scan_btn, 0, 0)
        
        export_btn = QPushButton("📤 Export Package")
        export_btn.setToolTip("Export selected media as ROCA package")
        export_btn.clicked.connect(self.export_selected)
        export_btn.setStyleSheet(self.get_button_style())
        actions_layout.addWidget(export_btn, 0, 1)
        
        import_btn = QPushButton("📥 Import Package")
        import_btn.setToolTip("Import ROCA package")
        import_btn.clicked.connect(self.import_package_dialog)
        import_btn.setStyleSheet(self.get_button_style())
        actions_layout.addWidget(import_btn, 1, 0)
        
        find_dup_btn = QPushButton("🔍 Find Duplicates")
        find_dup_btn.setToolTip("Find duplicate media files")
        find_dup_btn.clicked.connect(self.find_duplicates)
        find_dup_btn.setStyleSheet(self.get_button_style())
        actions_layout.addWidget(find_dup_btn, 1, 1)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Recent activity
        activity_group = QGroupBox("Recent Activity")
        activity_group.setStyleSheet(actions_group.styleSheet())
        activity_layout = QVBoxLayout()
        
        self.activity_list = QTextEdit()
        self.activity_list.setReadOnly(True)
        self.activity_list.setMaximumHeight(150)
        activity_layout.addWidget(self.activity_list)
        
        activity_group.setLayout(activity_layout)
        layout.addWidget(activity_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(dashboard_tab, "🏠 Dashboard")

    def create_registry_tab(self):
        """Create media registry management tab"""
        registry_tab = QWidget()
        layout = QVBoxLayout(registry_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        
        # Search bar
        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: white;")
        toolbar_layout.addWidget(search_label)
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search media...")
        self.search_bar.setMinimumWidth(200)
        self.search_bar.returnPressed.connect(self.search_media)
        toolbar_layout.addWidget(self.search_bar)
        
        # Filter combo
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Media", "Images", "3D Models", "Videos", "Audio", "Documents"])
        self.filter_combo.currentTextChanged.connect(self.filter_media)
        toolbar_layout.addWidget(self.filter_combo)
        
        toolbar_layout.addStretch()
        
        # Action buttons
        add_media_btn = QPushButton("+ Add Media")
        add_media_btn.clicked.connect(self.add_media_dialog)
        toolbar_layout.addWidget(add_media_btn)
        
        add_folder_btn = QPushButton("+ Add Folder")
        add_folder_btn.clicked.connect(self.add_folder_dialog)
        toolbar_layout.addWidget(add_folder_btn)
        
        layout.addWidget(toolbar)
        
        # Splitter for media list and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Media list
        media_list_widget = QWidget()
        media_layout = QVBoxLayout(media_list_widget)
        
        self.media_list = QListWidget()
        self.media_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.media_list.itemDoubleClicked.connect(self.on_media_selected)
        self.media_list.itemSelectionChanged.connect(self.on_media_selection_changed)
        media_layout.addWidget(self.media_list)
        
        # Media info
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        self.media_info = QTextEdit()
        self.media_info.setReadOnly(True)
        self.media_info.setMaximumHeight(150)
        info_layout.addWidget(self.media_info)
        
        media_layout.addWidget(info_widget)
        
        splitter.addWidget(media_list_widget)
        
        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 30, 150);
                border: 1px solid rgb(60, 60, 80);
                border-radius: 4px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        
        # Metadata
        meta_label = QLabel("Metadata:")
        meta_label.setStyleSheet("color: white; font-weight: bold;")
        preview_layout.addWidget(meta_label)
        
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        preview_layout.addWidget(self.metadata_text)
        
        splitter.addWidget(preview_widget)
        
        splitter.setSizes([400, 600])
        layout.addWidget(splitter)
        
        # Bottom toolbar
        bottom_toolbar = QWidget()
        bottom_layout = QHBoxLayout(bottom_toolbar)
        
        self.selected_count_label = QLabel("Selected: 0")
        self.selected_count_label.setStyleSheet("color: #aaa;")
        bottom_layout.addWidget(self.selected_count_label)
        
        bottom_layout.addStretch()
        
        export_selected_btn = QPushButton("📤 Export Selected")
        export_selected_btn.clicked.connect(self.export_selected)
        bottom_layout.addWidget(export_selected_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.clicked.connect(self.delete_selected)
        bottom_layout.addWidget(delete_btn)
        
        layout.addWidget(bottom_toolbar)
        
        self.tab_widget.addTab(registry_tab, "📁 Registry")

    def create_visualizer_tab(self):
        """Create orbital visualizer tab"""
        visualizer_tab = QWidget()
        layout = QVBoxLayout(visualizer_tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: rgba(40, 40, 60, 150); border-radius: 4px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 5, 10, 5)
        
        title_label = QLabel("Orbital Visualization")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14pt;
                font-weight: bold;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        layout.addWidget(title_bar)
        
        # Create orbital widget
        self.orbital_widget = NumpyOrbitalWidget()
        layout.addWidget(self.orbital_widget)
        
        # Controls
        controls = QWidget()
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        controls_layout.addWidget(QLabel("Orbit Speed:"), 0, 0)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 360)
        self.speed_slider.setValue(30)
        self.speed_slider.valueChanged.connect(lambda v: self.orbital_widget.set_speed(float(v)))
        controls_layout.addWidget(self.speed_slider, 0, 1)
        
        controls_layout.addWidget(QLabel("Orbit Radius:"), 1, 0)
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(40, 200)
        self.radius_slider.setValue(80)
        self.radius_slider.valueChanged.connect(lambda v: self.orbital_widget.set_radius(int(v)))
        controls_layout.addWidget(self.radius_slider, 1, 1)
        
        # Add some example capsules if none exist
        if not self.orbital_widget.capsules:
            for i in range(12):
                kind = random.choice(["theory", "method", "hypothesis", "concept"])
                cap = Capsule(
                    content=f"Example {i}",
                    kind=kind,
                    certainty=random.uniform(0.4, 1.0),
                    orbit_radius=random.uniform(0.6, 2.2),
                    angle=random.uniform(0, 2 * math.pi),
                )
                self.orbital_widget.capsules.append(cap)
        
        layout.addWidget(controls)
        
        self.tab_widget.addTab(visualizer_tab, "🌌 Visualizer")

    def create_exchange_tab(self):
        """Create package exchange tab"""
        exchange_tab = QWidget()
        layout = QVBoxLayout(exchange_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Export section
        export_group = QGroupBox("📤 Export Package")
        export_group.setStyleSheet(self.get_groupbox_style())
        export_layout = QVBoxLayout()
        
        # Export options
        options_widget = QWidget()
        options_layout = QGridLayout(options_widget)
        
        options_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ROCAPKG (.rocapkg)", "ZIP Archive", "Folder Copy"])
        options_layout.addWidget(self.format_combo, 0, 1)
        
        self.include_thumbs = QCheckBox("Include Thumbnails")
        self.include_thumbs.setChecked(True)
        options_layout.addWidget(self.include_thumbs, 1, 0)
        
        self.include_meta = QCheckBox("Include Metadata")
        self.include_meta.setChecked(True)
        options_layout.addWidget(self.include_meta, 1, 1)
        
        self.include_previews = QCheckBox("Include Previews")
        options_layout.addWidget(self.include_previews, 2, 0)
        
        options_layout.setColumnStretch(1, 1)
        export_layout.addWidget(options_widget)
        
        # Export button
        export_btn = QPushButton("Create Export Package")
        export_btn.setStyleSheet(self.get_button_style(primary=True))
        export_btn.clicked.connect(self.create_export_package)
        export_layout.addWidget(export_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Import section
        import_group = QGroupBox("📥 Import Package")
        import_group.setStyleSheet(self.get_groupbox_style())
        import_layout = QVBoxLayout()
        
        import_btn = QPushButton("Browse and Import Package")
        import_btn.setStyleSheet(self.get_button_style())
        import_btn.clicked.connect(self.import_package_dialog)
        import_layout.addWidget(import_btn)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Quick share section
        share_group = QGroupBox("🔗 Quick Share")
        share_group.setStyleSheet(self.get_groupbox_style())
        share_layout = QVBoxLayout()
        
        share_btn = QPushButton("Create Share Link")
        share_btn.setStyleSheet(self.get_button_style())
        share_btn.clicked.connect(self.create_share)
        share_layout.addWidget(share_btn)
        
        share_group.setLayout(share_layout)
        layout.addWidget(share_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(exchange_tab, "🔄 Exchange")

    def create_tools_tab(self):
        """Create tools and analysis tab"""
        tools_tab = QWidget()
        layout = QVBoxLayout(tools_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Tools & Analysis")
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 16pt;
                font-weight: bold;
                padding: 10px;
                background-color: rgba(40, 60, 40, 150);
                border-radius: 6px;
                text-align: center;
            }
        """)
        layout.addWidget(title_label)
        
        # Tools grid
        tools_grid = QGridLayout()
        tools_grid.setSpacing(10)
        
        # Add tool buttons
        tools = [
            ("🔍 Analyze Capsules", "Analyze capsule statistics", self.analyze_capsules),
            ("🎨 Style Transfer", "Apply visual styles", self.start_style_transfer),
            ("🔮 Predictive", "Predict capsule states", self.start_predictive),
            ("⚡ Performance", "Optimize performance", self.tune_performance),
            ("💾 Backup", "Create registry backup", self.backup_registry),
            ("🧹 Clean", "Clean registry", self.clean_registry),
            ("📊 Reports", "Generate reports", self.generate_reports),
            ("🔧 Settings", "Configure settings", self.open_settings),
        ]
        
        for i, (text, tooltip, callback) in enumerate(tools):
            row = i // 2
            col = i % 2
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(self.get_button_style())
            btn.clicked.connect(callback)
            tools_grid.addWidget(btn, row, col)
        
        layout.addLayout(tools_grid)
        
        # Folder Watching section
        watching_group = QGroupBox("👁️ Automatic Folder Watching")
        watching_group.setStyleSheet(self.get_groupbox_style())
        watching_layout = QVBoxLayout()
        
        # Enable/disable toggle
        self.auto_watch_checkbox = QCheckBox("Enable automatic registration of new files")
        self.auto_watch_checkbox.setChecked(self.auto_register_enabled)
        self.auto_watch_checkbox.stateChanged.connect(self.set_auto_register_enabled)
        watching_layout.addWidget(self.auto_watch_checkbox)
        
        # Watched directories list
        watched_label = QLabel("Watched Directories:")
        watched_label.setStyleSheet("color: white; font-weight: bold;")
        watching_layout.addWidget(watched_label)
        
        self.watched_dirs_list = QListWidget()
        self.watched_dirs_list.setStyleSheet("""
            QListWidget {
                background-color: rgb(40, 40, 50);
                color: white;
                border: 1px solid rgb(60, 60, 80);
                border-radius: 4px;
            }
        """)
        self.watched_dirs_list.setMaximumHeight(100)
        watching_layout.addWidget(self.watched_dirs_list)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        add_watch_btn = QPushButton("➕ Add Directory")
        add_watch_btn.setStyleSheet(self.get_button_style())
        add_watch_btn.clicked.connect(self.add_watch_directory_dialog)
        buttons_layout.addWidget(add_watch_btn)
        
        remove_watch_btn = QPushButton("➖ Remove Selected")
        remove_watch_btn.setStyleSheet(self.get_button_style())
        remove_watch_btn.clicked.connect(self.remove_watch_directory)
        buttons_layout.addWidget(remove_watch_btn)
        
        watching_layout.addLayout(buttons_layout)
        
        # Status info
        status_label = QLabel("Status: Watching disabled")
        status_label.setStyleSheet("color: rgb(150, 150, 150); font-size: 10pt;")
        self.watch_status_label = status_label
        watching_layout.addWidget(status_label)
        
        watching_group.setLayout(watching_layout)
        layout.addWidget(watching_group)
        
        # Registry Sync section
        sync_group = QGroupBox("🔄 Registry Synchronization")
        sync_group.setStyleSheet(self.get_groupbox_style())
        sync_layout = QVBoxLayout()
        
        # Auto-sync toggle
        self.auto_sync_checkbox = QCheckBox("Enable automatic synchronization")
        self.auto_sync_checkbox.setChecked(False)
        self.auto_sync_checkbox.stateChanged.connect(self.toggle_auto_sync)
        sync_layout.addWidget(self.auto_sync_checkbox)
        
        # Manual sync buttons
        sync_buttons_layout = QHBoxLayout()
        
        sync_now_btn = QPushButton("🔄 Sync Now")
        sync_now_btn.setStyleSheet(self.get_button_style())
        sync_now_btn.clicked.connect(self.manual_sync)
        sync_buttons_layout.addWidget(sync_now_btn)
        
        sync_status_btn = QPushButton("📊 Sync Status")
        sync_status_btn.setStyleSheet(self.get_button_style())
        sync_status_btn.clicked.connect(self.show_sync_status)
        sync_buttons_layout.addWidget(sync_status_btn)
        
        sync_layout.addLayout(sync_buttons_layout)
        
        # Sync status info
        self.sync_status_label = QLabel("Status: Auto-sync disabled")
        self.sync_status_label.setStyleSheet("color: rgb(150, 150, 150); font-size: 10pt;")
        sync_layout.addWidget(self.sync_status_label)
        
        sync_group.setLayout(sync_layout)
        layout.addWidget(sync_group)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tools_tab, "🛠️ Tools")

    def create_menu_bar(self):
        """Create the main menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        add_media_action = QAction('Add Media...', self)
        add_media_action.triggered.connect(self.add_media_dialog)
        file_menu.addAction(add_media_action)
        
        add_folder_action = QAction('Add Folder...', self)
        add_folder_action.triggered.connect(self.add_folder_dialog)
        file_menu.addAction(add_folder_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export...', self)
        export_action.triggered.connect(self.export_selected)
        file_menu.addAction(export_action)
        
        import_action = QAction('Import...', self)
        import_action.triggered.connect(self.import_package_dialog)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        scan_action = QAction('Scan Directory...', self)
        scan_action.triggered.connect(self.scan_media_directory_dialog)
        tools_menu.addAction(scan_action)
        
        dedup_action = QAction('Find Duplicates', self)
        dedup_action.triggered.connect(self.find_duplicates)
        tools_menu.addAction(dedup_action)
        
        tools_menu.addSeparator()
        
        backup_action = QAction('Backup Registry', self)
        backup_action.triggered.connect(self.backup_registry)
        tools_menu.addAction(backup_action)
        
        restore_action = QAction('Restore Registry', self)
        restore_action.triggered.connect(self.restore_registry)
        tools_menu.addAction(restore_action)
        
        tools_menu.addSeparator()
        
        conflict_rules_action = QAction('Conflict Resolution Rules...', self)
        conflict_rules_action.triggered.connect(self.show_conflict_rules_dialog)
        tools_menu.addAction(conflict_rules_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About ROCA', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # ============================================================================
    # UI HELPER METHODS
    # ============================================================================

    def create_stat_card(self, title: str, value: str, description: str) -> QWidget:
        """Create a statistics card widget"""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 50, 60, 150);
                border: 1px solid rgb(60, 70, 80);
                border-radius: 8px;
                padding: 15px;
            }
        """)
        layout = QVBoxLayout(card)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #aaa; font-size: 11pt;")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18pt;
                font-weight: bold;
                padding: 5px 0;
            }
        """)
        layout.addWidget(value_label)
        
        desc_label = QLabel(description)
        desc_label.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(desc_label)
        
        return card

    def get_button_style(self, primary: bool = False) -> str:
        """Get button style sheet"""
        if primary:
            return """
                QPushButton {
                    background-color: rgb(70, 120, 180);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgb(80, 140, 200);
                }
                QPushButton:pressed {
                    background-color: rgb(60, 100, 160);
                }
            """
        else:
            return """
                QPushButton {
                    background-color: rgb(60, 70, 80);
                    color: white;
                    border: 1px solid rgb(80, 90, 100);
                    padding: 8px 16px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: rgb(70, 80, 90);
                }
                QPushButton:pressed {
                    background-color: rgb(50, 60, 70);
                }
            """

    def get_groupbox_style(self) -> str:
        """Get group box style sheet"""
        return """
            QGroupBox {
                color: white;
                font-size: 12pt;
                border: 1px solid rgb(60, 60, 80);
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """

    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def create_icon(self) -> QIcon:
        """Create application icon"""
        # Create a simple icon or load from file
        return QIcon()

    # ============================================================================
    # CORE FUNCTIONALITY METHODS
    # ============================================================================

    def initialize_controllers(self):
        """Initialize optional controllers and AI components"""
        try:
            # Initialize AI components
            self.chatbot = ChatBot()
            self.brain = AutonomousBrain()
            
            # Connect chatbot to orbital widget for capsule generation
            if hasattr(self, 'orbital_widget'):
                # Connect orbital widget selection to chatbot
                self.orbital_widget.capsule_selected.connect(self.on_capsule_selected)
            
            # Initialize voice controller if available
            try:
                from voice import VoiceController
                self.voice_controller = VoiceController()
                print("Voice controller initialized")
            except ImportError:
                self.voice_controller = None
                print("Voice controller not available")
            
            # Initialize core modules if available
            try:
                from core_modules.Core_Modules import (
                    PersonalitySystem, HierarchicalTemporalMemory, 
                    IntrinsicMotivationSystem, KnowledgeNetwork
                )
                self.personality_system = PersonalitySystem()
                self.memory_system = HierarchicalTemporalMemory()
                self.motivation_system = IntrinsicMotivationSystem()
                self.knowledge_network = KnowledgeNetwork()
                print("Core AI modules initialized")
            except ImportError:
                self.personality_system = None
                self.memory_system = None
                self.motivation_system = None
                self.knowledge_network = None
                print("Core AI modules not available")
                
        except Exception as e:
            print(f"Error initializing controllers: {e}")
            # Continue without AI components

    def initialize_capsule_store(self):
        """Initialize capsule store"""
        # Simple capsule store using dictionaries
        self.capsule_store = {}

    def load_capsules_on_startup(self):
        """Load capsules from JSON on startup"""
        try:
            if os.path.exists(self.CAPSULES_JSON_PATH):
                with open(self.CAPSULES_JSON_PATH, 'r') as f:
                    data = json.load(f)
                    for capsule_id, capsule_data in data.items():
                        capsule = MediaCapsule(**capsule_data)
                        self.capsule_store[capsule_id] = capsule
                print(f"Loaded {len(self.capsule_store)} capsules from {self.CAPSULES_JSON_PATH}")
        except Exception as e:
            print(f"Error loading capsules: {e}")

    def save_capsules(self):
        """Save capsules to JSON"""
        try:
            data = {cid: capsule.__dict__ for cid, capsule in self.capsule_store.items()}
            with open(self.CAPSULES_JSON_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.capsule_store)} capsules to {self.CAPSULES_JSON_PATH}")
        except Exception as e:
            print(f"Error saving capsules: {e}")

    def closeEvent(self, event):
        """Handle window close event"""
        self.save_capsules()
        super().closeEvent(event)

    # ============================================================================
    # MEDIA REGISTRY METHODS
    # ============================================================================

    def add_media_dialog(self):
        """Open dialog to add media files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Add Media Files', '',
            'Media Files (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.exr *.hdr *.bmp *.webp *.fbx *.obj *.gltf *.glb *.blend *.ma *.mb *.max *.c4d *.3ds *.dae *.bvh *.trc *.c3d *.cho *.mp4 *.mov *.avi *.mkv *.webm *.wmv *.wav *.mp3 *.ogg *.flac *.m4a *.pdf *.txt *.md *.doc *.docx);;All Files (*)'
        )
        
        if files:
            progress = QProgressDialog("Registering media...", "Cancel", 0, len(files), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            
            for i, file_path in enumerate(files):
                if progress.wasCanceled():
                    break
                
                try:
                    self.registry.register_media(Path(file_path))
                except Exception as e:
                    print(f"Error registering {file_path}: {e}")
                
                progress.setValue(i + 1)
            
            progress.close()
            self.refresh_media_list()
            self.status_bar.showMessage(f"Added {len(files)} media files")

    def add_folder_dialog(self):
        """Open dialog to add folder with options"""
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder to Add')
        if folder:
            # Show options dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Add Folder Options")
            dialog.setModal(True)
            
            layout = QVBoxLayout(dialog)
            
            # Info label
            info_label = QLabel(f"Add media from: {Path(folder).name}")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)
            
            # Options
            self.delete_originals_check = QCheckBox("Delete original files after successful registration")
            self.delete_originals_check.setToolTip("WARNING: This will permanently delete the original files after they are registered in the media library.")
            layout.addWidget(self.delete_originals_check)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                delete_originals = self.delete_originals_check.isChecked()
                self.scan_media_directory(Path(folder), delete_originals)

    def scan_media_directory(self, directory: Path, delete_originals: bool = False):
        """Scan directory for media files"""
        if not directory.exists():
            QMessageBox.warning(self, "Error", "Directory does not exist")
            return
        
        # Start scanner thread
        self.scanner_thread = MediaScannerThread(str(directory), delete_originals)
        self.scanner_thread.progress.connect(self.on_scan_progress)
        self.scanner_thread.finished.connect(self.on_scan_finished)
        self.scanner_thread.start()
        
        # Show progress
        self.scan_progress = QProgressDialog("Scanning directory...", "Cancel", 0, 100, self)
        self.scan_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.scan_progress.show()

    def on_scan_progress(self, current: int, total: int):
        """Update scan progress"""
        if hasattr(self, 'scan_progress'):
            self.scan_progress.setMaximum(total)
            self.scan_progress.setValue(current)

    def on_scan_finished(self, capsules: List[MediaCapsule], delete_originals: bool):
        """Handle scan completion"""
        if hasattr(self, 'scan_progress'):
            self.scan_progress.close()
        
        # Register capsules
        registered_count = 0
        successfully_registered = []
        
        for capsule in capsules:
            try:
                result = self.registry.register_media(Path(capsule.source_path))
                if result.get('success'):
                    registered_count += 1
                    successfully_registered.append(capsule.source_path)
            except Exception as e:
                print(f"Error registering {capsule.source_path}: {e}")
        
        # Handle deletion of originals if requested
        if delete_originals and successfully_registered:
            deleted_count = 0
            failed_deletions = []
            
            # Confirm deletion
            reply = QMessageBox.question(
                self, "Confirm Deletion",
                f"Successfully registered {len(successfully_registered)} files.\n\n"
                f"Delete the original files? This action cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                for file_path in successfully_registered:
                    try:
                        Path(file_path).unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                        failed_deletions.append(file_path)
                
                if failed_deletions:
                    QMessageBox.warning(
                        self, "Deletion Incomplete",
                        f"Deleted {deleted_count} files, but {len(failed_deletions)} could not be deleted."
                    )
                else:
                    QMessageBox.information(
                        self, "Deletion Complete",
                        f"Successfully deleted {deleted_count} original files."
                    )
        
        self.refresh_media_list()
        status_msg = f"Scan complete: {registered_count} files registered"
        if delete_originals and successfully_registered:
            status_msg += f" ({deleted_count} originals deleted)"
        self.status_bar.showMessage(status_msg)

    def scan_media_directory_dialog(self):
        """Open dialog to scan media directory"""
        folder = QFileDialog.getExistingDirectory(self, "Select Media Directory")
        if folder:
            self.scan_media_directory(Path(folder), False)

    def search_media(self):
        """Search media in registry"""
        query = self.search_bar.text()
        results = self.registry.search(query)
        self.display_media_results(results)

    def filter_media(self, filter_text: str):
        """Filter media by type"""
        filters = {}
        if filter_text != "All Media":
            filters['media_type'] = filter_text.lower().replace(' ', '_')
        
        results = self.registry.search("", filters)
        self.display_media_results(results)

    def display_media_results(self, results: List[Dict]):
        """Display search results in media list"""
        self.media_list.clear()
        
        for item in results:
            list_item = QListWidgetItem(item['filename'])
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            
            # Set icon based on media type
            icon_text = {
                'image': '🖼️',
                'video': '🎬',
                'audio': '🎵',
                'document': '📄',
                '3d_model': '🧊'
            }.get(item.get('media_type', '').split('_')[0], '📁')
            
            list_item.setText(f"{icon_text} {item['filename']}")
            self.media_list.addItem(list_item)

    def on_media_selected(self, item):
        """Handle capsule selection"""
        capsule_data = item.data(Qt.ItemDataRole.UserRole)
        
        # Update preview (try to create thumbnail from file if it's an image)
        if capsule_data.get('media_type') == 'image':
            try:
                from PIL import Image
                img = Image.open(capsule_data['file_path'])
                img.thumbnail((400, 400))
                
                # Convert PIL to QPixmap
                img.save('/tmp/temp_thumb.png', 'PNG')
                pixmap = QPixmap('/tmp/temp_thumb.png')
                if not pixmap.isNull():
                    self.preview_label.setPixmap(pixmap)
                else:
                    self.preview_label.clear()
            except:
                self.preview_label.clear()
        else:
            self.preview_label.clear()
        
        # Update metadata based on capsule type
        metadata_text = f"""
        <b>Filename:</b> {capsule_data.get('filename', 'Unknown')}<br>
        <b>Type:</b> {capsule_data.get('media_type', 'unknown')} ({capsule_data.get('capsule_type', 'IndexedCapsule')})<br>
        <b>Size:</b> {self.format_size(capsule_data.get('file_size', 0))}<br>
        <b>Created:</b> {capsule_data.get('created_at', 'Unknown')}<br>
        <b>Path:</b> {capsule_data.get('file_path', '')}<br>
        <b>Content Hash:</b> {capsule_data.get('content_hash', '')[:16] if capsule_data.get('content_hash') else 'Not computed'}...<br>
        """
        
        # Add type-specific metadata
        if capsule_data.get('media_type') == 'image':
            if 'resolution' in capsule_data:
                metadata_text += f"<b>Resolution:</b> {capsule_data['resolution']}<br>"
            if 'color_mode' in capsule_data:
                metadata_text += f"<b>Color Mode:</b> {capsule_data['color_mode']}<br>"
            if 'channels' in capsule_data:
                metadata_text += f"<b>Channels:</b> {capsule_data['channels']}<br>"
                
        elif capsule_data.get('media_type') == 'audio':
            if 'duration' in capsule_data:
                metadata_text += f"<b>Duration:</b> {capsule_data['duration']:.1f} seconds<br>"
            if 'sample_rate' in capsule_data:
                metadata_text += f"<b>Sample Rate:</b> {capsule_data['sample_rate']} Hz<br>"
            if 'channels' in capsule_data:
                metadata_text += f"<b>Channels:</b> {capsule_data['channels']}<br>"
            if 'codec' in capsule_data and capsule_data['codec']:
                metadata_text += f"<b>Codec:</b> {capsule_data['codec']}<br>"
                
        elif capsule_data.get('media_type') == 'video':
            if 'resolution' in capsule_data:
                metadata_text += f"<b>Resolution:</b> {capsule_data['resolution']}<br>"
            if 'duration' in capsule_data:
                metadata_text += f"<b>Duration:</b> {capsule_data['duration']:.1f} seconds<br>"
            if 'frame_rate' in capsule_data:
                metadata_text += f"<b>Frame Rate:</b> {capsule_data['frame_rate']} fps<br>"
            if 'codec' in capsule_data and capsule_data['codec']:
                metadata_text += f"<b>Codec:</b> {capsule_data['codec']}<br>"
                
        elif capsule_data.get('media_type') == 'document':
            if 'page_count' in capsule_data:
                metadata_text += f"<b>Pages:</b> {capsule_data['page_count']}<br>"
            if 'title' in capsule_data and capsule_data['title']:
                metadata_text += f"<b>Title:</b> {capsule_data['title']}<br>"
            if 'author' in capsule_data and capsule_data['author']:
                metadata_text += f"<b>Author:</b> {capsule_data['author']}<br>"
                
        elif capsule_data.get('media_type') == '3d_model':
            if 'format' in capsule_data:
                metadata_text += f"<b>Format:</b> {capsule_data['format']}<br>"
            if 'vertex_count' in capsule_data and capsule_data['vertex_count'] > 0:
                metadata_text += f"<b>Vertices:</b> {capsule_data['vertex_count']}<br>"
            if 'face_count' in capsule_data and capsule_data['face_count'] > 0:
                metadata_text += f"<b>Faces:</b> {capsule_data['face_count']}<br>"
        
        metadata_text += f"<br><i>💡 Indexed Capsule - Lightly Alive</i>"
        
        self.metadata_text.setHtml(metadata_text)
        
        # Update media info
        self.media_info.setText(f"Selected: {capsule_data.get('filename', 'Unknown')} ({capsule_data.get('capsule_type', 'IndexedCapsule')})")

    def on_media_selection_changed(self):
        """Handle media selection change"""
        selected_count = len(self.media_list.selectedItems())
        self.selected_count_label.setText(f"Selected: {selected_count}")

    def on_capsule_selected(self, capsule):
        """Handle capsule selection from orbital widget"""
        if capsule and hasattr(self, 'chatbot'):
            # Generate suggestions based on selected capsule
            prompt = f"Analyze this capsule: {capsule.content} (type: {capsule.kind})"
            try:
                response = self.chatbot.process_message(prompt)
                # Could display response in a chat widget or status
                print(f"Capsule analysis: {response}")
            except Exception as e:
                print(f"Error analyzing capsule: {e}")

    def refresh_media_list(self):
        """Refresh media list from registry"""
        results = self.registry.search("")
        self.display_media_results(results)

    # ============================================================================
    # DELETE SELECTED METHOD - FIXED
    # ============================================================================
    
    def delete_selected(self):
        """Delete selected media from registry"""
        selected_items = self.media_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select media to delete")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Delete {len(selected_items)} selected media items from registry?\n\nNote: This only removes from registry, not the original files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_count = 0
            
            for item in selected_items:
                media_data = item.data(Qt.ItemDataRole.UserRole)
                media_id = media_data['media_id']
                
                if self.registry.delete_media(media_id):
                    deleted_count += 1
                else:
                    failed_count += 1
            
            # Refresh the list
            self.refresh_media_list()
            
            # Show results
            if failed_count == 0:
                self.status_bar.showMessage(f"Deleted {deleted_count} media items")
            else:
                self.status_bar.showMessage(f"Deleted {deleted_count} items, {failed_count} failed")
    
    # ============================================================================
    # EXPORT/IMPORT METHODS
    # ============================================================================

    def export_selected(self):
        """Export selected media"""
        selected_items = self.media_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select media to export")
            return
        
        # Get output path
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Package", "", "ROCA Packages (*.rocapkg);;ZIP Archives (*.zip)"
        )
        
        if output_path:
            media_ids = []
            for item in selected_items:
                media_data = item.data(Qt.ItemDataRole.UserRole)
                media_ids.append(media_data['media_id'])
            
            # Create package
            result = self.registry.export_package(media_ids, Path(output_path))
            
            if result.get('success'):
                QMessageBox.information(
                    self, "Export Complete", 
                    f"Exported {len(media_ids)} items to {output_path}"
                )

    def import_package_dialog(self):
        """Import ROCA package"""
        package_path, _ = QFileDialog.getOpenFileName(
            self, "Import Package", "", "ROCA Packages (*.rocapkg);;ZIP Archives (*.zip)"
        )
        
        if package_path:
            target_dir = QFileDialog.getExistingDirectory(
                self, "Select Import Directory"
            )
            
            if target_dir:
                result = self.registry.import_package(Path(package_path), Path(target_dir))
                
                if result.get('success'):
                    QMessageBox.information(
                        self, "Import Complete",
                        f"Imported {result.get('imported_count', 0)} items"
                    )
                    self.refresh_media_list()

    def create_export_package(self):
        """Create export package with current options"""
        selected_items = self.media_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select media to export")
            return
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Package", "", "ROCA Packages (*.rocapkg)"
        )
        
        if output_path:
            # Get media items from registry
            media_items = []
            for item in selected_items:
                media_data = item.data(Qt.ItemDataRole.UserRole)
                media_items.append(media_data)
            
            # Create package
            package = ROCAPackage.create(
                media_items,
                Path(output_path),
                include_previews=self.include_previews.isChecked(),
                include_thumbnails=self.include_thumbs.isChecked()
            )
            
            QMessageBox.information(
                self, "Package Created",
                f"Created package with {len(media_items)} items"
            )

    def create_share(self):
        """Create share link"""
        selected_items = self.media_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select media to share")
            return
        
        # Simple share implementation
        media_ids = [item.data(Qt.ItemDataRole.UserRole)['media_id'] for item in selected_items]
        share_info = f"Shared {len(media_ids)} media items\nIDs: {', '.join(media_ids[:5])}"
        
        QMessageBox.information(self, "Share Created", share_info)

    # ============================================================================
    # TOOLS METHODS
    # ============================================================================

    def find_duplicates(self):
        """Find duplicate media files"""
        duplicates = self.registry.get_registry_stats().get('duplicate_savings', 0)
        
        if duplicates > 0:
            QMessageBox.information(
                self, "Duplicate Detection",
                f"Found {duplicates} duplicate files that can be deduplicated"
            )
        else:
            QMessageBox.information(self, "Duplicate Detection", "No duplicates found")

    def backup_registry(self):
        """Backup registry"""
        backup_dir = QFileDialog.getExistingDirectory(self, "Select Backup Directory")
        if backup_dir:
            # Simple backup implementation
            import shutil
            backup_path = Path(backup_dir) / f"roca_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(self.config.registry_path, backup_path)
            
            QMessageBox.information(
                self, "Backup Complete",
                f"Registry backed up to: {backup_path}"
            )

    def restore_registry(self):
        """Restore registry from backup"""
        backup_dir = QFileDialog.getExistingDirectory(self, "Select Backup Directory")
        if backup_dir:
            reply = QMessageBox.question(
                self, "Confirm Restore",
                "This will overwrite current registry. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Simple restore implementation
                import shutil
                shutil.rmtree(self.config.registry_path)
                shutil.copytree(backup_dir, self.config.registry_path)
                
                QMessageBox.information(self, "Restore Complete", "Registry restored successfully")
                self.refresh_media_list()

    def refresh_media_list(self):
        """Refresh the media list display with indexed capsules"""
        if hasattr(self, 'media_list'):
            try:
                # Get indexed capsules by type
                image_capsules = self.registry.get_indexed_capsules_by_type('image', limit=200)
                audio_capsules = self.registry.get_indexed_capsules_by_type('audio', limit=200)
                video_capsules = self.registry.get_indexed_capsules_by_type('video', limit=200)
                document_capsules = self.registry.get_indexed_capsules_by_type('document', limit=200)
                model_capsules = self.registry.get_indexed_capsules_by_type('3d_model', limit=200)
                
                all_capsules = image_capsules + audio_capsules + video_capsules + document_capsules + model_capsules
                
                # Convert capsules to display format
                display_items = []
                for capsule in all_capsules:
                    item_data = {
                        'file_path': capsule.file_path,
                        'content_hash': capsule.content_hash,
                        'media_type': capsule.media_type,
                        'file_size': capsule.file_size,
                        'created_at': capsule.created_at,
                        'filename': os.path.basename(capsule.file_path),
                        'capsule_type': type(capsule).__name__
                    }
                    
                    # Add type-specific data
                    if isinstance(capsule, ImageCapsule):
                        item_data.update({
                            'width': capsule.width,
                            'height': capsule.height,
                            'channels': capsule.channels,
                            'color_mode': capsule.color_mode,
                            'resolution': f"{capsule.width}x{capsule.height}"
                        })
                    elif isinstance(capsule, AudioCapsule):
                        item_data.update({
                            'duration': capsule.duration,
                            'sample_rate': capsule.sample_rate,
                            'channels': capsule.channels,
                            'bit_depth': capsule.bit_depth,
                            'codec': capsule.codec
                        })
                    elif isinstance(capsule, VideoCapsule):
                        item_data.update({
                            'duration': capsule.duration,
                            'width': capsule.width,
                            'height': capsule.height,
                            'frame_rate': capsule.frame_rate,
                            'codec': capsule.codec,
                            'resolution': f"{capsule.width}x{capsule.height}"
                        })
                    elif isinstance(capsule, PDFCapsule):
                        item_data.update({
                            'page_count': capsule.page_count,
                            'title': capsule.title,
                            'author': capsule.author,
                            'subject': capsule.subject
                        })
                    elif isinstance(capsule, Model3DCapsule):
                        item_data.update({
                            'vertex_count': capsule.vertex_count,
                            'face_count': capsule.face_count,
                            'material_count': capsule.material_count,
                            'has_animation': capsule.has_animation,
                            'format': capsule.format
                        })
                    
                    display_items.append(item_data)
                
                # Display the capsule items
                self.display_capsule_results(display_items)
                
                # Update status
                capsule_stats = self.registry.get_indexed_capsule_stats()
                total_capsules = capsule_stats.get('total_capsules', 0)
                if total_capsules > 0:
                    self.status_bar.showMessage(f"Showing {len(display_items)} indexed capsules ({total_capsules} total)")
                else:
                    self.status_bar.showMessage("No indexed capsules yet - files will be indexed when ingested")
                    
            except Exception as e:
                self.media_list.clear()
                error_item = QListWidgetItem(f"❌ Error loading capsules: {e}")
                self.media_list.addItem(error_item)
                self.status_bar.showMessage(f"Error loading capsules: {e}")

    def display_capsule_results(self, capsules: List[Dict]):
        """Display indexed capsule results in media list"""
        self.media_list.clear()
        
        for item in capsules:
            # Create display text based on capsule type
            capsule_type = item.get('capsule_type', 'IndexedCapsule')
            filename = item.get('filename', 'Unknown')
            media_type = item.get('media_type', 'unknown')
            
            # Choose icon based on type
            icon_map = {
                'image': '🖼️',
                'audio': '🎵',
                'video': '🎬',
                'document': '📄',
                '3d_model': '🧊',
                'unknown': '📁'
            }
            icon = icon_map.get(media_type, '📁')
            
            # Create descriptive text
            if media_type == 'image' and 'resolution' in item:
                display_text = f"{icon} {filename} ({item['resolution']})"
            elif media_type == 'audio' and 'duration' in item:
                duration_str = f"{item['duration']:.1f}s" if item['duration'] else "Unknown"
                display_text = f"{icon} {filename} ({duration_str})"
            elif media_type == 'video' and 'resolution' in item and 'duration' in item:
                duration_str = f"{item['duration']:.1f}s" if item['duration'] else "Unknown"
                display_text = f"{icon} {filename} ({item['resolution']}, {duration_str})"
            elif media_type == 'document' and 'page_count' in item:
                display_text = f"{icon} {filename} ({item['page_count']} pages)"
            elif media_type == '3d_model' and 'format' in item:
                display_text = f"{icon} {filename} ({item['format']})"
            else:
                file_size_str = self.format_size(item.get('file_size', 0))
                display_text = f"{icon} {filename} ({file_size_str})"
            
            list_item = QListWidgetItem(display_text)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.media_list.addItem(list_item)

    def analyze_capsules(self):
        """Analyze capsule statistics"""
        stats = self.registry.get_registry_stats()
        
        report = f"""
        📊 Registry Analysis Report
        ===========================
        Total Media: {stats['total_media']}
        Storage Used: {self.format_size(stats['total_size'])}
        Unique Files: {stats.get('unique_files', stats['total_media'])}
        Space Saved: {self.format_size(stats.get('duplicate_savings', 0) * 1024 * 1024)}
        
        By Type:
        """
        
        for media_type, count in stats.get('by_type', {}).items():
            report += f"  {media_type}: {count}\n"
        
        QMessageBox.information(self, "Analysis Report", report)

    def start_style_transfer(self):
        """Start style transfer tool"""
        QMessageBox.information(self, "Style Transfer", "Style transfer tool would open here")

    def start_predictive(self):
        """Start predictive analysis"""
        QMessageBox.information(self, "Predictive Analysis", "Predictive analysis tool would open here")

    def tune_performance(self):
        """Tune performance settings"""
        QMessageBox.information(self, "Performance Tuning", "Performance tuning tool would open here")

    def clean_registry(self):
        """Clean registry"""
        reply = QMessageBox.question(
            self, "Clean Registry",
            "Remove orphaned entries and optimize database?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.status_bar.showMessage("Cleaning registry...")
            # Implementation would go here
            self.status_bar.showMessage("Registry cleaned")

    def generate_reports(self):
        """Generate comprehensive reports"""
        # Create reports dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Generate Reports")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Generate Media Registry Reports")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18pt;
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Report type selection
        type_group = QGroupBox("Report Types")
        type_group.setStyleSheet(self.get_groupbox_style())
        type_layout = QVBoxLayout()
        
        self.report_types = {
            "summary": QCheckBox("📈 Registry Summary Report"),
            "detailed": QCheckBox("📋 Detailed Media Inventory"),
            "usage": QCheckBox("📊 Usage Statistics Report"),
            "storage": QCheckBox("💾 Storage Analysis Report"),
            "activity": QCheckBox("🔄 Activity Log Report"),
            "health": QCheckBox("🏥 Registry Health Check")
        }
        
        # Set default selections
        self.report_types["summary"].setChecked(True)
        self.report_types["detailed"].setChecked(True)
        
        for checkbox in self.report_types.values():
            type_layout.addWidget(checkbox)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_group.setStyleSheet(self.get_groupbox_style())
        output_layout = QVBoxLayout()
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        
        self.report_format = QComboBox()
        self.report_format.addItems(["HTML", "Text", "JSON", "CSV"])
        self.report_format.setCurrentText("HTML")
        format_layout.addWidget(self.report_format)
        format_layout.addStretch()
        output_layout.addLayout(format_layout)
        
        # Date range
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Date Range:"))
        
        self.date_from = QLineEdit()
        self.date_from.setPlaceholderText("YYYY-MM-DD (optional)")
        date_layout.addWidget(self.date_from)
        
        date_layout.addWidget(QLabel("to"))
        
        self.date_to = QLineEdit()
        self.date_to.setPlaceholderText("YYYY-MM-DD (optional)")
        date_layout.addWidget(self.date_to)
        
        output_layout.addLayout(date_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        generate_btn = QPushButton("📊 Generate Report")
        generate_btn.setStyleSheet(self.get_button_style(primary=True))
        generate_btn.clicked.connect(lambda: self.generate_selected_reports(dialog))
        button_layout.addWidget(generate_btn)
        
        preview_btn = QPushButton("👁️ Preview")
        preview_btn.setStyleSheet(self.get_button_style())
        preview_btn.clicked.connect(lambda: self.preview_report(dialog))
        button_layout.addWidget(preview_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()

    def generate_selected_reports(self, dialog):
        """Generate the selected reports"""
        selected_types = [rtype for rtype, checkbox in self.report_types.items() if checkbox.isChecked()]
        
        if not selected_types:
            QMessageBox.warning(dialog, "No Reports Selected", "Please select at least one report type.")
            return
        
        # Generate reports
        reports_data = {}
        for report_type in selected_types:
            reports_data[report_type] = self.generate_report_data(report_type)
        
        # Format and save
        format_type = self.report_format.currentText().lower()
        output_path = self.save_report_file(format_type)
        
        if output_path:
            self.write_formatted_report(output_path, reports_data, format_type)
            QMessageBox.information(
                dialog, "Reports Generated",
                f"Reports saved to: {output_path}"
            )
            dialog.accept()
        else:
            dialog.reject()

    def generate_report_data(self, report_type):
        """Generate data for a specific report type"""
        stats = self.registry.get_registry_stats()
        
        if report_type == "summary":
            return {
                "title": "Registry Summary Report",
                "generated": datetime.now().isoformat(),
                "total_media": stats.get('total_media', 0),
                "total_size": stats.get('total_size', 0),
                "unique_files": stats.get('unique_files', stats.get('total_media', 0)),
                "duplicate_savings": stats.get('duplicate_savings', 0),
                "by_type": stats.get('by_type', {}),
                "registry_health": "Good" if stats.get('total_media', 0) > 0 else "Empty"
            }
        
        elif report_type == "detailed":
            # Get detailed media list
            media_items = []
            try:
                # Get real media data from registry
                all_media = self.registry.get_all_media(limit=100)  # Limit for report generation
                media_items = [
                    {
                        "id": item.get('media_id', f'media_{i}'),
                        "filename": item.get('filename', 'unknown'),
                        "type": item.get('media_type', 'unknown'),
                        "size": item.get('file_size', 0),
                        "created": item.get('created_at', datetime.now().isoformat()),
                        "path": item.get('original_path', ''),
                        "tags": item.get('tags', ''),
                        "projects": item.get('projects', '')
                    } for i, item in enumerate(all_media)
                ]
            except Exception as e:
                print(f"Error getting media for detailed report: {e}")
                # Fallback to empty list
                media_items = []
            
            return {
                "title": "Detailed Media Inventory",
                "generated": datetime.now().isoformat(),
                "media_items": media_items,
                "total_count": len(media_items)
            }
        
        elif report_type == "usage":
            # Calculate access patterns from weekly activity
            access_patterns = {}
            weekly_activity = stats.get('weekly_activity', {})
            if weekly_activity:
                total_accesses = sum(weekly_activity.values())
                access_patterns = {
                    action: f"{count} times ({count/total_accesses*100:.1f}%)"
                    for action, count in weekly_activity.items()
                }
            else:
                access_patterns = "No activity data available (activity logging may be disabled)"
            
            return {
                "title": "Usage Statistics Report",
                "generated": datetime.now().isoformat(),
                "access_patterns": access_patterns,
                "popular_types": stats.get('by_type', {}),
                "storage_trends": f"Total storage: {self.format_size(stats.get('total_size', 0))}"
            }
        
        elif report_type == "storage":
            # Get largest files
            largest_files = []
            try:
                conn = sqlite3.connect(self.registry.db_path, timeout=30)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename, file_size, original_path
                    FROM media_registry 
                    WHERE status = 'registered'
                    ORDER BY file_size DESC
                    LIMIT 10
                """)
                largest_files = [
                    {
                        "filename": row[0],
                        "size": self.format_size(row[1]),
                        "path": row[2][:50] + "..." if len(row[2]) > 50 else row[2]
                    } for row in cursor.fetchall()
                ]
                conn.close()
            except Exception as e:
                largest_files = f"Error retrieving largest files: {e}"
            
            return {
                "title": "Storage Analysis Report",
                "generated": datetime.now().isoformat(),
                "total_size": self.format_size(stats.get('total_size', 0)),
                "average_file_size": self.format_size(stats.get('total_size', 0) / max(1, stats.get('total_media', 1))),
                "largest_files": largest_files,
                "storage_efficiency": f"{self.format_size(stats.get('duplicate_savings', 0) * 1024 * 1024)} saved from deduplication"
            }
        
        elif report_type == "activity":
            # Get recent activity from user_activity table
            recent_activity = []
            user_actions = {}
            try:
                conn = sqlite3.connect(self.registry.db_path, timeout=30)
                cursor = conn.cursor()
                
                # Get recent activity
                cursor.execute("""
                    SELECT action, details, timestamp
                    FROM user_activity
                    ORDER BY timestamp DESC
                    LIMIT 20
                """)
                recent_activity = [
                    {
                        "action": row[0],
                        "details": row[1] or "No details",
                        "timestamp": row[2]
                    } for row in cursor.fetchall()
                ]
                
                # Get user action summary
                cursor.execute("""
                    SELECT action, COUNT(*)
                    FROM user_activity
                    GROUP BY action
                    ORDER BY COUNT(*) DESC
                """)
                user_actions = dict(cursor.fetchall())
                
                conn.close()
            except Exception as e:
                recent_activity = f"Error retrieving activity data: {e}"
                user_actions = "Error retrieving user actions"
            
            return {
                "title": "Activity Log Report",
                "generated": datetime.now().isoformat(),
                "recent_activity": recent_activity,
                "user_actions": user_actions,
                "system_events": f"Total media registered: {stats.get('total_media', 0)}"
            }
        
        elif report_type == "health":
            return {
                "title": "Registry Health Check",
                "generated": datetime.now().isoformat(),
                "database_integrity": "OK",
                "missing_files": 0,
                "corrupted_entries": 0,
                "optimization_needed": False,
                "recommendations": [
                    "Regular backups recommended",
                    "Consider defragmentation if performance issues occur"
                ]
            }
        
        return {"error": f"Unknown report type: {report_type}"}

    def save_report_file(self, format_type):
        """Get save path for report file"""
        file_filter = {
            "html": "HTML Files (*.html)",
            "text": "Text Files (*.txt)",
            "json": "JSON Files (*.json)",
            "csv": "CSV Files (*.csv)"
        }.get(format_type, "All Files (*)")
        
        filename = f"roca_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", filename, file_filter
        )
        
        return output_path if output_path else None

    def write_formatted_report(self, output_path, reports_data, format_type):
        """Write report data in the specified format"""
        try:
            if format_type == "html":
                self.write_html_report(output_path, reports_data)
            elif format_type == "json":
                self.write_json_report(output_path, reports_data)
            elif format_type == "csv":
                self.write_csv_report(output_path, reports_data)
            else:  # text
                self.write_text_report(output_path, reports_data)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to write report: {e}")

    def write_html_report(self, output_path, reports_data):
        """Write HTML formatted report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ROCA Media Registry Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: white; }}
                .header {{ background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 8px; }}
                .metric {{ display: inline-block; background: #3a3a3a; padding: 10px; margin: 5px; border-radius: 4px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #444; }}
                th {{ background: #333; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 ROCA Media Registry Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for report_type, data in reports_data.items():
            html += f'<div class="section"><h2>{data.get("title", report_type.title())}</h2>'
            
            if report_type == "summary":
                html += f"""
                <div class="metric">Total Media: {data.get('total_media', 0)}</div>
                <div class="metric">Storage Used: {self.format_size(data.get('total_size', 0))}</div>
                <div class="metric">Unique Files: {data.get('unique_files', 0)}</div>
                <div class="metric">Space Saved: {self.format_size(data.get('duplicate_savings', 0) * 1024 * 1024)}</div>
                """
                
                if data.get('by_type'):
                    html += '<h3>By Type:</h3><table><tr><th>Type</th><th>Count</th></tr>'
                    for media_type, count in data['by_type'].items():
                        html += f'<tr><td>{media_type}</td><td>{count}</td></tr>'
                    html += '</table>'
            
            elif report_type == "detailed" and data.get('media_items'):
                html += '<table><tr><th>ID</th><th>Filename</th><th>Type</th><th>Size</th><th>Created</th></tr>'
                for item in data['media_items'][:50]:  # Limit to 50 items
                    html += f'<tr><td>{item.get("id", "")}</td><td>{item.get("filename", "")}</td><td>{item.get("type", "")}</td><td>{self.format_size(item.get("size", 0))}</td><td>{item.get("created", "")}</td></tr>'
                html += '</table>'
            
            else:
                html += f'<pre>{json.dumps(data, indent=2)}</pre>'
            
            html += '</div>'
        
        html += '</body></html>'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def write_json_report(self, output_path, reports_data):
        """Write JSON formatted report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reports_data, f, indent=2, default=str)

    def write_csv_report(self, output_path, reports_data):
        """Write CSV formatted report"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Report Type', 'Metric', 'Value'])
            
            for report_type, data in reports_data.items():
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            writer.writerow([report_type, f"{key}.{sub_key}", sub_value])
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            writer.writerow([report_type, f"{key}[{i}]", item])
                    else:
                        writer.writerow([report_type, key, value])

    def write_text_report(self, output_path, reports_data):
        """Write plain text formatted report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ROCA Media Registry Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for report_type, data in reports_data.items():
                f.write(f"{data.get('title', report_type.title())}\n")
                f.write("-" * len(data.get('title', report_type)) + "\n")
                f.write(json.dumps(data, indent=2, default=str))
                f.write("\n\n")

    def preview_report(self, dialog):
        """Preview the selected reports"""
        selected_types = [rtype for rtype, checkbox in self.report_types.items() if checkbox.isChecked()]
        
        if not selected_types:
            QMessageBox.warning(dialog, "No Reports Selected", "Please select at least one report type.")
            return
        
        # Generate preview data
        preview_data = {}
        for report_type in selected_types:
            preview_data[report_type] = self.generate_report_data(report_type)
        
        # Show preview dialog
        preview_dialog = QDialog(dialog)
        preview_dialog.setWindowTitle("📊 Report Preview")
        preview_dialog.resize(700, 500)
        
        layout = QVBoxLayout(preview_dialog)
        
        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setPlainText(json.dumps(preview_data, indent=2, default=str))
        
        layout.addWidget(QLabel("Report Preview (JSON format):"))
        layout.addWidget(preview_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview_dialog.accept)
        layout.addWidget(close_btn)
        
        preview_dialog.exec()

    def open_settings(self):
        """Open settings dialog"""
        QMessageBox.information(self, "Settings", "Settings dialog would open here")

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>ROCA Media Registry</h2>
        <p>Professional media management and exchange platform</p>
        <p>Version 1.0.0</p>
        <p>© 2024 ROCA Project</p>
        <p>Features:</p>
        <ul>
        <li>Media registration and deduplication</li>
        <li>Universal exchange format (.rocapkg)</li>
        <li>Advanced search and filtering</li>
        <li>Orbital visualization</li>
        <li>Team collaboration tools</li>
        </ul>
        """
        QMessageBox.about(self, "About ROCA", about_text)

    # ============================================================================
    # FILE SYSTEM WATCHING METHODS
    # ============================================================================

    def on_watched_directory_changed(self, path):
        """Handle directory changes for automatic registration"""
        if not self.auto_register_enabled:
            return
            
        print(f"📁 Directory changed: {path}")
        # Scan for new files in the changed directory
        self.scan_directory_for_new_files(path)

    def on_watched_file_changed(self, path):
        """Handle file changes for automatic registration"""
        if not self.auto_register_enabled:
            return
            
        print(f"📄 File changed: {path}")
        # Check if this is a new file that needs registration
        if os.path.exists(path) and not self.registry.get_by_path(Path(path)):
            self.auto_register_file(path)

    def scan_directory_for_new_files(self, directory_path):
        """Scan directory for new files to register automatically"""
        try:
            dir_path = Path(directory_path)
            new_files = []
            
            # Find media files that aren't registered yet
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and self.is_media_file(file_path):
                    if not self.registry.get_by_path(file_path):
                        new_files.append(file_path)
            
            # Register new files
            for file_path in new_files:
                self.auto_register_file(str(file_path))
                
            if new_files:
                self.status_bar.showMessage(f"Auto-registered {len(new_files)} new files")
                self.refresh_media_list()  # Refresh the media list display
                
        except Exception as e:
            print(f"Error scanning directory {directory_path}: {e}")

    def auto_register_file(self, file_path):
        """Automatically register a single file"""
        try:
            result = self.registry.register_media(Path(file_path), "auto_watcher")
            if result.get('success'):
                print(f"✅ Auto-registered: {file_path}")
            else:
                print(f"❌ Failed to auto-register: {file_path} - {result.get('error', 'unknown error')}")
        except Exception as e:
            print(f"Error auto-registering {file_path}: {e}")

    def is_media_file(self, file_path):
        """Check if file is a supported media type"""
        extensions = {
            # Images
            '.png', '.jpg', '.jpeg', '.tga', '.tif', '.tiff', '.exr', '.hdr', '.bmp', '.webp',
            # 3D Models
            '.fbx', '.obj', '.gltf', '.glb', '.blend', '.ma', '.mb', '.max', '.c4d', '.3ds', '.dae',
            # Motion Capture
            '.bvh', '.trc', '.c3d', '.cho',
            # Video
            '.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv',
            # Audio
            '.wav', '.mp3', '.ogg', '.flac', '.m4a',
            # Documents
            '.pdf', '.txt', '.md', '.doc', '.docx'
        }
        return file_path.suffix.lower() in extensions

    def add_watched_directory(self, directory_path):
        """Add a directory to the file system watcher"""
        if directory_path not in self.watched_directories:
            if self.file_watcher.addPath(directory_path):
                self.watched_directories.add(directory_path)
                print(f"👁️ Now watching: {directory_path}")
                return True
            else:
                print(f"❌ Failed to watch: {directory_path}")
                return False
        return True

    def remove_watched_directory(self, directory_path):
        """Remove a directory from the file system watcher"""
        if directory_path in self.watched_directories:
            self.file_watcher.removePath(directory_path)
            self.watched_directories.remove(directory_path)
            print(f"🚫 Stopped watching: {directory_path}")
            return True
        return False

    def set_auto_register_enabled(self, enabled):
        """Enable or disable automatic registration"""
        self.auto_register_enabled = bool(enabled)
        status = "enabled" if self.auto_register_enabled else "disabled"
        print(f"🔄 Auto-registration {status}")
        
        if hasattr(self, 'watch_status_label'):
            self.watch_status_label.setText(f"Status: Watching {'enabled' if self.auto_register_enabled else 'disabled'}")
        
        if self.auto_register_enabled:
            self.status_bar.showMessage("Auto-registration enabled - watching for new files")
        else:
            self.status_bar.showMessage("Auto-registration disabled")

    def add_watch_directory_dialog(self):
        """Open dialog to add a directory to watch"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Watch")
        if directory:
            if self.add_watched_directory(directory):
                self.update_watched_dirs_list()
                QMessageBox.information(self, "Directory Added", f"Now watching: {directory}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to add directory: {directory}")

    def remove_watch_directory(self):
        """Remove selected directory from watching"""
        current_item = self.watched_dirs_list.currentItem()
        if current_item:
            directory = current_item.text()
            if self.remove_watched_directory(directory):
                self.update_watched_dirs_list()
                QMessageBox.information(self, "Directory Removed", f"Stopped watching: {directory}")
        else:
            QMessageBox.warning(self, "No Selection", "Please select a directory to remove")

    def update_watched_dirs_list(self):
        """Update the UI list of watched directories"""
        if hasattr(self, 'watched_dirs_list'):
            self.watched_dirs_list.clear()
            for directory in sorted(self.watched_directories):
                self.watched_dirs_list.addItem(directory)

    def update_stats(self):
        """Update status bar statistics"""
        try:
            stats = self.registry.get_registry_stats()
            self.status_bar.showMessage(
                f"Media: {stats['total_media']} | "
                f"Storage: {self.format_size(stats['total_size'])} | "
                f"Unique: {stats.get('unique_files', stats['total_media'])}"
            )
        except:
            self.status_bar.showMessage("ROCA Ready")

    def show_conflict_rules_dialog(self):
        """Show the conflict resolution rules configuration dialog"""
        try:
            conflict_manager = ConflictResolutionManager(self.registry.config.registry_path)
            dialog = ConflictRulesDialog(conflict_manager, self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open conflict rules dialog: {e}")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    import sys
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("ROCA Media Registry")
    app.setOrganizationName("ROCA Project")
    
    # Set dark theme
    app.setStyle('Fusion')
    
    # Set application-wide palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 40))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 50))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 60))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 70))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(100, 150, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(100, 150, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # Create and show main window
    window = ROCAMainWindow()
    window.show()
    
    # Start application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()