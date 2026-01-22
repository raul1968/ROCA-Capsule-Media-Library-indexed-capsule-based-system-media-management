def media_capsule_to_numpy_capsule(media_capsule):
    """Convert a MediaCapsule object to a NumpyCapsule object for visualization."""
    import numpy as np
    from datetime import datetime
    # Defensive: handle missing fields gracefully
    content = getattr(media_capsule, 'filename', None) or getattr(media_capsule, 'source_path', None) or getattr(media_capsule, 'id', 'Unknown')
    kind = getattr(media_capsule, 'media_type', None)
    if kind is not None:
        kind = str(kind).split('.')[-1].lower()
    else:
        kind = 'concept'
    certainty = np.float32(getattr(media_capsule, 'complexity', 0.6))
    orbit_radius = np.float32(np.random.uniform(1.0, 2.5))
    angle = np.float32(np.random.uniform(0, 2 * np.pi))
    capsule_id = getattr(media_capsule, 'id', None) or str(np.random.randint(100000, 999999))
    created_at = None
    try:
        created_at = float(getattr(media_capsule, 'created', datetime.now().timestamp()))
    except Exception:
        created_at = datetime.now().timestamp()
    # Numpy arrays for physics
    position = np.zeros(2, dtype=np.float32)
    velocity = np.zeros(2, dtype=np.float32)
    acceleration = np.zeros(2, dtype=np.float32)
    # Visualization properties
    color = np.array([0.7, 0.7, 1.0])
    size = np.float32(16.0)
    metadata = {}
    connections = getattr(media_capsule, 'related_capsules', [])
    # Compose NumpyCapsule
    return NumpyCapsule(
        content=content,
        kind=kind,
        certainty=certainty,
        orbit_radius=orbit_radius,
        angle=angle,
        character=None,
        id=capsule_id,
        created_at=created_at,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        color=color,
        size=size,
        metadata=metadata,
        connections=connections
    )
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

    def update_capsule_visualization(self):
        """Use LOD to reduce rendering load based on capsule count."""
        if len(self.capsules) > 500:
            self.draw_simplified_capsules()
        elif len(self.capsules) > 200:
            self.draw_medium_detail_capsules()
        else:
            self.draw_full_detail_capsules()

    def draw_simplified_capsules(self):
        # Draw only basic dots or bounding circles for distant capsules
        # (Stub: implement as needed)
        pass

    def draw_medium_detail_capsules(self):
        # Draw medium detail (no text, simple shapes)
        # (Stub: implement as needed)
        pass

    def draw_full_detail_capsules(self):
        # Draw full detail (gradient, text, etc.)
        # (Stub: implement as needed)
        pass

    capsule_selected = pyqtSignal(object)
    capsule_hovered = pyqtSignal(object)
    capsules_changed = pyqtSignal()

    @property
    def capsules(self):
        return self._capsules

    @capsules.setter
    def capsules(self, value):
        self._capsules = value
        self.capsules_changed.emit()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._capsules: List[NumpyCapsule] = []
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
        self.capsules_changed.emit()
        return capsule
    
    def add_capsule_object(self, capsule: NumpyCapsule):
        """Add an existing capsule object to the orbital display"""
        self.capsules.append(capsule)
        self.capsules_changed.emit()
    
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
        
        # Extract positions and velocities
        positions = np.array([c.position for c in self.capsules], dtype=np.float32)
        velocities = np.array([c.velocity for c in self.capsules], dtype=np.float32)
        accelerations = np.zeros((n, 2), dtype=np.float32)
        
        # Calculate forces between all pairs (optimized)
        for i in range(n):
            for j in range(i + 1, n):
                # Vector from i to j
                dx = positions[j] - positions[i]
                dist_sq = np.dot(dx, dx) + 0.001  # Add small value to avoid division by zero
                dist = np.sqrt(dist_sq)
                
                # Normalized direction
                dir_vec = dx / dist
                
                # Force magnitude (attractive when far, repulsive when close)
                if dist > 100:
                    force = self.attraction_strength / dist_sq
                else:
                    force = -self.repulsion_strength / dist_sq
                
                # Apply forces
                accelerations[i] += dir_vec * force
                accelerations[j] -= dir_vec * force
        
        # Update velocities and positions
        for i, capsule in enumerate(self.capsules):
            capsule.velocity += accelerations[i] * dt
            capsule.velocity *= self.damping  # Damping
            capsule.position += capsule.velocity * dt
    
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

        # Level-of-detail rendering for capsules
        self.update_capsule_visualization()
        
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
        
        # Enhanced controls
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
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        analyze_action = QAction("Analyze Capsules", self)
        analyze_action.triggered.connect(self.analyze_capsules)
        tools_menu.addAction(analyze_action)
        
        # Add NumPy info menu
        numpy_menu = menubar.addMenu("NumPy")
        
        info_action = QAction("NumPy Info", self)
        info_action.triggered.connect(self.show_numpy_info)
        numpy_menu.addAction(info_action)
    
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
            return

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
            self.orbital_widget.capsules_changed.emit()
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