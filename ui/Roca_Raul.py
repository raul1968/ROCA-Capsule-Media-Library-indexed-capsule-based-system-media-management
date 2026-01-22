import pygame
import sys
import os
import json
import re
import uuid
import time
import math
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import threading
import queue
from dataclasses import dataclass
from enum import Enum
import wave
import pyaudio
import speech_recognition as sr
from gtts import gTTS
import tempfile
import subprocess
import platform

# PDF processing imports (optional)
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import the AutonomousBrain
sys.path.append(os.path.join(os.path.dirname(__file__), 'Brain'))
try:
    from Brain.autonomous_brain import AutonomousBrain, ThoughtType
    from Brain.paper_ingester import PaperIngester
    from Brain.graph_manager import GraphManager, ROCAGraph
    BRAIN_AVAILABLE = True
except ImportError:
    print("⚠️ Brain modules not available, running in limited mode")
    BRAIN_AVAILABLE = False

# Load configurations
def load_json_config(path, default=None):
    """Safely load JSON configuration file."""
    if default is None:
        default = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load {path}: {e}")
        return default

SEMANTIC_DEFINITIONS_PATH = os.path.join(os.path.dirname(__file__), 'Json', 'semantic.json')
semantic_definitions = load_json_config(SEMANTIC_DEFINITIONS_PATH, {})

INGEST_DEFINITIONS_PATH = os.path.join(os.path.dirname(__file__), 'Json', 'ingest.json')
ingest_definitions = load_json_config(INGEST_DEFINITIONS_PATH, {})

UTILITY_DEFINITIONS_PATH = os.path.join(os.path.dirname(__file__), 'Json', 'utility.json')
utility_definitions = load_json_config(UTILITY_DEFINITIONS_PATH, {})

# Pygame initialization
pygame.init()
pygame.mixer.init()  # For audio playback

# ============================================================================
# SPEECH AND AUDIO SYSTEM
# ============================================================================

class SpeechState(Enum):
    """States for avatar speech."""
    SILENT = "silent"
    SPEAKING = "speaking"
    LISTENING = "listening"
    THINKING = "thinking"

class VoiceSystem:
    """Handles voice input and output."""
    
    def __init__(self):
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.current_audio_file = None
        self.voice_thread = None
        self.speech_enabled = True
        
        # Configure microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("✅ Voice system initialized")
    
    def speak(self, text: str, language: str = 'en'):
        """Convert text to speech and play it."""
        if not self.speech_enabled:
            return
        
        def _speak_thread():
            try:
                # Create temporary file for speech
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    temp_file = f.name
                
                # Generate speech
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(temp_file)
                
                # Play audio
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                # Clean up
                os.unlink(temp_file)
                
            except Exception as e:
                print(f"⚠️ Speech synthesis failed: {e}")
        
        # Run in thread to avoid blocking
        threading.Thread(target=_speak_thread, daemon=True).start()
    
    def start_listening(self):
        """Start listening for voice commands."""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.voice_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.voice_thread.start()
        print("🎤 Listening for voice commands...")
    
    def stop_listening(self):
        """Stop listening for voice commands."""
        self.is_listening = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1)
        print("🎤 Stopped listening")
    
    def _listen_loop(self):
        """Main listening loop."""
        while self.is_listening:
            try:
                with self.microphone as source:
                    print("🎤 Listening... (say 'ROCA' to activate)")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                    # Try to recognize speech
                    try:
                        text = self.recognizer.recognize_google(audio)
                        print(f"🎤 Heard: {text}")
                        
                        # Check for activation word
                        if 'roca' in text.lower():
                            print("🎤 Activation word detected!")
                            self.command_queue.put(("activate", text))
                        
                        # Process commands
                        self._process_voice_command(text)
                        
                    except sr.UnknownValueError:
                        print("🎤 Could not understand audio")
                    except sr.RequestError as e:
                        print(f"🎤 Recognition service error: {e}")
                        
            except Exception as e:
                print(f"🎤 Listening error: {e}")
                time.sleep(1)
    
    def _process_voice_command(self, text: str):
        """Process voice commands."""
        text_lower = text.lower()
        
        # Drawing commands
        if any(cmd in text_lower for cmd in ['draw', 'paint', 'sketch']):
            self.command_queue.put(("draw", text))
        elif any(cmd in text_lower for cmd in ['clear', 'erase']):
            self.command_queue.put(("clear", text))
        elif any(cmd in text_lower for cmd in ['new frame', 'add frame']):
            self.command_queue.put(("new_frame", text))
        elif any(cmd in text_lower for cmd in ['play', 'playback']):
            self.command_queue.put(("play", text))
        elif any(cmd in text_lower for cmd in ['stop', 'pause']):
            self.command_queue.put(("stop", text))
        
        # Color commands
        elif 'red' in text_lower:
            self.command_queue.put(("color", "red"))
        elif 'blue' in text_lower:
            self.command_queue.put(("color", "blue"))
        elif 'green' in text_lower:
            self.command_queue.put(("color", "green"))
        elif 'black' in text_lower:
            self.command_queue.put(("color", "black"))
        
        # System commands
        elif any(cmd in text_lower for cmd in ['help', 'what can you do']):
            self.command_queue.put(("help", text))
        elif any(cmd in text_lower for cmd in ['thank', 'thanks']):
            self.command_queue.put(("thanks", text))
        
        # Default: treat as chat message
        else:
            self.command_queue.put(("chat", text))
    
    def get_command(self):
        """Get next voice command if available."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

# ============================================================================
# ENHANCED AVATAR WITH SPEECH
# ============================================================================

class EnhancedAvatar:
    """Enhanced avatar with speech visualization."""
    
    def __init__(self, x: int, y: int, width: int, height: int, voice_system: VoiceSystem):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.voice_system = voice_system
        
        # Emotion and speech state
        self.emotion = "neutral"
        self.speech_state = SpeechState.SILENT
        self.speech_text = ""
        self.speech_progress = 0
        self.mouth_open = False
        self.blink_timer = 0
        self.eye_direction = (0, 0)
        self.rotation = 0
        
        # Speech bubble
        self.speech_bubble_visible = False
        self.speech_bubble_text = ""
        self.speech_bubble_timer = 0
        
        # Create surface
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Colors
        self.face_color = (255, 224, 189)  # Skin tone
        self.eye_color = (255, 255, 255)
        self.pupil_color = (64, 64, 64)
        self.mouth_color = (200, 75, 75)
        self.hair_color = (45, 30, 20)
        self.glasses_color = (100, 100, 100, 180)
        
    def update(self, dt: float):
        """Update avatar animation."""
        self.rotation += dt * 30  # Slow rotation
        
        # Update blink
        self.blink_timer += dt
        if self.blink_timer > 5:  # Blink every 5 seconds
            self.blink_timer = 0
        
        # Update speech animation
        if self.speech_state == SpeechState.SPEAKING:
            self.mouth_open = (time.time() * 10) % 2 > 1
            self.speech_progress = (self.speech_progress + dt * 2) % 1.0
        elif self.speech_state == SpeechState.LISTENING:
            self.mouth_open = False
            # Pulse animation for listening
            pulse = math.sin(time.time() * 5) * 0.5 + 0.5
            self.face_color = (255, 224 + int(30 * pulse), 189)
        else:
            self.mouth_open = False
            self.face_color = (255, 224, 189)  # Reset to normal
        
        # Update speech bubble
        if self.speech_bubble_visible:
            self.speech_bubble_timer += dt
            if self.speech_bubble_timer > 3:  # Show for 3 seconds
                self.speech_bubble_visible = False
    
    def speak(self, text: str):
        """Make avatar speak text."""
        self.speech_state = SpeechState.SPEAKING
        self.speech_text = text
        self.speech_progress = 0
        
        # Show speech bubble
        self.speech_bubble_visible = True
        self.speech_bubble_text = text
        self.speech_bubble_timer = 0
        
        # Actually speak using voice system
        self.voice_system.speak(text)
        
        # Set timer to return to neutral state
        def reset_state():
            time.sleep(len(text.split()) * 0.3)  # Rough estimate based on word count
            self.speech_state = SpeechState.SILENT
        
        threading.Thread(target=reset_state, daemon=True).start()
    
    def start_listening(self):
        """Switch to listening state."""
        self.speech_state = SpeechState.LISTENING
    
    def stop_listening(self):
        """Return to neutral state."""
        self.speech_state = SpeechState.SILENT
    
    def set_emotion(self, emotion: str):
        """Set avatar emotion."""
        self.emotion = emotion
        # Emotion-specific colors
        if emotion == "happy":
            self.mouth_color = (220, 100, 100)
            self.face_color = (255, 234, 199)
        elif emotion == "thinking":
            self.mouth_color = (150, 150, 150)
            self.face_color = (240, 240, 240)
        elif emotion == "surprised":
            self.mouth_color = (255, 100, 100)
            self.face_color = (255, 244, 209)
    
    def render(self) -> pygame.Surface:
        """Render the avatar."""
        self.surface.fill((0, 0, 0, 0))  # Clear with transparency
        
        # Draw head (with subtle 3D effect)
        center_x = self.width // 2
        center_y = self.height // 2
        head_radius = min(self.width, self.height) // 3
        
        # Head shadow for 3D effect
        for i in range(10, 0, -1):
            alpha = 30 - i * 3
            radius = head_radius + i
            color = (*self.face_color[:3], alpha)
            pygame.draw.circle(self.surface, color, 
                             (center_x + 2, center_y + 2), radius)
        
        # Head
        pygame.draw.circle(self.surface, self.face_color, 
                         (center_x, center_y), head_radius)
        
        # Draw hair
        hair_points = []
        for angle in np.linspace(0, 2 * math.pi, 20):
            x = center_x + head_radius * 0.9 * math.cos(angle)
            y = center_y + head_radius * 0.9 * math.sin(angle) - 10
            noise = random.randint(-5, 5)
            hair_points.append((x + noise, y + noise))
        if len(hair_points) > 2:
            pygame.draw.polygon(self.surface, self.hair_color, hair_points)
        
        # Draw eyes
        eye_y = center_y - head_radius // 4
        eye_radius = head_radius // 6
        
        # Blinking
        blink_factor = 1.0
        if self.blink_timer > 4.8 and self.blink_timer < 5.0:
            blink_factor = max(0.1, (5.0 - self.blink_timer) / 0.2)
        
        # Left eye
        left_eye_x = center_x - head_radius // 3
        pygame.draw.circle(self.surface, self.eye_color, 
                         (left_eye_x, eye_y), eye_radius)
        
        # Pupil with direction
        pupil_offset_x = self.eye_direction[0] * eye_radius * 0.5
        pupil_offset_y = self.eye_direction[1] * eye_radius * 0.5
        pygame.draw.circle(self.surface, self.pupil_color,
                         (left_eye_x + pupil_offset_x, eye_y + pupil_offset_y),
                         eye_radius * 0.5)
        
        # Right eye
        right_eye_x = center_x + head_radius // 3
        pygame.draw.circle(self.surface, self.eye_color,
                         (right_eye_x, eye_y), eye_radius)
        pygame.draw.circle(self.surface, self.pupil_color,
                         (right_eye_x + pupil_offset_x, eye_y + pupil_offset_y),
                         eye_radius * 0.5)
        
        # Draw glasses
        glasses_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        frame_thickness = 3
        glass_radius = eye_radius * 1.5
        
        # Left lens
        pygame.draw.circle(glasses_surface, (*self.glasses_color[:3], 100),
                         (left_eye_x, eye_y), glass_radius, frame_thickness)
        
        # Right lens
        pygame.draw.circle(glasses_surface, (*self.glasses_color[:3], 100),
                         (right_eye_x, eye_y), glass_radius, frame_thickness)
        
        # Bridge
        pygame.draw.line(glasses_surface, self.glasses_color,
                        (left_eye_x + glass_radius - 5, eye_y),
                        (right_eye_x - glass_radius + 5, eye_y),
                        frame_thickness)
        
        self.surface.blit(glasses_surface, (0, 0))
        
        # Draw mouth based on emotion and speech
        mouth_y = center_y + head_radius // 3
        mouth_width = head_radius // 2
        
        if self.speech_state == SpeechState.SPEAKING and self.mouth_open:
            # Open mouth for speech
            mouth_height = head_radius // 4
            mouth_rect = pygame.Rect(center_x - mouth_width // 2,
                                   mouth_y - mouth_height // 2,
                                   mouth_width, mouth_height)
            pygame.draw.ellipse(self.surface, self.mouth_color, mouth_rect)
            
            # Tongue for speaking
            tongue_y = mouth_y + mouth_height // 4
            tongue_width = mouth_width * 0.6
            pygame.draw.ellipse(self.surface, (255, 150, 150),
                              (center_x - tongue_width // 2, tongue_y,
                               tongue_width, mouth_height // 3))
        
        elif self.emotion == "happy":
            # Smile
            mouth_rect = pygame.Rect(center_x - mouth_width // 2,
                                   mouth_y - 5,
                                   mouth_width, 15)
            pygame.draw.arc(self.surface, self.mouth_color,
                          mouth_rect, math.pi, 2 * math.pi, 3)
        
        elif self.emotion == "thinking":
            # Straight line
            pygame.draw.line(self.surface, self.mouth_color,
                           (center_x - mouth_width // 2, mouth_y),
                           (center_x + mouth_width // 2, mouth_y), 3)
        
        else:
            # Neutral mouth
            mouth_rect = pygame.Rect(center_x - mouth_width // 2,
                                   mouth_y,
                                   mouth_width, 8)
            pygame.draw.ellipse(self.surface, self.mouth_color, mouth_rect, 2)
        
        # Draw speech bubble if visible
        if self.speech_bubble_visible:
            self._draw_speech_bubble()
        
        # Draw state indicator
        self._draw_state_indicator()
        
        return self.surface
    
    def _draw_speech_bubble(self):
        """Draw speech bubble with text."""
        bubble_x = self.width + 10
        bubble_y = 20
        bubble_width = 250
        bubble_height = 100
        
        # Bubble background
        bubble_rect = pygame.Rect(bubble_x, bubble_y, bubble_width, bubble_height)
        pygame.draw.rect(self.surface, (255, 255, 255), bubble_rect, border_radius=10)
        pygame.draw.rect(self.surface, (100, 100, 100), bubble_rect, 2, border_radius=10)
        
        # Pointer to avatar
        points = [
            (bubble_x, bubble_y + 20),
            (bubble_x - 10, bubble_y + 30),
            (self.width - 10, self.height // 2)
        ]
        pygame.draw.polygon(self.surface, (255, 255, 255), points)
        pygame.draw.polygon(self.surface, (100, 100, 100), points, 2)
        
        # Text
        font = pygame.font.SysFont("Arial", 14)
        words = self.speech_bubble_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] < bubble_width - 20:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        y_offset = bubble_y + 10
        for line in lines[:3]:  # Max 3 lines
            text_surface = font.render(line, True, (0, 0, 0))
            self.surface.blit(text_surface, (bubble_x + 10, y_offset))
            y_offset += 18
    
    def _draw_state_indicator(self):
        """Draw indicator for current state."""
        indicator_y = self.height - 20
        indicator_radius = 8
        
        if self.speech_state == SpeechState.SPEAKING:
            color = (0, 200, 0)  # Green for speaking
            # Pulsing effect
            pulse_size = int(indicator_radius * (1 + 0.3 * math.sin(time.time() * 10)))
            pygame.draw.circle(self.surface, color,
                             (self.width - 20, indicator_y), pulse_size)
        elif self.speech_state == SpeechState.LISTENING:
            color = (0, 150, 255)  # Blue for listening
            # Rotating effect
            angle = time.time() * 5
            points = []
            for i in range(3):
                point_angle = angle + i * (2 * math.pi / 3)
                x = self.width - 20 + math.cos(point_angle) * indicator_radius
                y = indicator_y + math.sin(point_angle) * indicator_radius
                points.append((x, y))
            pygame.draw.polygon(self.surface, color, points)
        elif self.speech_state == SpeechState.THINKING:
            color = (255, 200, 0)  # Yellow for thinking
            # Blinking effect
            if int(time.time() * 2) % 2:
                pygame.draw.circle(self.surface, color,
                                 (self.width - 20, indicator_y), indicator_radius)
        
        # Draw indicator border
        pygame.draw.circle(self.surface, (0, 0, 0),
                         (self.width - 20, indicator_y), indicator_radius, 1)

# ============================================================================
# SIMPLIFIED DRAWING SYSTEM
# ============================================================================

@dataclass
class DrawingState:
    """Simplified drawing state without PyTorch dependencies."""
    drawing: bool = False
    last_pos: Optional[Tuple[int, int]] = None
    brush_color: Tuple[int, int, int] = (0, 0, 0)
    brush_size: int = 5
    layers: List[pygame.Surface] = None
    current_layer: int = 0
    animation_frames: List[pygame.Surface] = None
    current_frame: int = 0
    playback: bool = False
    playback_frame: int = 0
    playback_speed: int = 30
    onion_skinning: bool = False
    undo_stack: List[pygame.Surface] = None
    redo_stack: List[pygame.Surface] = None
    is_recording: bool = False
    recording_frames: List[pygame.Surface] = None
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [pygame.Surface((800, 450), pygame.SRCALPHA)]
        if self.animation_frames is None:
            self.animation_frames = [pygame.Surface((800, 450), pygame.SRCALPHA)]
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []
        if self.recording_frames is None:
            self.recording_frames = []
    
    def save_current_frame(self):
        """Save current layers to animation frames."""
        if self.current_frame < len(self.animation_frames):
            combined = pygame.Surface((800, 450), pygame.SRCALPHA)
            for layer in self.layers:
                combined.blit(layer, (0, 0))
            self.animation_frames[self.current_frame] = combined.copy()
    
    def switch_to_frame(self, frame_index: int):
        """Switch to a different frame."""
        frame_index = max(0, min(frame_index, len(self.animation_frames) - 1))
        self.current_frame = frame_index
        self.layers = [self.animation_frames[frame_index].copy()]
    
    def new_frame(self):
        """Create a new frame."""
        new_frame = pygame.Surface((800, 450), pygame.SRCALPHA)
        self.animation_frames.append(new_frame)
        self.switch_to_frame(len(self.animation_frames) - 1)
        return len(self.animation_frames)
    
    def add_layer(self):
        """Add a new drawing layer."""
        self.layers.append(pygame.Surface((800, 450), pygame.SRCALPHA))
        self.current_layer = len(self.layers) - 1
        return len(self.layers)
    
    def remove_layer(self):
        """Remove the current layer if more than one exists."""
        if len(self.layers) > 1:
            self.layers.pop()
            self.current_layer = min(self.current_layer, len(self.layers) - 1)
        return len(self.layers)
    
    def undo(self):
        """Undo the last drawing action."""
        if self.undo_stack:
            self.redo_stack.append(self.layers[self.current_layer].copy())
            self.layers[self.current_layer] = self.undo_stack.pop()
            return True
        return False
    
    def redo(self):
        """Redo the last undone action."""
        if self.redo_stack:
            self.undo_stack.append(self.layers[self.current_layer].copy())
            self.layers[self.current_layer] = self.redo_stack.pop()
            return True
        return False

# ============================================================================
# SIMPLIFIED FILE PROCESSOR
# ============================================================================

class SimpleFileProcessor:
    """Process files without PyTorch dependencies."""
    
    def __init__(self, brain=None, avatar=None):
        self.brain = brain
        self.avatar = avatar
        self.supported_extensions = {
            '.json': self._process_json,
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.pdf': self._process_pdf,
            '.png': self._process_image,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
        }
    
    def process_file(self, file_path: str) -> str:
        """Process a dropped file."""
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.supported_extensions:
            try:
                result = self.supported_extensions[file_ext](file_path)
                
                # Notify avatar
                if self.avatar:
                    self.avatar.speak(f"Processed {file_name}")
                
                return result
            except Exception as e:
                return f"❌ Error processing {file_ext} file: {str(e)}"
        else:
            return f"❌ Unsupported file type: {file_ext}"
    
    def _process_json(self, file_path: str) -> str:
        """Process JSON files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Simple analysis
        if isinstance(data, dict):
            keys = list(data.keys())
            return f"✅ JSON object with {len(keys)} keys: {', '.join(keys[:5])}..."
        elif isinstance(data, list):
            return f"✅ JSON array with {len(data)} items"
        else:
            return f"✅ JSON data loaded"
    
    def _process_text(self, file_path: str) -> str:
        """Process text files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        word_count = len(content.split())
        char_count = len(content)
        return f"✅ Text file: {word_count} words, {char_count} characters"
    
    def _process_markdown(self, file_path: str) -> str:
        """Process Markdown files."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count headings
        headings = len(re.findall(r'^#+', content, re.MULTILINE))
        return f"✅ Markdown: {headings} headings found"
    
    def _process_pdf(self, file_path: str) -> str:
        """Process PDF files if PyMuPDF is available."""
        if not PDF_AVAILABLE:
            return "❌ PDF processing requires PyMuPDF: pip install PyMuPDF"
        
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return f"✅ PDF: {page_count} pages"
        except Exception as e:
            return f"❌ PDF error: {str(e)}"
    
    def _process_image(self, file_path: str) -> str:
        """Process image files."""
        try:
            img = pygame.image.load(file_path)
            size = img.get_size()
            return f"✅ Image: {size[0]}x{size[1]} pixels"
        except Exception as e:
            return f"❌ Image error: {str(e)}"

# ============================================================================
# ENHANCED CHATBOT
# ============================================================================

class EnhancedChatbot:
    """Enhanced chatbot with voice integration."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 brain=None, avatar=None, voice_system=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.brain = brain
        self.avatar = avatar
        self.voice_system = voice_system
        
        # Chat state
        self.messages = []
        self.input_text = ""
        self.input_active = False
        self.scroll_offset = 0
        
        # UI elements
        self.chat_area = pygame.Rect(x, y, width, height - 50)
        self.input_area = pygame.Rect(x, y + height - 40, width, 35)
        
        # Fonts
        self.chat_font = pygame.font.SysFont("Arial", 16)
        self.input_font = pygame.font.SysFont("Arial", 18)
        
        # Colors
        self.bg_color = (240, 240, 240)
        self.border_color = (100, 100, 100)
        self.input_bg = (255, 255, 255)
        self.input_border = (150, 150, 150)
        self.text_color = (0, 0, 0)
        
        # Initial greeting
        self.add_message("ROCA", "Hello! I'm your creative AI assistant. You can talk to me or type your commands!")
        if voice_system:
            self.add_message("ROCA", "Try saying 'ROCA' followed by a command like 'draw a circle' or 'clear the canvas'")
    
    def add_message(self, sender: str, message: str):
        """Add a message to chat."""
        timestamp = time.strftime("%H:%M", time.localtime())
        self.messages.append({
            'sender': sender,
            'message': message,
            'timestamp': timestamp
        })
        
        # Auto-scroll to bottom
        self._update_scroll_to_bottom()
        
        # Speak if from ROCA and avatar exists
        if sender == "ROCA" and self.avatar:
            self.avatar.speak(message)
    
    def handle_voice_command(self, command_type: str, text: str):
        """Handle voice commands."""
        if command_type == "chat":
            self.add_message("You (Voice)", text)
            self._process_message(text)
        elif command_type == "activate":
            self.add_message("System", "Voice activated! Listening for commands...")
            if self.avatar:
                self.avatar.start_listening()
                self.avatar.speak("I'm listening")
        elif command_type == "draw":
            self.add_message("You (Voice)", f"Draw command: {text}")
            self.add_message("ROCA", "I'll help you draw! Use your mouse or tell me what to draw.")
        elif command_type == "clear":
            self.add_message("You (Voice)", "Clear canvas")
            self.add_message("ROCA", "Canvas cleared!")
        elif command_type == "help":
            self.show_help()
        elif command_type == "thanks":
            self.add_message("ROCA", "You're welcome! Happy creating!")
    
    def show_help(self):
        """Show help message."""
        help_text = """
Available commands:
- Voice: Say 'ROCA' then your command
- Drawing: Click and drag on canvas
- Colors: Say 'red', 'blue', 'green', or 'black'
- Frames: 'new frame', 'play', 'stop'
- File: Drag & drop files to import
- Chat: Type or speak naturally
        """
        self.add_message("ROCA", help_text)
    
    def _process_message(self, message: str):
        """Process a chat message."""
        # Simple response logic (can be enhanced with brain)
        message_lower = message.lower()
        
        if any(greet in message_lower for greet in ['hello', 'hi', 'hey']):
            self.add_message("ROCA", f"Hello! How can I help you today?")
        elif any(question in message_lower for question in ['how are you', 'how do you']):
            self.add_message("ROCA", "I'm doing great! Ready to help you create amazing art!")
        elif 'draw' in message_lower:
            self.add_message("ROCA", "Go ahead and draw on the canvas! You can also tell me what colors to use.")
        elif 'color' in message_lower:
            colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']
            found_colors = [c for c in colors if c in message_lower]
            if found_colors:
                self.add_message("ROCA", f"Setting color to {found_colors[0]}!")
            else:
                self.add_message("ROCA", "What color would you like?")
        elif 'thank' in message_lower:
            self.add_message("ROCA", "You're welcome! Keep creating!")
        else:
            # Default creative response
            responses = [
                "That's interesting! Tell me more about your creative vision.",
                "I'd love to help you bring that idea to life on the canvas!",
                "Great idea! How would you like to visualize it?",
                "Let's create something amazing together!",
                "I'm here to help with your artistic journey!"
            ]
            self.add_message("ROCA", random.choice(responses))
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse clicks."""
        if self.input_area.collidepoint(pos):
            self.input_active = True
        else:
            self.input_active = False
    
    def handle_key(self, event: pygame.event.Event):
        """Handle keyboard input."""
        if not self.input_active:
            return
        
        if event.key == pygame.K_RETURN:
            if self.input_text.strip():
                self.add_message("You", self.input_text.strip())
                self._process_message(self.input_text.strip())
                self.input_text = ""
        elif event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.key == pygame.K_ESCAPE:
            self.input_active = False
        elif event.key == pygame.K_UP:
            # Scroll up
            self.scroll_offset = max(0, self.scroll_offset - 30)
        elif event.key == pygame.K_DOWN:
            # Scroll down
            self.scroll_offset += 30
        else:
            if len(self.input_text) < 100:
                self.input_text += event.unicode
    
    def _update_scroll_to_bottom(self):
        """Update scroll to show latest messages."""
        # Simplified - just keep at bottom
        self.scroll_offset = 0
    
    def draw(self, screen: pygame.Surface):
        """Draw the chatbot."""
        # Background
        pygame.draw.rect(screen, self.bg_color, 
                        (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, self.border_color,
                        (self.x, self.y, self.width, self.height), 2)
        
        # Chat area
        pygame.draw.rect(screen, (255, 255, 255), self.chat_area)
        pygame.draw.rect(screen, self.border_color, self.chat_area, 1)
        
        # Draw messages
        y_offset = self.chat_area.y + 10 - self.scroll_offset
        for msg in self.messages[-10:]:  # Show last 10 messages
            if y_offset > self.chat_area.bottom:
                break
            
            # Sender and timestamp
            sender_text = f"{msg['sender']} [{msg['timestamp']}]:"
            sender_surface = self.chat_font.render(sender_text, True, (50, 50, 150))
            screen.blit(sender_surface, (self.chat_area.x + 10, y_offset))
            y_offset += 20
            
            # Message (with word wrap)
            words = msg['message'].split()
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                if self.chat_font.size(test_line)[0] < self.chat_area.width - 20:
                    line = test_line
                else:
                    if line:
                        msg_surface = self.chat_font.render(line, True, self.text_color)
                        screen.blit(msg_surface, (self.chat_area.x + 20, y_offset))
                        y_offset += 20
                    line = word
            
            if line:
                msg_surface = self.chat_font.render(line, True, self.text_color)
                screen.blit(msg_surface, (self.chat_area.x + 20, y_offset))
                y_offset += 20
            
            y_offset += 10
        
        # Input area
        pygame.draw.rect(screen, self.input_bg, self.input_area)
        pygame.draw.rect(screen, 
                        self.input_border if self.input_active else self.border_color,
                        self.input_area, 2)
        
        # Input text
        input_surface = self.input_font.render(self.input_text, True, self.text_color)
        screen.blit(input_surface, (self.input_area.x + 10, self.input_area.y + 8))
        
        # Cursor
        if self.input_active and int(time.time() * 2) % 2:
            cursor_x = self.input_area.x + 10 + input_surface.get_width()
            pygame.draw.line(screen, self.text_color,
                           (cursor_x, self.input_area.y + 8),
                           (cursor_x, self.input_area.y + 28), 2)

# ============================================================================
# SIMPLIFIED GUI
# ============================================================================

class DrawingGUI:
    """Simplified GUI without complex dependencies."""
    
    def __init__(self):
        self.buttons = [
            {"label": "🎤 Voice", "rect": pygame.Rect(50, 610, 100, 30), "tooltip": "Toggle voice commands"},
            {"label": "🔄 Mode", "rect": pygame.Rect(160, 610, 100, 30), "tooltip": "Switch modes"},
            {"label": "🎨 Color", "rect": pygame.Rect(270, 610, 100, 30), "tooltip": "Change color"},
            {"label": "📷 Frame+", "rect": pygame.Rect(380, 610, 100, 30), "tooltip": "New frame"},
            {"label": "▶️ Play", "rect": pygame.Rect(490, 610, 100, 30), "tooltip": "Play animation"},
            {"label": "💾 Save", "rect": pygame.Rect(600, 610, 100, 30), "tooltip": "Save project"},
            {"label": "📂 Import", "rect": pygame.Rect(50, 650, 100, 30), "tooltip": "Import files"},
            {"label": "🔄 Undo", "rect": pygame.Rect(160, 650, 100, 30), "tooltip": "Undo last action"},
            {"label": "↩️ Redo", "rect": pygame.Rect(270, 650, 100, 30), "tooltip": "Redo action"},
            {"label": "🧅 Onion", "rect": pygame.Rect(380, 650, 100, 30), "tooltip": "Toggle onion skinning"},
            {"label": "🎞️ Timeline", "rect": pygame.Rect(490, 650, 100, 30), "tooltip": "Show timeline"},
            {"label": "❓ Help", "rect": pygame.Rect(600, 650, 100, 30), "tooltip": "Show help"},
        ]
        
        self.sliders = {
            "brush_size": {
                "rect": pygame.Rect(50, 690, 200, 20),
                "value": 5,
                "min": 1,
                "max": 50,
                "label": "Brush Size"
            },
            "playback_speed": {
                "rect": pygame.Rect(300, 690, 200, 20),
                "value": 30,
                "min": 1,
                "max": 60,
                "label": "Speed"
            }
        }
        
        self.font = pygame.font.SysFont("Arial", 16)
        self.tooltip_font = pygame.font.SysFont("Arial", 14)
        self.hover_button = None
        self.active_slider = None
        self.show_timeline = True
        self.timeline = pygame.Rect(50, 720, 800, 30)
    
    def handle_hover(self, pos: Tuple[int, int]):
        """Update hover state."""
        self.hover_button = None
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                self.hover_button = button
                break
    
    def handle_click(self, pos: Tuple[int, int]) -> Optional[str]:
        """Handle clicks and return button label if clicked."""
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                return button["label"]
        
        # Check sliders
        for name, slider in self.sliders.items():
            if slider["rect"].collidepoint(pos):
                self.active_slider = name
                self._update_slider(pos, slider)
                return f"slider_{name}"
        
        # Timeline
        if self.timeline.collidepoint(pos) and self.show_timeline:
            return "timeline"
        
        return None
    
    def _update_slider(self, pos: Tuple[int, int], slider: dict):
        """Update slider value based on position."""
        x = max(slider["rect"].left, min(pos[0], slider["rect"].right))
        ratio = (x - slider["rect"].left) / slider["rect"].width
        slider["value"] = slider["min"] + ratio * (slider["max"] - slider["min"])
        slider["value"] = int(slider["value"])
    
    def draw(self, screen: pygame.Surface, state: DrawingState):
        """Draw the GUI."""
        # Background panel
        pygame.draw.rect(screen, (245, 245, 245), (0, 600, 1200, 200))
        pygame.draw.line(screen, (200, 200, 200), (0, 600), (1200, 600), 2)
        
        # Buttons
        for button in self.buttons:
            # Button background
            color = (220, 220, 220) if button == self.hover_button else (200, 200, 200)
            pygame.draw.rect(screen, color, button["rect"], border_radius=5)
            pygame.draw.rect(screen, (150, 150, 150), button["rect"], 1, border_radius=5)
            
            # Button text
            text = self.font.render(button["label"], True, (40, 40, 40))
            text_rect = text.get_rect(center=button["rect"].center)
            screen.blit(text, text_rect)
        
        # Sliders
        for name, slider in self.sliders.items():
            # Slider track
            pygame.draw.rect(screen, (180, 180, 180), slider["rect"], border_radius=3)
            pygame.draw.rect(screen, (120, 120, 120), slider["rect"], 1, border_radius=3)
            
            # Slider handle
            handle_x = slider["rect"].left + int(
                (slider["value"] - slider["min"]) / 
                (slider["max"] - slider["min"]) * slider["rect"].width
            )
            pygame.draw.circle(screen, (80, 80, 80), 
                             (handle_x, slider["rect"].centery), 8)
            pygame.draw.circle(screen, (40, 40, 40),
                             (handle_x, slider["rect"].centery), 8, 1)
            
            # Slider label
            label = f"{slider['label']}: {slider['value']}"
            label_surface = self.font.render(label, True, (60, 60, 60))
            screen.blit(label_surface, (slider["rect"].x, slider["rect"].y - 20))
        
        # Timeline
        if self.show_timeline:
            pygame.draw.rect(screen, (230, 230, 230), self.timeline, border_radius=3)
            pygame.draw.rect(screen, (150, 150, 150), self.timeline, 1, border_radius=3)
            
            if state.animation_frames:
                frame_width = self.timeline.width / len(state.animation_frames)
                for i in range(len(state.animation_frames)):
                    frame_rect = pygame.Rect(
                        self.timeline.x + i * frame_width,
                        self.timeline.y,
                        frame_width - 1,
                        self.timeline.height
                    )
                    
                    # Current frame highlight
                    if i == state.current_frame:
                        pygame.draw.rect(screen, (100, 150, 200), frame_rect, border_radius=2)
                    elif i == state.playback_frame and state.playback:
                        pygame.draw.rect(screen, (200, 100, 100), frame_rect, border_radius=2)
                    
                    # Frame number
                    if frame_width > 20:  # Only show numbers if there's space
                        num_text = self.font.render(str(i+1), True, (60, 60, 60))
                        num_rect = num_text.get_rect(center=frame_rect.center)
                        screen.blit(num_text, num_rect)
        
        # Tooltip
        if self.hover_button:
            tooltip = self.hover_button["tooltip"]
            tooltip_surface = self.tooltip_font.render(tooltip, True, (255, 255, 255))
            tooltip_bg = pygame.Rect(
                pygame.mouse.get_pos()[0] + 10,
                pygame.mouse.get_pos()[1] + 10,
                tooltip_surface.get_width() + 10,
                tooltip_surface.get_height() + 5
            )
            pygame.draw.rect(screen, (0, 0, 0, 180), tooltip_bg, border_radius=3)
            screen.blit(tooltip_surface, (tooltip_bg.x + 5, tooltip_bg.y + 2))
        
        # Status text
        status = f"Frame: {state.current_frame + 1}/{len(state.animation_frames)} | "
        status += f"Layer: {state.current_layer + 1}/{len(state.layers)} | "
        status += f"Brush: {state.brush_size}px"
        status_surface = self.font.render(status, True, (80, 80, 80))
        screen.blit(status_surface, (850, 750))
        
        # Voice status
        voice_status = "🎤 Voice: ON" if state.is_recording else "🎤 Voice: OFF"
        voice_surface = self.font.render(voice_status, True, (80, 80, 80))
        screen.blit(voice_surface, (850, 770))

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Initialize Pygame
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("ROCA Creative Studio - Voice-Enabled Drawing & Animation")
    
    # Initialize systems
    voice_system = VoiceSystem()
    drawing_state = DrawingState()
    gui = DrawingGUI()
    
    # Initialize AI brain (if available)
    brain = None
    if BRAIN_AVAILABLE:
        try:
            brain = AutonomousBrain()
            print("✅ AI Brain initialized")
        except Exception as e:
            print(f"⚠️ Brain initialization failed: {e}")
            brain = None
    
    # Initialize avatar and chatbot
    avatar = EnhancedAvatar(860, 50, 320, 90, voice_system)
    chatbot = EnhancedChatbot(860, 150, 320, 450, brain, avatar, voice_system)
    
    # Initialize file processor
    file_processor = SimpleFileProcessor(brain, avatar)
    
    # Define drawing area
    DRAWING_AREA = pygame.Rect(50, 50, 800, 450)
    
    # Colors
    COLOR_MAP = {
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
    }
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    last_time = time.time()
    
    # Start voice system
    voice_system.start_listening()
    
    print("\n" + "="*60)
    print("ROCA Creative Studio - Voice Commands Enabled")
    print("="*60)
    print("Try saying: 'ROCA draw a circle' or 'ROCA change to blue'")
    print("Or click the 🎤 button to toggle voice control")
    print("="*60 + "\n")
    
    while running:
        dt = time.time() - last_time
        last_time = time.time()
        
        # Update avatar
        avatar.update(dt)
        
        # Process voice commands
        voice_command = voice_system.get_command()
        if voice_command:
            cmd_type, cmd_text = voice_command
            chatbot.handle_voice_command(cmd_type, cmd_text)
            
            # Execute commands
            if cmd_type == "color" and cmd_text in COLOR_MAP:
                drawing_state.brush_color = COLOR_MAP[cmd_text]
                avatar.speak(f"Changed to {cmd_text}")
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.DROPFILE:
                # Handle file drag & drop
                result = file_processor.process_file(event.file)
                chatbot.add_message("System", result)
            
            elif event.type == pygame.KEYDOWN:
                # Handle keyboard
                chatbot.handle_key(event)
                
                # Quick shortcuts
                if event.key == pygame.K_SPACE:
                    drawing_state.playback = not drawing_state.playback
                    if drawing_state.playback:
                        drawing_state.playback_frame = drawing_state.current_frame
                        chatbot.add_message("System", "Playback started")
                    else:
                        chatbot.add_message("System", "Playback stopped")
                elif event.key == pygame.K_c:
                    # Clear current layer
                    drawing_state.layers[drawing_state.current_layer].fill((0, 0, 0, 0))
                    drawing_state.undo_stack.append(
                        drawing_state.layers[drawing_state.current_layer].copy()
                    )
                    chatbot.add_message("System", "Canvas cleared")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Drawing
                if DRAWING_AREA.collidepoint(event.pos):
                    x = event.pos[0] - DRAWING_AREA.x
                    y = event.pos[1] - DRAWING_AREA.y
                    drawing_state.drawing = True
                    drawing_state.last_pos = (x, y)
                    
                    # Save for undo
                    drawing_state.undo_stack.append(
                        drawing_state.layers[drawing_state.current_layer].copy()
                    )
                    drawing_state.redo_stack.clear()
                
                # GUI
                else:
                    button = gui.handle_click(event.pos)
                    if button:
                        handle_button_click(button, drawing_state, gui, chatbot, avatar, voice_system)
                    else:
                        chatbot.handle_click(event.pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing_state.drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                gui.handle_hover(event.pos)
                
                # Update active slider
                if gui.active_slider and pygame.mouse.get_pressed()[0]:
                    slider = gui.sliders[gui.active_slider]
                    gui._update_slider(event.pos, slider)
                    
                    # Apply slider values
                    if gui.active_slider == "brush_size":
                        drawing_state.brush_size = slider["value"]
                    elif gui.active_slider == "playback_speed":
                        drawing_state.playback_speed = slider["value"]
                
                # Drawing
                if drawing_state.drawing and DRAWING_AREA.collidepoint(event.pos):
                    x = event.pos[0] - DRAWING_AREA.x
                    y = event.pos[1] - DRAWING_AREA.y
                    
                    if drawing_state.last_pos:
                        pygame.draw.line(
                            drawing_state.layers[drawing_state.current_layer],
                            drawing_state.brush_color,
                            drawing_state.last_pos,
                            (x, y),
                            drawing_state.brush_size
                        )
                    
                    drawing_state.last_pos = (x, y)
        
        # Handle playback
        if drawing_state.playback:
            current_time = time.time()
            if (current_time - drawing_state.last_frame_time > 
                1 / drawing_state.playback_speed):
                drawing_state.playback_frame = (
                    drawing_state.playback_frame + 1
                ) % len(drawing_state.animation_frames)
                drawing_state.last_frame_time = current_time
        
        # Draw everything
        screen.fill((255, 255, 255))
        
        # Draw drawing area
        pygame.draw.rect(screen, (240, 240, 240), DRAWING_AREA)
        pygame.draw.rect(screen, (100, 100, 100), DRAWING_AREA, 2)
        
        if drawing_state.playback:
            # Show playback frame
            frame = drawing_state.animation_frames[drawing_state.playback_frame]
            screen.blit(frame, DRAWING_AREA.topleft)
        else:
            # Show current drawing
            for layer in drawing_state.layers:
                screen.blit(layer, DRAWING_AREA.topleft)
            
            # Onion skinning
            if drawing_state.onion_skinning and len(drawing_state.animation_frames) > 1:
                prev_idx = max(0, drawing_state.current_frame - 1)
                prev_frame = drawing_state.animation_frames[prev_idx].copy()
                prev_frame.set_alpha(100)
                screen.blit(prev_frame, DRAWING_AREA.topleft)
        
        # Draw avatar
        avatar_surface = avatar.render()
        screen.blit(avatar_surface, (860, 50))
        
        # Draw chatbot
        chatbot.draw(screen)
        
        # Draw GUI
        gui.draw(screen, drawing_state)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    voice_system.stop_listening()
    pygame.quit()
    print("\n👋 Thanks for using ROCA Creative Studio!")

def handle_button_click(button_label: str, state: DrawingState, gui: DrawingGUI,
                       chatbot: EnhancedChatbot, avatar: EnhancedAvatar,
                       voice_system: VoiceSystem):
    """Handle button clicks."""
    if button_label == "🎤 Voice":
        if voice_system.is_listening:
            voice_system.stop_listening()
            avatar.stop_listening()
            chatbot.add_message("System", "Voice commands disabled")
            state.is_recording = False
        else:
            voice_system.start_listening()
            avatar.start_listening()
            chatbot.add_message("System", "Voice commands enabled - say 'ROCA' to activate")
            state.is_recording = True
    
    elif button_label == "🔄 Mode":
        # Toggle between draw and select mode (simplified)
        chatbot.add_message("System", "Switched mode")
    
    elif button_label == "🎨 Color":
        # Cycle through colors
        colors = list(COLOR_MAP.keys())
        current_idx = list(COLOR_MAP.values()).index(state.brush_color)
        next_idx = (current_idx + 1) % len(colors)
        state.brush_color = COLOR_MAP[colors[next_idx]]
        avatar.speak(f"Changed to {colors[next_idx]} color")
        chatbot.add_message("System", f"Color: {colors[next_idx]}")
    
    elif button_label == "📷 Frame+":
        frame_count = state.new_frame()
        avatar.speak(f"New frame created. Total: {frame_count}")
        chatbot.add_message("System", f"New frame created ({frame_count} total)")
    
    elif button_label == "▶️ Play":
        state.playback = not state.playback
        if state.playback:
            state.playback_frame = state.current_frame
            state.last_frame_time = time.time()
            avatar.speak("Playing animation")
            chatbot.add_message("System", "Playback started")
        else:
            avatar.speak("Stopped playback")
            chatbot.add_message("System", "Playback stopped")
    
    elif button_label == "💾 Save":
        # Simple save to PNG
        try:
            combined = pygame.Surface((800, 450), pygame.SRCALPHA)
            for layer in state.layers:
                combined.blit(layer, (0, 0))
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_{timestamp}.png"
            pygame.image.save(combined, filename)
            avatar.speak(f"Drawing saved as {filename}")
            chatbot.add_message("System", f"Saved as {filename}")
        except Exception as e:
            chatbot.add_message("System", f"Save failed: {e}")
    
    elif button_label == "📂 Import":
        # File import dialog
        root = tk.Tk()
        root.withdraw()
        filetypes = [
            ("All supported", "*.png *.jpg *.jpeg *.txt *.json *.md"),
            ("Images", "*.png *.jpg *.jpeg"),
            ("Text", "*.txt *.json *.md"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            for file_path in files:
                result = file_processor.process_file(file_path)
                chatbot.add_message("System", f"{os.path.basename(file_path)}: {result}")
            avatar.speak(f"Imported {len(files)} files")
    
    elif button_label == "🔄 Undo":
        if state.undo():
            avatar.speak("Undone")
            chatbot.add_message("System", "Undo")
    
    elif button_label == "↩️ Redo":
        if state.redo():
            avatar.speak("Redone")
            chatbot.add_message("System", "Redo")
    
    elif button_label == "🧅 Onion":
        state.onion_skinning = not state.onion_skinning
        status = "ON" if state.onion_skinning else "OFF"
        avatar.speak(f"Onion skinning {status}")
        chatbot.add_message("System", f"Onion skinning: {status}")
    
    elif button_label == "🎞️ Timeline":
        gui.show_timeline = not gui.show_timeline
        status = "shown" if gui.show_timeline else "hidden"
        avatar.speak(f"Timeline {status}")
        chatbot.add_message("System", f"Timeline {status}")
    
    elif button_label == "❓ Help":
        chatbot.show_help()
    
    elif button_label.startswith("slider_"):
        # Slider already handled
        pass
    
    elif button_label == "timeline":
        # Timeline frame selection
        if state.animation_frames:
            frame_width = gui.timeline.width / len(state.animation_frames)
            mouse_x = pygame.mouse.get_pos()[0]
            frame_index = int((mouse_x - gui.timeline.x) / frame_width)
            frame_index = max(0, min(frame_index, len(state.animation_frames) - 1))
            state.switch_to_frame(frame_index)
            avatar.speak(f"Frame {frame_index + 1}")
            chatbot.add_message("System", f"Switched to frame {frame_index + 1}")

if __name__ == "__main__":
    # Check for required packages
    required = ['pygame', 'PIL', 'numpy']
    optional = ['fitz', 'speech_recognition', 'gtts', 'pyaudio']
    
    print("Checking dependencies...")
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Please install: pip install {package}")
            sys.exit(1)
    
    for package in optional:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️ {package} (optional) - Some features will be limited")
    
    print("\nStarting ROCA Creative Studio...")
    main()