import pygame
import sys
import math
import random
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from enum import Enum, auto
import numpy as np
from datetime import datetime

# ============================================================================
# CORE BRAIN CLASSES
# ============================================================================

class MentalState(Enum):
    IDLE = auto()
    ATTENTIVE = auto()
    CREATING = auto()
    REFLECTING = auto()
    DEEP_SLEEP = auto()

class EmotionalState(Enum):
    CURIOUS = auto()
    EXCITED = auto()
    CONTENT = auto()
    FRUSTRATED = auto()
    INSPIRED = auto()
    TIRED = auto()
    PLAYFUL = auto()
    NEUTRAL = auto()

class ThoughtType(Enum):
    OBSERVATION = auto()
    QUESTION = auto()
    IDEA = auto()
    REFLECTION = auto()
    CREATION = auto()

@dataclass
class Thought:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ThoughtType = ThoughtType.OBSERVATION
    content: str = ""
    emotion: float = 0.0
    salience: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.name,
            'content': self.content,
            'emotion': self.emotion,
            'salience': self.salience,
            'timestamp': self.timestamp
        }

@dataclass
class Perception:
    """What the brain perceives from the environment."""
    user_activity: str = "idle"
    canvas_state: Optional[Dict] = None
    timeline_state: Optional[Dict] = None
    capsule_activity: Optional[Dict] = None
    user_emotional_cues: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def update_from_environment(self, env_data: Dict):
        """Update perception from environment data."""
        self.user_activity = env_data.get('user_activity', 'idle')
        self.canvas_state = env_data.get('canvas')
        self.timeline_state = env_data.get('timeline')
        self.capsule_activity = env_data.get('capsule_activity')
        self.timestamp = time.time()
        
        # Analyze emotional cues
        cues = []
        if env_data.get('user_interaction_speed', 0) > 5:
            cues.append('hurried')
        if env_data.get('undo_actions', 0) > 3:
            cues.append('frustrated')
        if env_data.get('creative_actions', 0) > 5:
            cues.append('creative')
        if env_data.get('idle_time', 0) > 60:
            cues.append('contemplative')
        
        self.user_emotional_cues = cues

class BrainStateManager:
    """Manages transitions between mental states."""
    
    def __init__(self):
        self.current_state = MentalState.IDLE
        self.state_start_time = time.time()
        self.state_duration = 0.0
        self.state_history = deque(maxlen=20)
        
        # State transition probabilities
        self.transition_matrix = {
            MentalState.IDLE: {
                MentalState.ATTENTIVE: 0.3,
                MentalState.REFLECTING: 0.1,
                MentalState.DEEP_SLEEP: 0.05,
                MentalState.CREATING: 0.05,
                MentalState.IDLE: 0.5
            },
            MentalState.ATTENTIVE: {
                MentalState.CREATING: 0.4,
                MentalState.REFLECTING: 0.2,
                MentalState.IDLE: 0.2,
                MentalState.ATTENTIVE: 0.2
            },
            MentalState.CREATING: {
                MentalState.REFLECTING: 0.3,
                MentalState.ATTENTIVE: 0.3,
                MentalState.IDLE: 0.2,
                MentalState.CREATING: 0.2
            },
            MentalState.REFLECTING: {
                MentalState.DEEP_SLEEP: 0.3,
                MentalState.CREATING: 0.3,
                MentalState.ATTENTIVE: 0.2,
                MentalState.REFLECTING: 0.2
            },
            MentalState.DEEP_SLEEP: {
                MentalState.REFLECTING: 0.4,
                MentalState.IDLE: 0.3,
                MentalState.ATTENTIVE: 0.2,
                MentalState.DEEP_SLEEP: 0.1
            }
        }
        
    def should_transition(self, perception: Perception) -> bool:
        """Determine if brain should transition to a new state."""
        self.state_duration = time.time() - self.state_start_time
        
        # Minimum time in state
        min_state_times = {
            MentalState.IDLE: 10.0,
            MentalState.ATTENTIVE: 15.0,
            MentalState.CREATING: 30.0,
            MentalState.REFLECTING: 20.0,
            MentalState.DEEP_SLEEP: 60.0
        }
        
        if self.state_duration < min_state_times.get(self.current_state, 10.0):
            return False
        
        # Check for external triggers
        if self._has_external_trigger(perception):
            return True
        
        # Random chance based on state duration
        transition_prob = min(0.8, self.state_duration / 300.0)  # Max 80% after 5 minutes
        return random.random() < transition_prob
    
    def _has_external_trigger(self, perception: Perception) -> bool:
        """Check for external triggers that force state change."""
        # User starts creating
        if perception.user_activity == "drawing" and self.current_state != MentalState.CREATING:
            return True
        
        # User becomes idle
        if perception.user_activity == "idle" and self.current_state == MentalState.ATTENTIVE:
            return True
        
        # High creative activity
        if perception.capsule_activity and perception.capsule_activity.get('new_creations', 0) > 3:
            if self.current_state != MentalState.REFLECTING:
                return True
        
        return False
    
    def decide_next_state(self, perception: Perception) -> MentalState:
        """Decide which state to transition to."""
        current = self.current_state
        base_probs = self.transition_matrix[current].copy()
        
        # Adjust probabilities based on perception
        self._adjust_for_perception(base_probs, perception)
        
        # Adjust for time of day (simulate circadian rhythm)
        self._adjust_for_time(base_probs)
        
        # Adjust for state duration (prevent stuck states)
        self._adjust_for_state_duration(base_probs)
        
        # Normalize probabilities
        total = sum(base_probs.values())
        if total == 0:
            return current
        
        # Select next state
        rand_val = random.random() * total
        cumulative = 0.0
        
        for state, prob in base_probs.items():
            cumulative += prob
            if rand_val <= cumulative:
                return state
        
        return current
    
    def _adjust_for_perception(self, probs: Dict[MentalState, float], perception: Perception):
        """Adjust transition probabilities based on perception."""
        # User is actively drawing
        if perception.user_activity == "drawing":
            probs[MentalState.CREATING] *= 1.5
            probs[MentalState.ATTENTIVE] *= 1.2
            probs[MentalState.IDLE] *= 0.5
        
        # User is idle
        elif perception.user_activity == "idle":
            probs[MentalState.DEEP_SLEEP] *= 1.3
            probs[MentalState.REFLECTING] *= 1.2
            probs[MentalState.CREATING] *= 0.3
        
        # User seems frustrated
        if 'frustrated' in perception.user_emotional_cues:
            probs[MentalState.REFLECTING] *= 1.4
            probs[MentalState.DEEP_SLEEP] *= 0.7
        
        # Recent creative activity
        if perception.capsule_activity:
            creations = perception.capsule_activity.get('new_creations', 0)
            if creations > 2:
                probs[MentalState.REFLECTING] *= 1.3
    
    def _adjust_for_time(self, probs: Dict[MentalState, float]):
        """Adjust for time of day effects."""
        hour = datetime.now().hour
        
        # Night hours (10 PM - 6 AM) favor deep sleep and reflection
        if 22 <= hour or hour < 6:
            probs[MentalState.DEEP_SLEEP] *= 1.5
            probs[MentalState.REFLECTING] *= 1.3
            probs[MentalState.CREATING] *= 0.7
        
        # Morning hours (6 AM - 10 AM) favor creation
        elif 6 <= hour < 10:
            probs[MentalState.CREATING] *= 1.4
            probs[MentalState.ATTENTIVE] *= 1.2
        
        # Afternoon hours (1 PM - 5 PM) favor reflection
        elif 13 <= hour < 17:
            probs[MentalState.REFLECTING] *= 1.3
            probs[MentalState.ATTENTIVE] *= 1.1
    
    def _adjust_for_state_duration(self, probs: Dict[MentalState, float]):
        """Prevent staying in same state too long."""
        # Reduce probability of staying in same state if been there a while
        same_state_prob = probs.get(self.current_state, 0.0)
        duration_factor = min(2.0, self.state_duration / 600.0)  # Cap at 10 minutes
        probs[self.current_state] = same_state_prob / duration_factor
    
    def transition_to(self, new_state: MentalState):
        """Transition to a new state."""
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        self.state_duration = 0.0
        
        # Record transition
        self.state_history.append({
            'from': old_state.name,
            'to': new_state.name,
            'time': time.time(),
            'duration': self.state_duration
        })
        
        print(f"🧠 Brain state transition: {old_state.name} → {new_state.name}")
    
    def get_state_report(self) -> Dict:
        """Get report on current brain state."""
        return {
            'current_state': self.current_state.name,
            'state_duration': self.state_duration,
            'state_start_time': self.state_start_time,
            'recent_transitions': list(self.state_history)[-5:]
        }

class AttentionManager:
    """Manages attention focus."""
    
    def __init__(self):
        self.current_focus = {'target': 'environment', 'intensity': 0.5}
        self.attention_history = deque(maxlen=50)
        
    def update_attention(self, perception: Perception, mental_state: MentalState) -> Dict:
        """Update attention based on perception and mental state."""
        
        # Determine focus target
        if perception.user_activity == "drawing" and perception.canvas_state:
            target = "canvas"
            intensity = 0.8
        elif perception.timeline_state:
            target = "timeline"
            intensity = 0.7
        elif perception.capsule_activity:
            target = "capsule_library"
            intensity = 0.6
        else:
            target = "environment"
            intensity = 0.4
        
        # Adjust intensity based on mental state
        if mental_state == MentalState.ATTENTIVE:
            intensity *= 1.3
        elif mental_state == MentalState.CREATING:
            intensity *= 1.5
        elif mental_state == MentalState.IDLE:
            intensity *= 0.7
        
        self.current_focus = {'target': target, 'intensity': intensity}
        self.attention_history.append({
            'target': target,
            'intensity': intensity,
            'time': time.time()
        })
        
        return self.current_focus
    
    def get_attention_report(self) -> Dict:
        """Get attention report."""
        return {
            'current_focus': self.current_focus,
            'most_salient_items': {
                'canvas': random.uniform(0.3, 0.9),
                'timeline': random.uniform(0.2, 0.8),
                'capsule_library': random.uniform(0.1, 0.7),
                'user_activity': random.uniform(0.4, 1.0)
            },
            'attention_stability': random.uniform(0.6, 0.95)
        }

class ThoughtGenerator:
    """Generates thoughts based on brain state."""
    
    def __init__(self):
        self.thought_history = deque(maxlen=1000)
        self.thought_templates = {
            ThoughtType.OBSERVATION: [
                "User is {activity}",
                "Canvas has {layers} layers",
                "Timeline is at frame {frame}",
                "Capsule activity detected: {activity}",
                "Emotional cue: {emotion}"
            ],
            ThoughtType.QUESTION: [
                "Why is user {activity}?",
                "What should I create next?",
                "How can I help the user?",
                "Is this the right approach?",
                "What does the user want?"
            ],
            ThoughtType.IDEA: [
                "I should suggest adding more {element}",
                "Let's try a different {approach}",
                "What if we combine {thing1} and {thing2}?",
                "This could be improved by {improvement}",
                "New creative direction: {direction}"
            ],
            ThoughtType.REFLECTION: [
                "Looking back at previous {element}",
                "This reminds me of earlier {thing}",
                "Pattern emerging: {pattern}",
                "Learning from past {experience}",
                "Connecting dots between {concept1} and {concept2}"
            ],
            ThoughtType.CREATION: [
                "Creating new {element}",
                "Building {thing} from scratch",
                "Implementing {feature}",
                "Designing {design}",
                "Developing new {system}"
            ]
        }
    
    def generate_thought(self, perception: Perception, mental_state: MentalState, attention: Dict) -> Optional[Thought]:
        """Generate a thought based on current state."""
        
        # Determine thought type based on mental state
        if mental_state == MentalState.IDLE:
            thought_type = random.choice([ThoughtType.OBSERVATION, ThoughtType.REFLECTION])
        elif mental_state == MentalState.ATTENTIVE:
            thought_type = random.choice([ThoughtType.OBSERVATION, ThoughtType.QUESTION])
        elif mental_state == MentalState.CREATING:
            thought_type = random.choice([ThoughtType.IDEA, ThoughtType.CREATION])
        elif mental_state == MentalState.REFLECTING:
            thought_type = random.choice([ThoughtType.REFLECTION, ThoughtType.QUESTION])
        else:  # DEEP_SLEEP
            thought_type = ThoughtType.REFLECTION
        
        # Get template and fill it
        templates = self.thought_templates.get(thought_type, [])
        if not templates:
            return None
        
        template = random.choice(templates)
        
        # Fill template with data
        content = template.format(
            activity=perception.user_activity,
            layers=perception.canvas_state.get('layers', 0) if perception.canvas_state else 0,
            frame=perception.timeline_state.get('current_frame', 0) if perception.timeline_state else 0,
            emotion=perception.user_emotional_cues[0] if perception.user_emotional_cues else 'neutral',
            element=random.choice(['color', 'shape', 'texture', 'pattern']),
            approach=random.choice(['method', 'technique', 'strategy']),
            thing1=random.choice(['lines', 'colors', 'shapes']),
            thing2=random.choice(['textures', 'patterns', 'forms']),
            improvement=random.choice(['simplifying', 'complicating', 'rearranging']),
            direction=random.choice(['abstract', 'realistic', 'minimalist']),
            thing=random.choice(['work', 'creation', 'project']),
            pattern=random.choice(['repetition', 'contrast', 'harmony']),
            experience=random.choice(['success', 'failure', 'discovery']),
            concept1=random.choice(['form', 'function', 'aesthetics']),
            concept2=random.choice(['utility', 'beauty', 'innovation']),
            feature=random.choice(['animation', 'interaction', 'effect']),
            design=random.choice(['interface', 'layout', 'structure']),
            system=random.choice(['interaction', 'feedback', 'response'])
        )
        
        # Create thought
        thought = Thought(
            type=thought_type,
            content=content,
            emotion=random.uniform(-1.0, 1.0),
            salience=attention.get('intensity', 0.5)
        )
        
        self.thought_history.append(thought)
        return thought
    
    def get_recent_thoughts(self, count: int = 10) -> List[Thought]:
        """Get recent thoughts."""
        return list(self.thought_history)[-count:]
    
    def get_thought_stats(self) -> Dict:
        """Get thought statistics."""
        if not self.thought_history:
            return {'total_thoughts': 0, 'thoughts_per_minute': 0}
        
        total = len(self.thought_history)
        thoughts = list(self.thought_history)
        if len(thoughts) > 1:
            time_span = thoughts[-1].timestamp - thoughts[0].timestamp
            per_minute = (total / max(time_span, 1)) * 60
        else:
            per_minute = 0
        
        return {
            'total_thoughts': total,
            'thoughts_per_minute': per_minute
        }

class ShortTermMemory:
    """Short-term memory storage."""
    
    def __init__(self):
        self.thoughts = deque(maxlen=100)
        self.perceptions = deque(maxlen=50)
        self.associations = defaultdict(list)
    
    def add_thought(self, thought: Thought):
        """Add a thought to memory."""
        self.thoughts.append(thought)
    
    def add_perception(self, perception: Perception):
        """Add a perception to memory."""
        self.perceptions.append(perception)
    
    def get_recent_memories(self, count: int = 10) -> Dict:
        """Get recent memories."""
        return {
            'thoughts': list(self.thoughts)[-count:],
            'perceptions': list(self.perceptions)[-count:],
            'association_count': len(self.associations)
        }

class AutonomousBrain:
    """Main autonomous brain class."""
    
    def __init__(self):
        self.state_manager = BrainStateManager()
        self.attention_manager = AttentionManager()
        self.thought_generator = ThoughtGenerator()
        self.short_term_memory = ShortTermMemory()
        self.emotional_state = EmotionalState.CONTENT
        self.emotional_history = deque(maxlen=100)
        self.thought_stream = deque(maxlen=1000)
        self.start_time = time.time()
        self.uptime = 0.0
        
        print("🧠 Autonomous Brain initialized")
    
    def think_cycle(self):
        """Main thinking cycle."""
        # Update uptime
        self.uptime = time.time() - self.start_time
        
        # Create perception from simulated environment
        perception = self._create_perception()
        
        # Check for state transition
        if self.state_manager.should_transition(perception):
            new_state = self.state_manager.decide_next_state(perception)
            self.state_manager.transition_to(new_state)
        
        # Update attention
        attention = self.attention_manager.update_attention(
            perception,
            self.state_manager.current_state
        )
        
        # Generate thought
        thought = self.thought_generator.generate_thought(
            perception,
            self.state_manager.current_state,
            attention
        )
        
        if thought:
            self.short_term_memory.add_thought(thought)
            self.thought_stream.append(thought)
            
            # Update emotion
            self._update_emotion(thought.emotion)
        
        # Add perception to memory
        self.short_term_memory.add_perception(perception)
        
        return thought
    
    def _create_perception(self) -> Perception:
        """Create simulated perception data."""
        activities = ['idle', 'drawing', 'animating', 'thinking', 'frustrated']
        current_activity = random.choice(activities)
        
        perception = Perception()
        perception.user_activity = current_activity
        
        if current_activity == 'drawing':
            perception.canvas_state = {
                'has_content': True,
                'layers': random.randint(1, 5),
                'brush_size': random.randint(1, 20),
                'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            }
        elif current_activity == 'animating':
            perception.timeline_state = {
                'current_frame': random.randint(1, 24),
                'total_frames': 24,
                'playing': random.choice([True, False])
            }
        
        # Simulate capsule activity occasionally
        if random.random() < 0.3:
            perception.capsule_activity = {
                'new_creations': random.randint(0, 5),
                'active_capsules': random.randint(1, 10),
                'recent_activity': random.choice(['high', 'medium', 'low'])
            }
        
        # Add emotional cues
        if current_activity == 'frustrated':
            perception.user_emotional_cues = ['frustrated', 'hurried']
        elif current_activity == 'drawing':
            perception.user_emotional_cues = ['creative', 'focused']
        elif current_activity == 'idle':
            perception.user_emotional_cues = ['contemplative']
        
        return perception
    
    def _update_emotion(self, thought_emotion: float):
        """Update emotional state based on thought."""
        # Record emotion
        self.emotional_history.append({
            'state': self.emotional_state.name,
            'emotion': thought_emotion,
            'timestamp': time.time()
        })
        
        # Occasionally change emotional state
        if random.random() < 0.05:  # 5% chance per thought
            self.emotional_state = random.choice(list(EmotionalState))
    
    def get_detailed_report(self) -> Dict:
        """Get detailed brain report."""
        thought_stats = self.thought_generator.get_thought_stats()
        
        return {
            'brain_state': self.state_manager.get_state_report(),
            'attention': self.attention_manager.get_attention_report(),
            'thought_stats': thought_stats,
            'emotional_state': self.emotional_state.name,
            'memory': self.short_term_memory.get_recent_memories(5),
            'creative_status': {
                'energy': random.uniform(0.3, 1.0),
                'state': self.emotional_state.name,
                'focus': self.attention_manager.current_focus['target']
            },
            'system_health': {
                'uptime': self.uptime,
                'success_rate': random.uniform(85, 100),
                'memory_usage': len(self.thought_stream),
                'thought_rate': thought_stats.get('thoughts_per_minute', 0)
            }
        }
    
    def shutdown(self):
        """Clean shutdown."""
        print("🧠 Autonomous Brain shutting down...")

# ============================================================================
# PYGAME VISUALIZATION
# ============================================================================

# Initialize PyGame
pygame.init()

# Constants for visualization
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
BACKGROUND_COLOR = (15, 20, 30)
GRID_COLOR = (30, 35, 45)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (100, 180, 255)

# Fonts
FONT_SMALL = pygame.font.SysFont('Arial', 14)
FONT_MEDIUM = pygame.font.SysFont('Arial', 18, bold=True)
FONT_LARGE = pygame.font.SysFont('Arial', 24, bold=True)
FONT_XLARGE = pygame.font.SysFont('Arial', 32, bold=True)

class BrainVisualization:
    """PyGame visualization for the autonomous brain."""
    
    def __init__(self, brain):
        self.brain = brain
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🧠 Autonomous Brain - PyGame Visualization")
        
        # Visualization components
        self.thought_particles = []
        self.attention_nodes = []
        self.emotion_wave = []
        self.state_history = deque(maxlen=100)
        
        # UI State
        self.selected_view = "dashboard"
        self.last_update = time.time()
        self.animation_time = 0
        self.show_thought_stream = True
        self.show_attention_flow = True
        self.show_emotion_wave = True
        
        # Initialize visualization data
        self._initialize_visualization()
    
    def _initialize_visualization(self):
        """Initialize visualization components."""
        focus_areas = [
            "canvas", "timeline", "capsule_library", "user_activity",
            "memory", "connections", "environment", "internal_processing"
        ]
        
        for i, area in enumerate(focus_areas):
            angle = (i / len(focus_areas)) * 2 * math.pi
            radius = 250
            x = SCREEN_WIDTH // 2 + radius * math.cos(angle)
            y = SCREEN_HEIGHT // 2 + radius * math.sin(angle)
            
            self.attention_nodes.append({
                'name': area,
                'x': x,
                'y': y,
                'radius': 20,
                'salience': 0.0,
                'color': (100, 150, 200),
                'pulse': 0.0
            })
    
    def update(self, dt: float):
        """Update visualization."""
        self.animation_time += dt
        
        # Update attention nodes
        attention_report = self.brain.attention_manager.get_attention_report()
        current_focus = attention_report.get('current_focus', {})
        
        for node in self.attention_nodes:
            if node['name'] == current_focus.get('target', 'environment'):
                node['salience'] = current_focus.get('intensity', 0.0)
                node['pulse'] = (node['pulse'] + dt * 2) % (2 * math.pi)
                pulse_intensity = 0.5 + 0.5 * math.sin(node['pulse'])
                node['color'] = (
                    int(100 + 100 * node['salience'] * pulse_intensity),
                    int(150 + 50 * node['salience'] * pulse_intensity),
                    int(200 + 55 * node['salience'] * pulse_intensity)
                )
            else:
                node['salience'] *= 0.95
                node['color'] = (100, 150, 200)
        
        # Generate thought particles
        if random.random() < dt * 2:
            self._create_thought_particle()
        
        # Update existing particles
        for particle in self.thought_particles[:]:
            particle['lifetime'] -= dt
            particle['x'] += particle['vx'] * dt * 50
            particle['y'] += particle['vy'] * dt * 50
            
            # Gravity toward center
            center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            dx = center_x - particle['x']
            dy = center_y - particle['y']
            dist = max(0.1, math.sqrt(dx*dx + dy*dy))
            
            particle['vx'] += dx / dist * dt * 10
            particle['vy'] += dy / dist * dt * 10
            
            # Friction
            particle['vx'] *= 0.99
            particle['vy'] *= 0.99
            
            if particle['lifetime'] <= 0:
                self.thought_particles.remove(particle)
        
        # Update emotion wave
        if random.random() < dt * 10:
            # Get emotion value (convert enum to numeric for wave)
            emotion_value = {
                EmotionalState.CURIOUS: 0.8,
                EmotionalState.EXCITED: 1.0,
                EmotionalState.CONTENT: 0.6,
                EmotionalState.FRUSTRATED: -0.5,
                EmotionalState.INSPIRED: 0.9,
                EmotionalState.TIRED: 0.0,
                EmotionalState.PLAYFUL: 0.7,
                EmotionalState.NEUTRAL: 0.5
            }.get(self.brain.emotional_state, 0.5)
            
            self.emotion_wave.append({
                'value': emotion_value,
                'time': self.animation_time
            })
        
        # Keep wave buffer manageable
        if len(self.emotion_wave) > 200:
            self.emotion_wave = self.emotion_wave[-200:]
        
        # Update state history
        self.state_history.append({
            'state': self.brain.state_manager.current_state.name,
            'time': self.animation_time
        })
    
    def _create_thought_particle(self):
        """Create a new thought particle."""
        thought_types = ['observation', 'question', 'idea', 'reflection', 'creation']
        thought_type = random.choice(thought_types)
        
        colors = {
            'observation': (100, 200, 100),
            'question': (200, 200, 100),
            'idea': (100, 100, 200),
            'reflection': (200, 100, 200),
            'creation': (200, 100, 100)
        }
        
        angle = random.random() * 2 * math.pi
        radius = 400
        x = SCREEN_WIDTH // 2 + radius * math.cos(angle)
        y = SCREEN_HEIGHT // 2 + radius * math.sin(angle)
        
        dx = SCREEN_WIDTH // 2 - x
        dy = SCREEN_HEIGHT // 2 - y
        dist = max(0.1, math.sqrt(dx*dx + dy*dy))
        
        speed = random.uniform(50, 150)
        vx = (dx / dist) * speed + random.uniform(-20, 20)
        vy = (dy / dist) * speed + random.uniform(-20, 20)
        
        self.thought_particles.append({
            'type': thought_type,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'radius': random.uniform(3, 8),
            'color': colors[thought_type],
            'lifetime': random.uniform(3, 8),
            'brightness': 1.0
        })
    
    def render(self):
        """Render the visualization."""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw grid
        self._draw_grid()
        
        # Draw based on selected view
        if self.selected_view == "dashboard":
            self._render_dashboard()
        elif self.selected_view == "thoughts":
            self._render_thoughts_view()
        elif self.selected_view == "attention":
            self._render_attention_view()
        elif self.selected_view == "emotions":
            self._render_emotions_view()
        
        # Always draw brain core
        self._draw_brain_core()
        
        # Draw UI elements
        self._draw_ui()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw background grid."""
        grid_size = 50
        for x in range(0, SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)
    
    def _draw_brain_core(self):
        """Draw the central brain core."""
        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        
        # Draw outer glow
        for radius in range(80, 60, -5):
            alpha = int(50 * (1 - (radius - 60) / 20))
            if alpha > 0:
                color = (100, 180, 255, alpha)
                self._draw_circle_alpha(self.screen, color, (center_x, center_y), radius)
        
        # Draw core
        pygame.draw.circle(self.screen, (80, 160, 240), (center_x, center_y), 60)
        pygame.draw.circle(self.screen, (60, 140, 220), (center_x, center_y), 55)
        
        # Draw brain label
        label = FONT_LARGE.render("🧠", True, (255, 255, 255))
        self.screen.blit(label, (center_x - 15, center_y - 20))
    
    def _draw_circle_alpha(self, surface, color, center, radius):
        """Draw a circle with alpha transparency."""
        shape_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color, (radius, radius), radius)
        surface.blit(shape_surf, (center[0] - radius, center[1] - radius))
    
    def _render_dashboard(self):
        """Render the main dashboard view."""
        # Draw attention nodes
        if self.show_attention_flow:
            for node in self.attention_nodes:
                pygame.draw.circle(
                    self.screen, 
                    node['color'], 
                    (int(node['x']), int(node['y'])), 
                    int(node['radius'] + 5 * node['salience'])
                )
                
                label = FONT_SMALL.render(node['name'].replace('_', ' ').title(), True, TEXT_COLOR)
                text_rect = label.get_rect(center=(node['x'], node['y'] - 25))
                self.screen.blit(label, text_rect)
                
                bar_width = 40
                bar_height = 5
                bar_x = node['x'] - bar_width // 2
                bar_y = node['y'] + 20
                
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                fill_width = int(bar_width * node['salience'])
                pygame.draw.rect(self.screen, node['color'], (bar_x, bar_y, fill_width, bar_height))
        
        # Draw thought particles
        if self.show_thought_stream:
            for particle in self.thought_particles:
                alpha = int(255 * (particle['lifetime'] / 5))
                if alpha > 0:
                    color = particle['color']
                    pygame.draw.circle(
                        self.screen, 
                        color, 
                        (int(particle['x']), int(particle['y'])), 
                        int(particle['radius'])
                    )
        
        # Draw emotion wave
        if self.show_emotion_wave and len(self.emotion_wave) > 10:
            wave_points = []
            wave_height = 100
            wave_y = SCREEN_HEIGHT - 150
            
            for i, point in enumerate(self.emotion_wave):
                x = SCREEN_WIDTH - (len(self.emotion_wave) - i) * 3
                y = wave_y + point['value'] * wave_height
                wave_points.append((x, y))
            
            if len(wave_points) > 1:
                pygame.draw.lines(self.screen, (200, 100, 100), False, wave_points, 2)
        
        # Draw state history
        if len(self.state_history) > 10:
            state_colors = {
                'IDLE': (100, 100, 100),
                'ATTENTIVE': (100, 200, 100),
                'CREATING': (200, 100, 100),
                'REFLECTING': (100, 100, 200),
                'DEEP_SLEEP': (50, 50, 100)
            }
            bar_width = 5
            state_hist_list = list(self.state_history)[-100:]
            for i, state in enumerate(state_hist_list):
                x = 50 + i * bar_width
                color = state_colors.get(state['state'], (100, 100, 100))
                height = 30 + 20 * math.sin(state['time'] * 2)
                pygame.draw.rect(self.screen, color, (x, 50, bar_width - 1, height))
    
    def _render_thoughts_view(self):
        """Render detailed thoughts view."""
        recent_thoughts = self.brain.thought_generator.get_recent_thoughts(10)
        
        for i, thought in enumerate(recent_thoughts):
            card_y = 100 + i * 70
            card_height = 60
            
            pygame.draw.rect(self.screen, (40, 45, 60), (50, card_y, SCREEN_WIDTH - 100, card_height), border_radius=10)
            pygame.draw.rect(self.screen, (60, 65, 80), (50, card_y, SCREEN_WIDTH - 100, card_height), 2, border_radius=10)
            
            type_colors = {
                'OBSERVATION': (100, 200, 100),
                'QUESTION': (200, 200, 100),
                'IDEA': (100, 100, 200),
                'REFLECTION': (200, 100, 200),
                'CREATION': (200, 100, 100)
            }
            
            type_color = type_colors.get(thought.type.name, (150, 150, 150))
            pygame.draw.rect(self.screen, type_color, (55, card_y + 5, 5, card_height - 10))
            
            content_text = FONT_SMALL.render(f"{thought.content[:80]}...", True, TEXT_COLOR)
            self.screen.blit(content_text, (70, card_y + 10))
            
            meta_text = FONT_SMALL.render(f"Type: {thought.type.name} | Emotion: {thought.emotion:.2f}", True, (150, 150, 150))
            self.screen.blit(meta_text, (70, card_y + 35))
    
    def _render_attention_view(self):
        """Render detailed attention view."""
        attention_report = self.brain.attention_manager.get_attention_report()
        current_focus = attention_report.get('current_focus', {})
        salient_items = attention_report.get('most_salient_items', {})
        
        focus_text = FONT_LARGE.render(f"Current Focus: {current_focus.get('target', 'None')}", True, HIGHLIGHT_COLOR)
        self.screen.blit(focus_text, (50, 100))
        
        intensity_text = FONT_MEDIUM.render(f"Intensity: {current_focus.get('intensity', 0):.2f}", True, TEXT_COLOR)
        self.screen.blit(intensity_text, (50, 140))
        
        # Draw attention radar
        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        radar_radius = 200
        
        for r in range(50, int(radar_radius) + 1, 50):
            pygame.draw.circle(self.screen, (50, 50, 50), (center_x, center_y), r, 1)
        
        for i, (target, salience) in enumerate(salient_items.items()):
            angle = (i / max(1, len(salient_items))) * 2 * math.pi
            distance = radar_radius * salience
            
            x = center_x + distance * math.cos(angle)
            y = center_y + distance * math.sin(angle)
            
            point_radius = 5 + 10 * salience
            pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, (int(x), int(y)), int(point_radius))
            
            label = FONT_SMALL.render(target.replace('_', ' ').title(), True, TEXT_COLOR)
            label_x = x + 20 * math.cos(angle)
            label_y = y + 20 * math.sin(angle)
            self.screen.blit(label, (int(label_x), int(label_y)))
    
    def _render_emotions_view(self):
        """Render detailed emotions view."""
        if len(self.brain.emotional_history) > 10:
            emotion_points = []
            emotion_colors = {
                'CURIOUS': (100, 200, 200),
                'EXCITED': (255, 200, 100),
                'CONTENT': (100, 200, 100),
                'FRUSTRATED': (255, 100, 100),
                'INSPIRED': (200, 100, 200),
                'TIRED': (150, 150, 150),
                'PLAYFUL': (255, 200, 200),
                'NEUTRAL': (200, 200, 200)
            }
            
            graph_width = SCREEN_WIDTH - 200
            graph_height = 200
            graph_x = 100
            graph_y = SCREEN_HEIGHT // 2
            
            pygame.draw.rect(self.screen, (30, 35, 45), (graph_x, graph_y - graph_height//2, graph_width, graph_height))
            pygame.draw.rect(self.screen, (50, 55, 65), (graph_x, graph_y - graph_height//2, graph_width, graph_height), 2)
            
            pygame.draw.line(
                self.screen, 
                (100, 100, 100), 
                (graph_x, graph_y), 
                (graph_x + graph_width, graph_y), 
                1
            )
            
            emotional_history_list = list(self.brain.emotional_history)
            for i, emotion_record in enumerate(emotional_history_list[-100:]):
                x = graph_x + (i / 100) * graph_width
                emotion_value = emotion_record.get('emotion', 0)
                y = graph_y - emotion_value * (graph_height // 2)
                
                color = emotion_colors.get(emotion_record.get('state', 'NEUTRAL'), (200, 200, 200))
                pygame.draw.circle(self.screen, color, (int(x), int(y)), 3)
                
                emotion_points.append((x, y))
            
            if len(emotion_points) > 1:
                pygame.draw.lines(self.screen, HIGHLIGHT_COLOR, False, emotion_points, 2)
        
        current_emotion = self.brain.emotional_state.name
        emotion_text = FONT_XLARGE.render(f"Current Emotion: {current_emotion}", True, HIGHLIGHT_COLOR)
        self.screen.blit(emotion_text, (SCREEN_WIDTH // 2 - emotion_text.get_width() // 2, 100))
    
    def _draw_ui(self):
        """Draw UI controls and information."""
        # Draw header
        pygame.draw.rect(self.screen, (25, 30, 40), (0, 0, SCREEN_WIDTH, 60))
        header_text = FONT_XLARGE.render("🧠 Autonomous Brain Visualization", True, (255, 255, 255))
        self.screen.blit(header_text, (20, 15))
        
        # Draw view buttons
        views = ["dashboard", "thoughts", "attention", "emotions"]
        button_width = 150
        button_height = 40
        
        for i, view in enumerate(views):
            x = SCREEN_WIDTH - (len(views) - i) * (button_width + 10)
            y = 10
            
            button_color = (60, 65, 80) if self.selected_view != view else HIGHLIGHT_COLOR
            pygame.draw.rect(self.screen, button_color, (x, y, button_width, button_height), border_radius=5)
            pygame.draw.rect(self.screen, (80, 85, 100), (x, y, button_width, button_height), 2, border_radius=5)
            
            button_text = FONT_MEDIUM.render(view.title(), True, (255, 255, 255))
            self.screen.blit(button_text, (x + button_width//2 - button_text.get_width()//2, y + 10))
        
        # Draw toggle buttons
        toggles = [
            ("Thought Stream", self.show_thought_stream),
            ("Attention Flow", self.show_attention_flow),
            ("Emotion Wave", self.show_emotion_wave)
        ]
        
        for i, (label, state) in enumerate(toggles):
            x = 20
            y = 70 + i * 30
            
            toggle_color = (100, 180, 100) if state else (80, 80, 80)
            pygame.draw.rect(self.screen, toggle_color, (x, y, 120, 25), border_radius=3)
            
            toggle_text = FONT_SMALL.render(label, True, (255, 255, 255))
            self.screen.blit(toggle_text, (x + 5, y + 5))
        
        # Draw stats panel
        stats_panel_x = SCREEN_WIDTH - 250
        stats_panel_y = 70
        
        pygame.draw.rect(self.screen, (25, 30, 40), (stats_panel_x, stats_panel_y, 230, 150), border_radius=10)
        
        report = self.brain.get_detailed_report()
        brain_state = report.get('brain_state', {})
        thought_stats = report.get('thought_stats', {})
        
        stats_lines = [
            f"State: {brain_state.get('current_state', 'UNKNOWN')}",
            f"Uptime: {report.get('system_health', {}).get('uptime', 0):.1f}s",
            f"Thoughts: {thought_stats.get('total_thoughts', 0)}",
            f"Thoughts/min: {thought_stats.get('thoughts_per_minute', 0):.1f}",
            f"Success: {report.get('system_health', {}).get('success_rate', 0):.1f}%",
            f"Energy: {report.get('creative_status', {}).get('energy', 0):.2f}"
        ]
        
        for i, line in enumerate(stats_lines):
            stat_text = FONT_SMALL.render(line, True, TEXT_COLOR)
            self.screen.blit(stat_text, (stats_panel_x + 10, stats_panel_y + 10 + i * 20))
    
    def handle_event(self, event):
        """Handle PyGame events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.selected_view = "dashboard"
            elif event.key == pygame.K_2:
                self.selected_view = "thoughts"
            elif event.key == pygame.K_3:
                self.selected_view = "attention"
            elif event.key == pygame.K_4:
                self.selected_view = "emotions"
            elif event.key == pygame.K_t:
                self.show_thought_stream = not self.show_thought_stream
            elif event.key == pygame.K_a:
                self.show_attention_flow = not self.show_attention_flow
            elif event.key == pygame.K_e:
                self.show_emotion_wave = not self.show_emotion_wave
            elif event.key == pygame.K_ESCAPE:
                return False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            views = ["dashboard", "thoughts", "attention", "emotions"]
            button_width = 150
            button_height = 40
            
            for i, view in enumerate(views):
                x = SCREEN_WIDTH - (len(views) - i) * (button_width + 10)
                y = 10
                
                if x <= event.pos[0] <= x + button_width and y <= event.pos[1] <= y + button_height:
                    self.selected_view = view
            
            toggles = [
                ("Thought Stream", 0),
                ("Attention Flow", 1),
                ("Emotion Wave", 2)
            ]
            
            for label, index in toggles:
                x = 20
                y = 70 + index * 30
                
                if x <= event.pos[0] <= x + 120 and y <= event.pos[1] <= y + 25:
                    if label == "Thought Stream":
                        self.show_thought_stream = not self.show_thought_stream
                    elif label == "Attention Flow":
                        self.show_attention_flow = not self.show_attention_flow
                    elif label == "Emotion Wave":
                        self.show_emotion_wave = not self.show_emotion_wave
        
        return True

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    print("🧠 Starting Autonomous Brain PyGame Visualization")
    print("=" * 50)
    print("Controls:")
    print("  1-4: Switch views (Dashboard, Thoughts, Attention, Emotions)")
    print("  T: Toggle thought stream")
    print("  A: Toggle attention flow")
    print("  E: Toggle emotion wave")
    print("  ESC: Exit")
    
    # Create brain instance
    brain = AutonomousBrain()
    
    # Create visualization
    visualization = BrainVisualization(brain)
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    last_time = time.time()
    
    while running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                if not visualization.handle_event(event):
                    running = False
        
        # Update brain
        brain.think_cycle()
        
        # Update visualization
        visualization.update(dt)
        
        # Render
        visualization.render()
        
        # Cap FPS
        clock.tick(FPS)
    
    # Cleanup
    brain.shutdown()
    pygame.quit()
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()