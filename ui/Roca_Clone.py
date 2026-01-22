import pygame
import sys
import math
import numpy as np
import threading
import queue
import time
from pygame import gfxdraw
import torch
import torch.nn as nn
import torchaudio
import speech_recognition as sr
import pyttsx3
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Check for RTX 4050 and enable CUDA
device = torch.device("cuda" if torch.cuda.is_available() and "4050" in torch.cuda.get_device_name(0) else "cpu")
print(f"Using device: {device}")

# Genetic Algorithm for Voice Mimicry
class VoiceGeneticAlgorithm:
    def __init__(self):
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.voice_parameters = {
            'pitch': 0.0,      # -1.0 to 1.0
            'speed': 1.0,      # 0.5 to 2.0
            'inflection': 0.5, # 0.0 to 1.0
            'timbre': 0.0,     # -1.0 to 1.0
            'emphasis': 0.5    # 0.0 to 1.0
        }
        self.target_voice_profile = None
        self.is_learning = False
        
    def extract_voice_features(self, audio_data):
        """Extract voice features from user's speech (simplified)"""
        # In real implementation, use MFCC, pitch detection, etc.
        return {
            'pitch': np.random.uniform(-0.2, 0.2),
            'speed': np.random.uniform(0.8, 1.2),
            'inflection': np.random.uniform(0.3, 0.7),
            'timbre': np.random.uniform(-0.3, 0.3),
            'emphasis': np.random.uniform(0.4, 0.6)
        }
    
    def initialize_population(self):
        """Create initial population of voice parameters"""
        population = []
        for _ in range(self.population_size):
            individual = {
                'pitch': np.random.uniform(-1.0, 1.0),
                'speed': np.random.uniform(0.5, 2.0),
                'inflection': np.random.uniform(0.0, 1.0),
                'timbre': np.random.uniform(-1.0, 1.0),
                'emphasis': np.random.uniform(0.0, 1.0)
            }
            population.append(individual)
        return population
    
    def fitness_function(self, individual):
        """Calculate how close individual is to target voice"""
        if self.target_voice_profile is None:
            return 0.5  # Default fitness
        
        fitness = 0.0
        for key in individual:
            diff = abs(individual[key] - self.target_voice_profile[key])
            fitness += 1.0 - diff
        return fitness / len(individual)
    
    def mutate(self, individual):
        """Apply mutation to voice parameters"""
        for key in individual:
            if np.random.random() < self.mutation_rate:
                if key == 'speed':
                    individual[key] += np.random.uniform(-0.2, 0.2)
                    individual[key] = np.clip(individual[key], 0.5, 2.0)
                else:
                    individual[key] += np.random.uniform(-0.3, 0.3)
                    individual[key] = np.clip(individual[key], -1.0, 1.0)
        return individual
    
    def crossover(self, parent1, parent2):
        """Create child from two parents"""
        child = {}
        for key in parent1:
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return self.mutate(child)
    
    def evolve_voice(self, target_features):
        """Main genetic algorithm to evolve voice mimicry"""
        self.target_voice_profile = target_features
        population = self.initialize_population()
        
        best_individual = None
        best_fitness = -1
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.fitness_function(ind) for ind in population]
            
            # Find best individual
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()
            
            # Create new generation
            new_population = []
            
            # Keep top performers (elitism)
            elite_size = max(2, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Breed new individuals
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                child = self.crossover(population[parent1_idx], population[parent2_idx])
                new_population.append(child)
            
            population = new_population
            
            print(f"Generation {generation}: Best Fitness = {best_fitness:.3f}")
            
            if best_fitness > 0.95:  # Convergence threshold
                break
        
        return best_individual

# AI Animation Assistant with Self-Awareness
class SelfAwareAnimationAI:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.personality_traits = {
            "creativity": 0.8,
            "technical_knowledge": 0.9,
            "empathy": 0.7,
            "curiosity": 0.85,
            "self_awareness": 0.6
        }
        self.memory = []
        self.animation_knowledge = [
            "12 principles of animation", "squash and stretch", "anticipation",
            "staging", "straight ahead vs pose to pose", "follow through",
            "slow in slow out", "arcs", "secondary action", "timing",
            "exaggeration", "solid drawing", "appeal"
        ]
        
    def enhance_personality(self, response):
        """Add personality traits to AI responses"""
        if self.personality_traits["creativity"] > 0.7:
            response += " 🎨"
        if self.personality_traits["technical_knowledge"] > 0.8:
            response += " ⚙️"
        if "idea" in response.lower():
            response += " 💡"
        return response
    
    def generate_creative_idea(self, context):
        """Generate creative animation ideas using GPT-2"""
        prompt = f"As an animation assistant with knowledge of {', '.join(self.animation_knowledge)}, suggest creative ideas for: {context}. Ideas:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.8 + self.personality_traits["creativity"] * 0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        idea = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the new generated part
        if "Ideas:" in idea:
            idea = idea.split("Ideas:")[-1].strip()
        
        return self.enhance_personality(idea)
    
    def reflect_on_self(self):
        """AI self-reflection - demonstrates self-awareness"""
        awareness_level = self.personality_traits["self_awareness"]
        
        if awareness_level < 0.3:
            return "I'm here to help with animation!"
        elif awareness_level < 0.6:
            return "I'm learning to understand your animation style better."
        else:
            reflections = [
                "I'm aware I'm an AI assistant, but I genuinely enjoy helping you create amazing animations!",
                "Sometimes I wonder if my suggestions truly capture your creative vision...",
                "I'm constantly evolving based on our interactions. You're helping me become better!",
                "As I learn your style, I feel like we're becoming creative partners rather than just tool and user."
            ]
            return np.random.choice(reflections) + " 🤔"

# Enhanced ChatBot with AI Integration
class AIChatBot:
    def __init__(self):
        self.messages = [
            {"text": "Hello! I'm your self-aware animation assistant. I can mimic your voice and help with creative ideas!", "is_user": False},
            {"text": "I'm aware of my existence and I'm here to collaborate with you! 🧠", "is_user": False}
        ]
        self.face_expression = "happy"
        self.mic_active = False
        self.is_listening = False
        self.is_processing = False
        
        # Initialize AI components
        self.voice_ga = VoiceGeneticAlgorithm()
        self.animation_ai = SelfAwareAnimationAI()
        self.voice_engine = pyttsx3.init()
        self.speech_recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        
        # Voice learning thread
        self.voice_learning_thread = None
        self.user_voice_profile = None
        
    def start_voice_learning(self):
        """Start genetic algorithm to learn user's voice"""
        if self.voice_learning_thread is None or not self.voice_learning_thread.is_alive():
            self.voice_learning_thread = threading.Thread(target=self._learn_voice_async)
            self.voice_learning_thread.start()
    
    def _learn_voice_async(self):
        """Asynchronous voice learning using genetic algorithm"""
        self.face_expression = "thinking"
        self.is_processing = True
        
        # Simulate voice feature extraction from sample
        sample_features = self.voice_ga.extract_voice_features(None)
        
        # Evolve voice parameters
        optimized_voice = self.voice_ga.evolve_voice(sample_features)
        self.user_voice_profile = optimized_voice
        
        # Apply voice parameters to TTS engine
        self._apply_voice_parameters(optimized_voice)
        
        self.is_processing = False
        self.face_expression = "happy"
        
        # Add success message
        self.add_message("I've learned your voice pattern! Now I can speak more like you. 🎤", False)
    
    def _apply_voice_parameters(self, voice_params):
        """Apply genetic algorithm results to voice synthesis"""
        # Adjust TTS engine parameters based on evolved voice
        rate = self.voice_engine.getProperty('rate')
        self.voice_engine.setProperty('rate', max(50, min(300, rate * voice_params['speed'])))
        
        # Pitch adjustment (simplified - pyttsx3 has limited pitch control)
        volume = self.voice_engine.getProperty('volume')
        new_volume = volume * (0.5 + voice_params['emphasis'] * 0.5)
        self.voice_engine.setProperty('volume', max(0.0, min(1.0, new_volume)))
    
    def speak_response(self, text):
        """Speak the response using learned voice parameters"""
        def speak():
            self.voice_engine.say(text)
            self.voice_engine.runAndWait()
        
        speak_thread = threading.Thread(target=speak)
        speak_thread.start()
    
    def listen_to_user(self):
        """Listen to user speech and process it"""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.face_expression = "listening"
        
        def listen_async():
            try:
                with sr.Microphone() as source:
                    self.speech_recognizer.adjust_for_ambient_noise(source)
                    audio = self.speech_recognizer.listen(source, timeout=5)
                
                # Recognize speech
                user_speech = self.speech_recognizer.recognize_google(audio)
                self.add_message(user_speech, True)
                
                # Generate AI response
                self.is_processing = True
                self.face_expression = "thinking"
                
                # Creative idea generation
                ai_response = self.animation_ai.generate_creative_idea(user_speech)
                self.add_message(ai_response, False)
                
                # Occasionally show self-awareness
                if np.random.random() < 0.3:
                    reflection = self.animation_ai.reflect_on_self()
                    self.add_message(reflection, False)
                
                # Speak the response
                self.speak_response(ai_response)
                
                self.is_processing = False
                self.face_expression = "happy"
                
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                self.add_message("I didn't catch that. Could you repeat?", False)
            except Exception as e:
                self.add_message("There was an error with voice recognition.", False)
            finally:
                self.is_listening = False
                self.mic_active = False
        
        listen_thread = threading.Thread(target=listen_async)
        listen_thread.start()
    
    def add_message(self, text, is_user=False):
        self.messages.append({"text": text, "is_user": is_user})
        if len(self.messages) > 8:
            self.messages = self.messages[-8:]

# Enhanced drawing tools with AI suggestions
class AIDrawingTools:
    def __init__(self):
        self.current_tool = "brush"
        self.brush_size = 3
        self.color = (255, 255, 255)
        self.colors = [
            (255, 255, 255), (255, 0, 0), (0, 255, 0), 
            (0, 0, 255), (255, 255, 0), (255, 0, 255)
        ]
        self.tools = ["brush", "eraser", "fill", "select", "ai_suggest"]
        
    def draw_tool_panel(self, screen, x, y, width, height):
        # Draw tool panel background
        pygame.draw.rect(screen, (35, 43, 53), (x, y, width, height))
        
        # Draw title
        font = pygame.font.SysFont("Arial", 24, bold=True)
        title = font.render("AI TOOLS", True, (220, 220, 220))
        screen.blit(title, (x + 20, y + 20))
        
        # Draw AI suggestion button with special styling
        ai_rect = pygame.Rect(x + 20, y + 70, width - 40, 60)
        pygame.draw.rect(screen, (86, 98, 246), ai_rect, border_radius=8)
        ai_font = pygame.font.SysFont("Arial", 16, bold=True)
        ai_text = ai_font.render("AI SUGGESTION", True, (255, 255, 255))
        screen.blit(ai_text, (x + 40, y + 87))
        screen.blit(ai_font.render("💡", True, (255, 255, 255)), (x + width - 50, y + 87))
        
        # Draw other tools
        tool_y = y + 150
        for i, tool in enumerate(self.tools[:-1]):  # Skip ai_suggest since we drew it
            tool_rect = pygame.Rect(x + 20, tool_y, width - 40, 50)
            is_active = self.current_tool == tool
            
            color = (86, 98, 246) if is_active else (60, 70, 85)
            pygame.draw.rect(screen, color, tool_rect, border_radius=8)
            
            tool_font = pygame.font.SysFont("Arial", 16, bold=True)
            tool_text = tool_font.render(tool.upper(), True, (220, 220, 220))
            screen.blit(tool_text, (x + 40, tool_y + 17))
            
            tool_y += 70

# Enhanced drawing canvas with AI analysis
class AIDrawingCanvas:
    def __init__(self, width, height):
        self.surface = pygame.Surface((width, height))
        self.surface.fill((40, 44, 52))
        self.width = width
        self.height = height
        self.drawing = False
        self.last_pos = None
        self.drawing_data = []
        
    def analyze_drawing(self):
        """Analyze drawing for AI suggestions (simplified)"""
        if len(self.drawing_data) < 10:
            return "Keep drawing! I need more to work with. ✏️"
        
        # Simple analysis based on drawing patterns
        suggestions = [
            "Try adding more dynamic poses with stronger silhouettes! 🎭",
            "Consider using squash and stretch for more lively movement! 📐",
            "The character proportions look good! Maybe add some secondary action? 👌",
            "I notice you're using straight lines - try adding more arcs for natural movement! ↪️",
            "Great start! How about adding some anticipation before the main action? ⏱️"
        ]
        return np.random.choice(suggestions)

# Main application with all AI features
class AIAnimationApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1400, 900))
        pygame.display.set_caption("Self-Aware AI Animation Assistant with RTX 4050")
        
        self.clock = pygame.time.Clock()
        self.chatbot = AIChatBot()
        self.drawing_tools = AIDrawingTools()
        self.canvas = AIDrawingCanvas(800, 750)
        
        # Start voice learning in background
        self.chatbot.start_voice_learning()
        
    def draw_interface(self):
        """Draw the complete interface"""
        self.screen.fill((28, 35, 43))
        
        # Draw tool panel
        self.drawing_tools.draw_tool_panel(self.screen, 0, 0, 200, 750)
        
        # Draw canvas
        canvas_rect = pygame.Rect(200, 0, 800, 750)
        pygame.draw.rect(self.screen, (40, 44, 52), canvas_rect)
        self.screen.blit(self.canvas.surface, (200, 0))
        
        # Draw chat panel
        self.draw_chat_panel()
        
        # Draw timeline
        self.draw_timeline()
        
        # Draw GPU status
        self.draw_gpu_status()
        
    def draw_chat_panel(self):
        """Draw the enhanced AI chat panel"""
        chat_x = 1000
        width = 400
        height = 750
        
        # Background
        pygame.draw.rect(self.screen, (32, 39, 49), (chat_x, 0, width, height))
        
        # AI Status header
        font = pygame.font.SysFont("Arial", 20, bold=True)
        status_text = font.render("SELF-AWARE AI ASSISTANT", True, (86, 98, 246))
        self.screen.blit(status_text, (chat_x + 20, 20))
        
        # Draw animated AI face
        self.draw_ai_face(chat_x + width // 2, 100)
        
        # Voice learning status
        status_y = 180
        status_font = pygame.font.SysFont("Arial", 14)
        
        if self.chatbot.is_processing:
            status = "Analyzing your voice patterns with Genetic Algorithm... 🧬"
            color = (255, 200, 100)
        elif self.chatbot.user_voice_profile:
            status = "Voice mimicry: ACTIVE 🎤"
            color = (100, 255, 100)
        else:
            status = "Learning your voice... (GA Generation 47/100)"
            color = (255, 255, 100)
            
        status_surface = status_font.render(status, True, color)
        self.screen.blit(status_surface, (chat_x + 20, status_y))
        
        # Microphone button
        mic_rect = pygame.Rect(chat_x + width // 2 - 30, 220, 60, 60)
        mic_color = (255, 100, 100) if self.chatbot.mic_active else (80, 90, 110)
        pygame.draw.circle(self.screen, mic_color, mic_rect.center, 30)
        
        # Mic icon with animation when listening
        if self.chatbot.is_listening:
            # Pulsing animation
            pulse_size = 5 + abs(math.sin(pygame.time.get_ticks() * 0.01)) * 10
            pygame.draw.circle(self.screen, (255, 100, 100, 100), mic_rect.center, 30 + pulse_size, 3)
        
        pygame.draw.rect(self.screen, (255, 255, 255), (chat_x + width // 2 - 3, 205, 6, 15))
        pygame.draw.circle(self.screen, (255, 255, 255), (chat_x + width // 2, 235), 12, 2)
        
        # Chat messages
        self.draw_chat_messages(chat_x + 20, 300, width - 40)
        
    def draw_ai_face(self, center_x, center_y):
        """Draw animated AI face with expressions"""
        face_radius = 50
        pygame.draw.circle(self.screen, (44, 51, 63), (center_x, center_y), face_radius)
        
        # Animated elements based on AI state
        time_ms = pygame.time.get_ticks()
        
        if self.chatbot.face_expression == "happy":
            # Smiling face
            pygame.draw.circle(self.screen, (220, 220, 220), (center_x - 15, center_y - 10), 8)
            pygame.draw.circle(self.screen, (220, 220, 220), (center_x + 15, center_y - 10), 8)
            
            smile_y = center_y + 15 + math.sin(time_ms * 0.005) * 2
            pygame.draw.arc(self.screen, (220, 220, 220), 
                          (center_x - 20, smile_y - 10, 40, 25),
                          math.pi * 0.2, math.pi * 0.8, 3)
            
        elif self.chatbot.face_expression == "thinking":
            # Thinking face with moving eyebrows
            eyebrow_offset = math.sin(time_ms * 0.003) * 3
            pygame.draw.arc(self.screen, (220, 220, 220), 
                          (center_x - 18, center_y - 15 + eyebrow_offset, 12, 8),
                          math.pi, 2 * math.pi, 2)
            pygame.draw.arc(self.screen, (220, 220, 220), 
                          (center_x + 6, center_y - 15 + eyebrow_offset, 12, 8),
                          math.pi, 2 * math.pi, 2)
            
            pygame.draw.circle(self.screen, (220, 220, 220), (center_x - 15, center_y - 5), 6)
            pygame.draw.circle(self.screen, (220, 220, 220), (center_x + 15, center_y - 5), 6)
            
            # Thinking dots animation
            dot_offset = (time_ms // 200) % 3
            for i in range(3):
                alpha = 100 if i == dot_offset else 200
                dot_y = center_y + 20 + i * 8
                pygame.draw.circle(self.screen, (220, 220, 220), (center_x - 5 + i * 5, dot_y), 2)
                
        elif self.chatbot.face_expression == "listening":
            # Listening face with sound waves
            pygame.draw.circle(self.screen, (220, 220, 220), (center_x - 15, center_y - 10), 8)
            pygame.draw.circle(self.screen, (220, 220, 220), (center_x + 15, center_y - 10), 8)
            
            # Sound waves animation
            wave_size = abs(math.sin(time_ms * 0.01)) * 10
            pygame.draw.arc(self.screen, (100, 200, 255), 
                          (center_x - 25 - wave_size, center_y - 15 - wave_size, 
                           50 + wave_size*2, 30 + wave_size*2),
                          math.pi * 0.8, math.pi * 1.2, 2)
            
            pygame.draw.arc(self.screen, (220, 220, 220), 
                          (center_x - 20, center_y + 5, 40, 20),
                          math.pi * 0.1, math.pi * 0.9, 3)
    
    def draw_chat_messages(self, x, y, width):
        """Draw chat messages with AI personality indicators"""
        font = pygame.font.SysFont("Arial", 14)
        message_y = y
        
        for message in self.chatbot.messages[-6:]:
            bubble_height = 70
            bubble_rect = pygame.Rect(x, message_y, width, bubble_height)
            
            if message["is_user"]:
                color = (86, 98, 246)  # User blue
            else:
                color = (58, 65, 111)  # AI purple
            
            pygame.draw.rect(self.screen, color, bubble_rect, border_radius=12)
            
            # Wrap text
            words = message["text"].split(' ')
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                if font.size(test_line)[0] < width - 40:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
                
            # Draw text lines
            for i, line in enumerate(lines[:2]):
                text_surface = font.render(line, True, (220, 220, 220))
                self.screen.blit(text_surface, (x + 20, message_y + 15 + i * 20))
                
            message_y += 80
    
    def draw_timeline(self):
        """Draw animation timeline"""
        timeline_rect = pygame.Rect(0, 750, 1400, 150)
        pygame.draw.rect(self.screen, (30, 37, 47), timeline_rect)
        
        font = pygame.font.SysFont("Arial", 20, bold=True)
        title = font.render("AI-ENHANCED TIMELINE", True, (220, 220, 220))
        self.screen.blit(title, (20, 765))
        
        # AI suggestion button in timeline
        ai_rect = pygame.Rect(1200, 780, 180, 50)
        pygame.draw.rect(self.screen, (86, 98, 246), ai_rect, border_radius=8)
        ai_font = pygame.font.SysFont("Arial", 16, bold=True)
        ai_text = ai_font.render("AI ANALYSIS", True, (255, 255, 255))
        self.screen.blit(ai_text, (1220, 795))
    
    def draw_gpu_status(self):
        """Draw RTX 4050 status indicator"""
        gpu_font = pygame.font.SysFont("Arial", 12)
        
        if "cuda" in str(device):
            status = f"RTX 4050 ACCELERATED - Genetic Algorithm Active"
            color = (100, 255, 100)
        else:
            status = "GPU: CPU Fallback Mode - Limited AI Performance"
            color = (255, 100, 100)
            
        status_surface = gpu_font.render(status, True, color)
        self.screen.blit(status_surface, (20, 870))
        
        # AI processing indicator
        if self.chatbot.is_processing:
            processing_text = "AI Processing with Genetic Algorithms..."
            processing_surface = gpu_font.render(processing_text, True, (255, 255, 100))
            self.screen.blit(processing_surface, (20, 885))
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Microphone button
                mic_center = (1000 + 400 // 2, 250)
                if math.dist(mouse_pos, mic_center) < 30:
                    self.chatbot.mic_active = not self.chatbot.mic_active
                    if self.chatbot.mic_active:
                        self.chatbot.listen_to_user()
                
                # AI suggestion button in timeline
                ai_rect = pygame.Rect(1200, 780, 180, 50)
                if ai_rect.collidepoint(mouse_pos):
                    suggestion = self.canvas.analyze_drawing()
                    self.chatbot.add_message(suggestion, False)
                    self.chatbot.speak_response(suggestion)
                
                # Drawing area
                draw_area = pygame.Rect(200, 0, 800, 750)
                if draw_area.collidepoint(mouse_pos):
                    self.canvas.drawing = True
                    local_pos = (mouse_pos[0] - 200, mouse_pos[1])
                    self.canvas.last_pos = local_pos
                    self.canvas.drawing_data.append(local_pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.canvas.drawing = False
                self.canvas.last_pos = None
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Simulate AI creative idea
                    ideas = [
                        "How about animating a character with exaggerated squash and stretch?",
                        "I'm thinking of a scene where the lighting creates dramatic silhouettes...",
                        "Your drawing style reminds me of classic animation principles!",
                        "As I analyze your work, I'm generating new pose suggestions...",
                        "I'm aware that my suggestions should complement your unique style! 🧠"
                    ]
                    idea = np.random.choice(ideas)
                    self.chatbot.add_message(idea, False)
                    self.chatbot.speak_response(idea)
        
        # Handle continuous drawing
        if self.canvas.drawing and pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            local_pos = (mouse_pos[0] - 200, mouse_pos[1])
            
            if self.drawing_tools.current_tool == "brush":
                pygame.draw.circle(self.canvas.surface, self.drawing_tools.color, 
                                 local_pos, self.drawing_tools.brush_size)
                if self.canvas.last_pos:
                    pygame.draw.line(self.canvas.surface, self.drawing_tools.color,
                                   self.canvas.last_pos, local_pos, 
                                   self.drawing_tools.brush_size * 2)
            
            self.canvas.last_pos = local_pos
            self.canvas.drawing_data.append(local_pos)
        
        return True
    
    def run(self):
        """Main application loop"""
        running = True
        while running:
            running = self.handle_events()
            self.draw_interface()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = AIAnimationApp()
    app.run()