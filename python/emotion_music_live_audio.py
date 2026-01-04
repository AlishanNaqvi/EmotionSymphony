"""
Emotion Symphony - Live Real-Time Application with AUDIO
=========================================================
Combines real emotion detection with live music generation and audio playback

Usage: python emotion_music_live_audio.py
"""

import cv2
import numpy as np
from tensorflow import keras
import threading
import time
from collections import deque
import random

# Audio imports
try:
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("WARNING: pygame not installed. Install with: pip install pygame")

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion to music mapping
EMOTION_MUSIC = {
    'happy': {
        'tempo': 140, 
        'scale': 'Major', 
        'key': 'C', 
        'character': 'Upbeat & Bright',
        'base_freq': 523.25,  # C5
        'chord': [523.25, 659.25, 783.99]  # C major chord
    },
    'sad': {
        'tempo': 70, 
        'scale': 'Minor', 
        'key': 'A', 
        'character': 'Melancholic & Slow',
        'base_freq': 440.00,  # A4
        'chord': [440.00, 523.25, 659.25]  # A minor chord
    },
    'angry': {
        'tempo': 160, 
        'scale': 'Minor', 
        'key': 'E', 
        'character': 'Intense & Aggressive',
        'base_freq': 329.63,  # E4
        'chord': [329.63, 392.00, 493.88]  # E minor chord
    },
    'fear': {
        'tempo': 90, 
        'scale': 'Diminished', 
        'key': 'F#', 
        'character': 'Tense & Dark',
        'base_freq': 369.99,  # F#4
        'chord': [369.99, 440.00, 523.25]  # Diminished chord
    },
    'surprise': {
        'tempo': 130, 
        'scale': 'Lydian', 
        'key': 'D', 
        'character': 'Playful & Bright',
        'base_freq': 587.33,  # D5
        'chord': [587.33, 739.99, 880.00]  # D major chord
    },
    'neutral': {
        'tempo': 100, 
        'scale': 'Major', 
        'key': 'G', 
        'character': 'Balanced & Steady',
        'base_freq': 392.00,  # G4
        'chord': [392.00, 493.88, 587.33]  # G major chord
    },
    'disgust': {
        'tempo': 85, 
        'scale': 'Minor', 
        'key': 'B', 
        'character': 'Dissonant & Uneasy',
        'base_freq': 493.88,  # B4
        'chord': [493.88, 587.33, 739.99]  # B minor chord
    }
}

# Musical scales (MIDI note offsets from root)
SCALES = {
    'Major': [0, 2, 4, 5, 7, 9, 11],
    'Minor': [0, 2, 3, 5, 7, 8, 10],
    'Diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'Lydian': [0, 2, 4, 6, 7, 9, 11]
}

class AudioSynthesizer:
    """Generate and play audio tones"""
    
    def __init__(self):
        """Initialize pygame mixer for audio"""
        if AUDIO_AVAILABLE:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.sample_rate = 22050
        else:
            self.sample_rate = None
    
    def generate_tone(self, frequency, duration=0.3, volume=0.3):
        """Generate a sine wave tone"""
        if not AUDIO_AVAILABLE:
            return None
        
        num_samples = int(self.sample_rate * duration)
        samples = np.arange(num_samples)
        
        # Generate sine wave
        wave = np.sin(2 * np.pi * frequency * samples / self.sample_rate)
        
        # Apply envelope (fade in/out to avoid clicks)
        envelope = np.ones(num_samples)
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave *= envelope
        
        # Apply volume
        wave *= volume
        
        # Convert to 16-bit integer
        wave = (wave * 32767).astype(np.int16)
        
        # Make stereo
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def generate_chord(self, frequencies, duration=1.0, volume=0.2):
        """Generate a chord from multiple frequencies"""
        if not AUDIO_AVAILABLE:
            return None
        
        num_samples = int(self.sample_rate * duration)
        samples = np.arange(num_samples)
        
        # Combine multiple frequencies
        wave = np.zeros(num_samples)
        for freq in frequencies:
            wave += np.sin(2 * np.pi * freq * samples / self.sample_rate)
        
        # Normalize
        wave /= len(frequencies)
        
        # Apply envelope
        envelope = np.ones(num_samples)
        fade_samples = int(0.02 * self.sample_rate)  # 20ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        wave *= envelope
        
        # Apply volume
        wave *= volume
        
        # Convert to 16-bit integer
        wave = (wave * 32767).astype(np.int16)
        
        # Make stereo
        stereo_wave = np.column_stack((wave, wave))
        
        return pygame.sndarray.make_sound(stereo_wave)
    
    def play_melody(self, emotion):
        """Play a simple melody for the emotion"""
        if not AUDIO_AVAILABLE:
            print("⚠️  Audio not available. Install pygame: pip install pygame")
            return
        
        music_info = EMOTION_MUSIC[emotion]
        base_freq = music_info['base_freq']
        tempo = music_info['tempo']
        
        # Calculate note duration based on tempo
        beat_duration = 60.0 / tempo  # seconds per beat
        note_duration = beat_duration / 2  # eighth notes
        
        # Generate a short melody (8 notes)
        melody_pattern = [0, 2, 4, 5, 4, 2, 0, 0]  # Scale degrees
        
        print(f"\n🎵 Playing {emotion.upper()} melody...")
        
        for degree in melody_pattern:
            # Calculate frequency (simple octave-based scaling)
            freq = base_freq * (2 ** (degree / 12))
            
            # Generate and play tone
            tone = self.generate_tone(freq, note_duration, volume=0.25)
            if tone:
                tone.play()
                time.sleep(note_duration * 0.9)  # Small gap between notes
        
        # Play final chord
        chord_sound = self.generate_chord(music_info['chord'], duration=1.0, volume=0.15)
        if chord_sound:
            chord_sound.play()
            time.sleep(1.0)
    
    def play_chord(self, emotion):
        """Play just a chord for the emotion (faster)"""
        if not AUDIO_AVAILABLE:
            print("⚠️  Audio not available. Install pygame: pip install pygame")
            return
        
        music_info = EMOTION_MUSIC[emotion]
        
        print(f"\n🎵 Playing {emotion.upper()} chord...")
        print(f"   Tempo: {music_info['tempo']} BPM")
        print(f"   Key: {music_info['key']} {music_info['scale']}")
        print(f"   Character: {music_info['character']}")
        
        # Play chord
        chord_sound = self.generate_chord(music_info['chord'], duration=1.5, volume=0.2)
        if chord_sound:
            chord_sound.play()
            time.sleep(1.5)

class EmotionMusicLive:
    """Real-time emotion detection with live music generation"""
    
    def __init__(self, model_path):
        """Initialize the application"""
        print("🎵 Emotion Symphony - Live Edition with AUDIO")
        print("=" * 50)
        
        # Load the trained model
        print("Loading emotion detection model...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Initialize audio synthesizer
        print("Initializing audio system...")
        self.audio = AudioSynthesizer()
        if AUDIO_AVAILABLE:
            print("✓ Audio system ready")
        else:
            print("⚠️  Audio not available - install pygame")
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam!")
        print("✓ Webcam initialized")
        
        # State variables
        self.current_emotion = 'neutral'
        self.emotion_history = deque(maxlen=10)
        self.last_music_emotion = None
        self.music_playing = False
        self.music_thread = None
        self.auto_music = False
        
        # Visual settings
        self.window_name = "🎵 Emotion Symphony - Live Detection + Audio"
        self.colors = {
            'happy': (0, 255, 255),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'fear': (128, 0, 128),
            'surprise': (0, 165, 255),
            'neutral': (255, 255, 255),
            'disgust': (0, 128, 0)
        }
        
        print("✓ Setup complete!")
        print("\n" + "=" * 50)
        print("Controls:")
        print("  SPACE - Play music for current emotion")
        print("  'M'   - Play full melody (slower)")
        print("  'C'   - Play chord only (faster)")
        print("  'A'   - Toggle auto-music mode")
        print("  'Q'   - Quit")
        print("=" * 50 + "\n")
    
    def preprocess_face(self, face):
        """Preprocess face for model input"""
        face = cv2.resize(face, (48, 48))
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        return face
    
    def detect_emotion(self, frame):
        """Detect emotion from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None, None, None
        
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        face_roi = gray[y:y+h, x:x+w]
        processed_face = self.preprocess_face(face_roi)
        
        predictions = self.model.predict(processed_face, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        emotion = EMOTIONS[emotion_idx]
        
        return emotion, confidence, (x, y, w, h), predictions
    
    def play_music_threaded(self, emotion, mode='chord'):
        """Play music in separate thread"""
        if self.music_playing:
            return
        
        self.music_playing = True
        self.last_music_emotion = emotion
        
        if mode == 'melody':
            self.audio.play_melody(emotion)
        else:
            self.audio.play_chord(emotion)
        
        self.music_playing = False
    
    def draw_ui(self, frame, emotion, confidence, face_box, predictions):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw face box
        if face_box is not None:
            (x, y, fw, fh) = face_box
            color = self.colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 3)
            
            label = f"{emotion.upper()}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw emotion bars
        bar_height = h // len(EMOTIONS)
        for i, emo in enumerate(EMOTIONS):
            y_pos = i * bar_height
            conf = predictions[i] if predictions is not None else 0
            bar_width = int(200 * conf)
            
            cv2.rectangle(frame, (10, y_pos + 10), (210, y_pos + bar_height - 10),
                         (40, 40, 40), -1)
            
            color = self.colors.get(emo, (255, 255, 255))
            cv2.rectangle(frame, (10, y_pos + 10), (10 + bar_width, y_pos + bar_height - 10),
                         color, -1)
            
            cv2.putText(frame, f"{emo.upper()}", (15, y_pos + bar_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{conf*100:.0f}%", (120, y_pos + bar_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw music info
        if self.last_music_emotion:
            music_info = EMOTION_MUSIC[self.last_music_emotion]
            y_offset = h - 150
            
            cv2.rectangle(frame, (w - 320, y_offset), (w - 10, h - 10),
                         (40, 40, 40), -1)
            
            cv2.putText(frame, "🎵 CURRENT MUSIC", (w - 300, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {self.last_music_emotion.upper()}",
                       (w - 300, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Tempo: {music_info['tempo']} BPM",
                       (w - 300, y_offset + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Key: {music_info['key']} {music_info['scale']}",
                       (w - 300, y_offset + 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw status
        if self.music_playing:
            status = "🎵 PLAYING..."
            color = (0, 255, 0)
        elif self.auto_music:
            status = "🎵 AUTO-MUSIC ON (press A to disable)"
            color = (0, 255, 255)
        else:
            status = "Press SPACE for chord, M for melody"
            color = (255, 255, 255)
        
        cv2.putText(frame, status, (w // 2 - 250, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                result = self.detect_emotion(frame)
                
                if result[0] is not None:
                    emotion, confidence, face_box, predictions = result
                    
                    self.emotion_history.append(emotion)
                    
                    if len(self.emotion_history) > 5:
                        from collections import Counter
                        emotion_counts = Counter(self.emotion_history)
                        self.current_emotion = emotion_counts.most_common(1)[0][0]
                    else:
                        self.current_emotion = emotion
                    
                    # Auto-music
                    if self.auto_music and self.current_emotion != self.last_music_emotion:
                        if not self.music_playing:
                            self.music_thread = threading.Thread(
                                target=self.play_music_threaded,
                                args=(self.current_emotion, 'chord')
                            )
                            self.music_thread.start()
                    
                    frame = self.draw_ui(frame, emotion, confidence, face_box, predictions)
                else:
                    cv2.putText(frame, "No face detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow(self.window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') or key == ord('c'):
                    # Spacebar or C - play chord
                    if not self.music_playing:
                        self.music_thread = threading.Thread(
                            target=self.play_music_threaded,
                            args=(self.current_emotion, 'chord')
                        )
                        self.music_thread.start()
                elif key == ord('m'):
                    # M - play full melody
                    if not self.music_playing:
                        self.music_thread = threading.Thread(
                            target=self.play_music_threaded,
                            args=(self.current_emotion, 'melody')
                        )
                        self.music_thread.start()
                elif key == ord('a'):
                    # Toggle auto-music
                    self.auto_music = not self.auto_music
                    status = "ON" if self.auto_music else "OFF"
                    print(f"\n🎵 Auto-music: {status}\n")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if AUDIO_AVAILABLE:
                pygame.mixer.quit()
            print("\n✓ Application closed")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = '../models/best_emotion_model.h5'
    
    try:
        app = EmotionMusicLive(model_path)
        app.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nUsage: python emotion_music_live_audio.py [model_path]")
        sys.exit(1)

if __name__ == "__main__":
    main()
