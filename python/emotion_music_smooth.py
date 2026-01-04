"""
Emotion Symphony - Live with SMOOTH Music Transitions
======================================================
Enhanced version with smooth audio transitions and better sound quality

Usage: python emotion_music_smooth.py
"""

import cv2
import numpy as np
from tensorflow import keras
import threading
import time
from collections import deque, Counter
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

# Emotion to music mapping (enhanced with more frequencies)
EMOTION_MUSIC = {
    'happy': {
        'tempo': 140, 
        'scale': 'Major', 
        'key': 'C', 
        'character': 'Upbeat & Bright',
        'base_freq': 523.25,  # C5
        'chord': [523.25, 659.25, 783.99],  # C major
        'color': (0, 255, 255)
    },
    'sad': {
        'tempo': 70, 
        'scale': 'Minor', 
        'key': 'A', 
        'character': 'Melancholic & Slow',
        'base_freq': 440.00,  # A4
        'chord': [440.00, 523.25, 659.25],  # A minor
        'color': (255, 0, 0)
    },
    'angry': {
        'tempo': 160, 
        'scale': 'Minor', 
        'key': 'E', 
        'character': 'Intense & Aggressive',
        'base_freq': 329.63,  # E4
        'chord': [329.63, 392.00, 493.88],  # E minor
        'color': (0, 0, 255)
    },
    'fear': {
        'tempo': 90, 
        'scale': 'Diminished', 
        'key': 'F#', 
        'character': 'Tense & Dark',
        'base_freq': 369.99,  # F#4
        'chord': [369.99, 440.00, 523.25],  # Diminished
        'color': (128, 0, 128)
    },
    'surprise': {
        'tempo': 130, 
        'scale': 'Lydian', 
        'key': 'D', 
        'character': 'Playful & Bright',
        'base_freq': 587.33,  # D5
        'chord': [587.33, 739.99, 880.00],  # D major
        'color': (0, 165, 255)
    },
    'neutral': {
        'tempo': 100, 
        'scale': 'Major', 
        'key': 'G', 
        'character': 'Balanced & Steady',
        'base_freq': 392.00,  # G4
        'chord': [392.00, 493.88, 587.33],  # G major
        'color': (255, 255, 255)
    },
    'disgust': {
        'tempo': 85, 
        'scale': 'Minor', 
        'key': 'B', 
        'character': 'Dissonant & Uneasy',
        'base_freq': 493.88,  # B4
        'chord': [493.88, 587.33, 739.99],  # B minor
        'color': (0, 128, 0)
    }
}

class SmoothAudioSynthesizer:
    """Generate smooth, musical audio with transitions"""
    
    def __init__(self):
        """Initialize audio system"""
        if AUDIO_AVAILABLE:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self.sample_rate = 44100
            self.last_emotion = None
        else:
            self.sample_rate = None
    
    def generate_smooth_tone(self, frequency, duration=0.5, volume=0.25):
        """Generate tone with smooth ADSR envelope and harmonics"""
        if not AUDIO_AVAILABLE:
            return None
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Add harmonics for richer sound
        wave = np.sin(2 * np.pi * frequency * t)  # Fundamental
        wave += 0.3 * np.sin(4 * np.pi * frequency * t)  # 2nd harmonic
        wave += 0.15 * np.sin(6 * np.pi * frequency * t)  # 3rd harmonic
        wave += 0.08 * np.sin(8 * np.pi * frequency * t)  # 4th harmonic
        wave /= 1.53  # Normalize
        
        # ADSR Envelope
        attack = int(0.08 * self.sample_rate)
        release = int(0.25 * self.sample_rate)
        
        envelope = np.ones(num_samples)
        envelope[:attack] = np.linspace(0, 1, attack) ** 2
        envelope[-release:] = np.linspace(1, 0, release) ** 2
        
        wave *= envelope * volume
        wave = (wave * 32767).astype(np.int16)
        
        return pygame.sndarray.make_sound(np.column_stack((wave, wave)))
    
    def generate_smooth_chord(self, frequencies, duration=2.5, volume=0.18):
        """Generate rich chord with smooth transitions"""
        if not AUDIO_AVAILABLE:
            return None
        
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        wave = np.zeros(num_samples)
        
        # Add each note with slight detuning for warmth
        for i, freq in enumerate(frequencies):
            detune = 1.0 + (0.0003 * (i - 1))  # Slight chorus effect
            wave += np.sin(2 * np.pi * freq * detune * t)
            wave += 0.25 * np.sin(4 * np.pi * freq * t)  # Harmonic
        
        wave /= (len(frequencies) * 1.25)
        
        # Smooth ADSR envelope
        attack = int(0.2 * self.sample_rate)
        decay = int(0.4 * self.sample_rate)
        release = int(0.5 * self.sample_rate)
        
        envelope = np.ones(num_samples)
        envelope[:attack] = np.linspace(0, 1, attack) ** 2
        
        if attack + decay < num_samples - release:
            envelope[attack:attack + decay] = np.linspace(1, 0.75, decay)
            envelope[attack + decay:-release] = 0.75
        
        envelope[-release:] = np.linspace(envelope[-release], 0, release) ** 1.5
        
        wave *= envelope * volume
        wave = (wave * 32767).astype(np.int16)
        
        # Stereo width
        left = wave
        right = np.roll(wave, int(0.002 * self.sample_rate))
        
        return pygame.sndarray.make_sound(np.column_stack((left, right)))
    
    def generate_transition(self, from_emotion, to_emotion):
        """Smooth gliding transition between emotions"""
        if not AUDIO_AVAILABLE or from_emotion is None:
            return None
        
        from_chord = EMOTION_MUSIC[from_emotion]['chord']
        to_chord = EMOTION_MUSIC[to_emotion]['chord']
        
        duration = 1.8
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        wave = np.zeros(num_samples)
        
        # Glide each voice smoothly
        for from_freq, to_freq in zip(from_chord, to_chord):
            # Exponential frequency glide
            freq_curve = from_freq * (to_freq / from_freq) ** (t / duration)
            phase = np.cumsum(2 * np.pi * freq_curve / self.sample_rate)
            wave += np.sin(phase)
        
        wave /= len(from_chord)
        
        # Crossfade envelope
        envelope = np.ones(num_samples)
        fade = int(0.4 * self.sample_rate)
        envelope[:fade] = np.linspace(0, 1, fade) ** 2
        envelope[-fade:] = np.linspace(1, 0, fade) ** 2
        
        wave *= envelope * 0.16
        wave = (wave * 32767).astype(np.int16)
        
        return pygame.sndarray.make_sound(np.column_stack((wave, wave)))
    
    def play_smooth_melody(self, emotion):
        """Play flowing melody with overlapping notes"""
        if not AUDIO_AVAILABLE:
            return
        
        info = EMOTION_MUSIC[emotion]
        base_freq = info['base_freq']
        tempo = info['tempo']
        beat_duration = 60.0 / tempo
        
        # Emotion-specific melodies
        melodies = {
            'happy': [0, 2, 4, 5, 7, 5, 4, 2, 0],
            'sad': [0, -2, -3, -5, -7, -5, -3, -2, 0],
            'angry': [0, 1, 0, 1, 0, -1, 0, 1, 0],
            'surprise': [0, 7, 4, 7, 0, 7, 4, 0],
            'neutral': [0, 2, 0, 2, 4, 2, 0, 2, 0],
            'fear': [0, 1, 3, 1, 0, 3, 1, 0],
            'disgust': [0, 2, 3, 2, 1, 0, -1, 0]
        }
        
        pattern = melodies.get(emotion, [0, 2, 4, 5, 4, 2, 0])
        
        print(f"\n🎵 Playing {emotion.upper()} melody (smooth & flowing)...")
        
        # Play with overlap
        for degree in pattern:
            freq = base_freq * (2 ** (degree / 12))
            tone = self.generate_smooth_tone(freq, beat_duration * 1.2, 0.22)
            if tone:
                tone.play()
                time.sleep(beat_duration * 0.8)
        
        # Ending chord
        time.sleep(0.2)
        chord = self.generate_smooth_chord(info['chord'], 3.0, 0.15)
        if chord:
            chord.play()
            time.sleep(3.0)
    
    def play_smooth_chord(self, emotion, with_transition=True):
        """Play chord with optional transition"""
        if not AUDIO_AVAILABLE:
            return
        
        info = EMOTION_MUSIC[emotion]
        
        print(f"\n🎵 {emotion.upper()}: {info['character']}")
        print(f"   {info['key']} {info['scale']} | {info['tempo']} BPM")
        
        # Transition if emotion changed
        if with_transition and self.last_emotion and self.last_emotion != emotion:
            print(f"   ↪ Gliding from {self.last_emotion.upper()}...")
            trans = self.generate_transition(self.last_emotion, emotion)
            if trans:
                trans.play()
                time.sleep(1.8)
        
        # Main chord
        chord = self.generate_smooth_chord(info['chord'], 2.8, 0.18)
        if chord:
            chord.play()
            time.sleep(2.8)
        
        self.last_emotion = emotion

class EmotionMusicApp:
    """Main application"""
    
    def __init__(self, model_path):
        print("🎵 Emotion Symphony - SMOOTH Edition")
        print("=" * 55)
        
        print("Loading model...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model ready")
        
        print("Initializing audio...")
        self.audio = SmoothAudioSynthesizer()
        print("✓ Audio ready" if AUDIO_AVAILABLE else "⚠ Audio unavailable")
        
        print("Starting camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Camera failed!")
        print("✓ Camera ready")
        
        self.current_emotion = 'neutral'
        self.emotion_history = deque(maxlen=12)
        self.last_music_emotion = None
        self.music_playing = False
        self.auto_music = False
        
        print("\n" + "=" * 55)
        print("Controls:")
        print("  SPACE/C - Play chord (2.5s)")
        print("  M       - Play melody (smooth & flowing)")
        print("  A       - Auto-music (plays on change)")
        print("  Q       - Quit")
        print("=" * 55 + "\n")
    
    def preprocess_face(self, face):
        face = cv2.resize(face, (48, 48))
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype('float32') / 255.0
        return np.expand_dims(np.expand_dims(face, 0), -1)
    
    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        processed = self.preprocess_face(face_roi)
        
        preds = self.model.predict(processed, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        confidence = preds[np.argmax(preds)]
        
        return emotion, confidence, (x, y, w, h), preds
    
    def play_music_threaded(self, emotion, mode='chord'):
        if self.music_playing:
            return
        
        self.music_playing = True
        self.last_music_emotion = emotion
        
        if mode == 'melody':
            self.audio.play_smooth_melody(emotion)
        else:
            self.audio.play_smooth_chord(emotion, with_transition=True)
        
        self.music_playing = False
    
    def draw_ui(self, frame, emotion, confidence, box, preds):
        h, w = frame.shape[:2]
        
        if box:
            x, y, fw, fh = box
            color = EMOTION_MUSIC[emotion]['color']
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 3)
            cv2.putText(frame, f"{emotion.upper()}: {confidence*100:.1f}%",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Emotion bars
        for i, emo in enumerate(EMOTIONS):
            y_pos = i * (h // 7)
            conf = preds[i] if preds is not None else 0
            bar_w = int(200 * conf)
            
            cv2.rectangle(frame, (10, y_pos+10), (210, y_pos+(h//7)-10), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, y_pos+10), (10+bar_w, y_pos+(h//7)-10),
                         EMOTION_MUSIC[emo]['color'], -1)
            cv2.putText(frame, f"{emo.upper()}", (15, y_pos+(h//7)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{conf*100:.0f}%", (120, y_pos+(h//7)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Music info
        if self.last_music_emotion:
            info = EMOTION_MUSIC[self.last_music_emotion]
            cv2.rectangle(frame, (w-320, h-150), (w-10, h-10), (40, 40, 40), -1)
            cv2.putText(frame, "🎵 NOW PLAYING", (w-300, h-120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"{self.last_music_emotion.upper()}",
                       (w-300, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info['color'], 2)
            cv2.putText(frame, f"{info['key']} {info['scale']} | {info['tempo']} BPM",
                       (w-300, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, info['character'], (w-300, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Status
        if self.music_playing:
            status, color = "🎵 PLAYING...", (0, 255, 0)
        elif self.auto_music:
            status, color = "AUTO-MUSIC ON (press A to disable)", (0, 255, 255)
        else:
            status, color = "SPACE=chord | M=melody | A=auto", (255, 255, 255)
        
        cv2.putText(frame, status, (w//2-300, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run(self):
        window_name = "🎵 Emotion Symphony - Smooth Audio Edition"
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                result = self.detect_emotion(frame)
                
                if result:
                    emotion, conf, box, preds = result
                    self.emotion_history.append(emotion)
                    
                    if len(self.emotion_history) > 6:
                        self.current_emotion = Counter(self.emotion_history).most_common(1)[0][0]
                    else:
                        self.current_emotion = emotion
                    
                    # Auto-music with transitions
                    if self.auto_music and self.current_emotion != self.last_music_emotion:
                        if not self.music_playing:
                            threading.Thread(target=self.play_music_threaded,
                                           args=(self.current_emotion, 'chord')).start()
                    
                    frame = self.draw_ui(frame, emotion, conf, box, preds)
                else:
                    cv2.putText(frame, "No face detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key in [ord(' '), ord('c')]:
                    if not self.music_playing:
                        threading.Thread(target=self.play_music_threaded,
                                       args=(self.current_emotion, 'chord')).start()
                elif key == ord('m'):
                    if not self.music_playing:
                        threading.Thread(target=self.play_music_threaded,
                                       args=(self.current_emotion, 'melody')).start()
                elif key == ord('a'):
                    self.auto_music = not self.auto_music
                    print(f"\n🎵 Auto-music: {'ON' if self.auto_music else 'OFF'}\n")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if AUDIO_AVAILABLE:
                pygame.mixer.quit()
            print("\n✓ Closed")

def main():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else '../models/best_emotion_model.h5'
    
    try:
        app = EmotionMusicApp(model_path)
        app.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
