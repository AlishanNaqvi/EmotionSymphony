"""
Emotion Symphony - Live Real-Time Application
==============================================
Combines real emotion detection with live music generation

Usage: python emotion_music_live.py
"""

import cv2
import numpy as np
from tensorflow import keras
import threading
import time
from collections import deque
import random

# Try to import audio libraries
try:
    from pydub import AudioSegment
    from pydub.playback import play
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    print("Note: Install pydub for audio playback: pip install pydub")

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion to music mapping
EMOTION_MUSIC = {
    'happy': {'tempo': 140, 'scale': 'Major', 'key': 'C', 'character': 'Upbeat & Bright'},
    'sad': {'tempo': 70, 'scale': 'Minor', 'key': 'A', 'character': 'Melancholic & Slow'},
    'angry': {'tempo': 160, 'scale': 'Minor', 'key': 'E', 'character': 'Intense & Aggressive'},
    'fear': {'tempo': 90, 'scale': 'Diminished', 'key': 'F#', 'character': 'Tense & Dark'},
    'surprise': {'tempo': 130, 'scale': 'Lydian', 'key': 'D', 'character': 'Playful & Bright'},
    'neutral': {'tempo': 100, 'scale': 'Major', 'key': 'G', 'character': 'Balanced & Steady'},
    'disgust': {'tempo': 85, 'scale': 'Minor', 'key': 'B', 'character': 'Dissonant & Uneasy'}
}

# Musical scales (simplified)
SCALES = {
    'Major': [0, 2, 4, 5, 7, 9, 11],
    'Minor': [0, 2, 3, 5, 7, 8, 10],
    'Diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'Lydian': [0, 2, 4, 6, 7, 9, 11]
}

class EmotionMusicLive:
    """Real-time emotion detection with live music generation"""
    
    def __init__(self, model_path):
        """Initialize the application"""
        print("🎵 Emotion Symphony - Live Edition")
        print("=" * 50)
        
        # Load the trained model
        print("Loading emotion detection model...")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam!")
        print("✓ Webcam initialized")
        
        # State variables
        self.current_emotion = 'neutral'
        self.emotion_history = deque(maxlen=10)  # Smooth predictions
        self.last_music_emotion = None
        self.music_playing = False
        self.music_thread = None
        
        # Visual settings
        self.window_name = "🎵 Emotion Symphony - Live Detection"
        self.colors = {
            'happy': (0, 255, 255),      # Yellow
            'sad': (255, 0, 0),          # Blue
            'angry': (0, 0, 255),        # Red
            'fear': (128, 0, 128),       # Purple
            'surprise': (0, 165, 255),   # Orange
            'neutral': (255, 255, 255),  # White
            'disgust': (0, 128, 0)       # Green
        }
        
        print("✓ Setup complete!")
        print("\n" + "=" * 50)
        print("Controls:")
        print("  SPACE - Generate music for current emotion")
        print("  'a'   - Enable auto-music (plays on emotion change)")
        print("  'q'   - Quit")
        print("=" * 50 + "\n")
    
    def preprocess_face(self, face):
        """Preprocess face for model input"""
        # Resize to 48x48
        face = cv2.resize(face, (48, 48))
        # Convert to grayscale
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Normalize
        face = face.astype('float32') / 255.0
        # Reshape for model
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        return face
    
    def detect_emotion(self, frame):
        """Detect emotion from frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract and preprocess face
        face_roi = gray[y:y+h, x:x+w]
        processed_face = self.preprocess_face(face_roi)
        
        # Predict emotion
        predictions = self.model.predict(processed_face, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        emotion = EMOTIONS[emotion_idx]
        
        return emotion, confidence, (x, y, w, h), predictions
    
    def generate_melody(self, emotion, duration_seconds=4):
        """Generate a simple melody for the emotion"""
        music_info = EMOTION_MUSIC[emotion]
        tempo = music_info['tempo']
        scale_type = music_info['scale']
        
        # Get scale notes
        base_note = 60  # Middle C
        scale = SCALES.get(scale_type, SCALES['Major'])
        notes = [base_note + note for note in scale]
        
        # Calculate number of notes based on tempo
        beats_per_second = tempo / 60.0
        num_notes = int(beats_per_second * duration_seconds)
        
        # Generate melody using scale notes
        melody = []
        for _ in range(num_notes):
            note = random.choice(notes)
            melody.append(note)
        
        return melody, music_info
    
    def play_music_console(self, emotion):
        """Play music representation in console (when audio not available)"""
        melody, music_info = self.generate_melody(emotion)
        
        print(f"\n🎵 Playing {emotion.upper()} music:")
        print(f"   Tempo: {music_info['tempo']} BPM")
        print(f"   Scale: {music_info['key']} {music_info['scale']}")
        print(f"   Character: {music_info['character']}")
        print(f"   Melody: {' '.join([str(n) for n in melody[:16]])}...")
        print()
    
    def generate_and_play_music(self, emotion):
        """Generate and play music for emotion"""
        if self.music_playing:
            return
        
        self.music_playing = True
        self.last_music_emotion = emotion
        
        # Console output (always available)
        self.play_music_console(emotion)
        
        # TODO: Could add actual audio synthesis here with libraries like:
        # - pygame.midi
        # - mido + fluidsynth
        # - pydub with generated tones
        
        # Simulate music duration
        time.sleep(2)
        self.music_playing = False
    
    def draw_ui(self, frame, emotion, confidence, face_box, predictions):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw face box
        if face_box is not None:
            (x, y, fw, fh) = face_box
            color = self.colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 3)
            
            # Draw emotion label
            label = f"{emotion.upper()}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw emotion bars on left side
        bar_height = h // len(EMOTIONS)
        for i, emo in enumerate(EMOTIONS):
            y_pos = i * bar_height
            conf = predictions[i] if predictions is not None else 0
            bar_width = int(200 * conf)
            
            # Background
            cv2.rectangle(frame, (10, y_pos + 10), (210, y_pos + bar_height - 10),
                         (40, 40, 40), -1)
            
            # Confidence bar
            color = self.colors.get(emo, (255, 255, 255))
            cv2.rectangle(frame, (10, y_pos + 10), (10 + bar_width, y_pos + bar_height - 10),
                         color, -1)
            
            # Label
            cv2.putText(frame, f"{emo.upper()}", (15, y_pos + bar_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{conf*100:.0f}%", (120, y_pos + bar_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current music info
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
        status = "🎵 PLAYING..." if self.music_playing else "Press SPACE to play music"
        cv2.putText(frame, status, (w // 2 - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main application loop"""
        auto_music = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect emotion
                result = self.detect_emotion(frame)
                
                if result[0] is not None:
                    emotion, confidence, face_box, predictions = result
                    
                    # Add to history for smoothing
                    self.emotion_history.append(emotion)
                    
                    # Get most common emotion from history
                    if len(self.emotion_history) > 5:
                        from collections import Counter
                        emotion_counts = Counter(self.emotion_history)
                        self.current_emotion = emotion_counts.most_common(1)[0][0]
                    else:
                        self.current_emotion = emotion
                    
                    # Auto-play music on emotion change
                    if auto_music and self.current_emotion != self.last_music_emotion:
                        if not self.music_playing:
                            self.music_thread = threading.Thread(
                                target=self.generate_and_play_music,
                                args=(self.current_emotion,)
                            )
                            self.music_thread.start()
                    
                    # Draw UI
                    frame = self.draw_ui(frame, emotion, confidence, face_box, predictions)
                else:
                    # No face detected
                    cv2.putText(frame, "No face detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Spacebar - generate music
                    if not self.music_playing:
                        self.music_thread = threading.Thread(
                            target=self.generate_and_play_music,
                            args=(self.current_emotion,)
                        )
                        self.music_thread.start()
                elif key == ord('a'):
                    # Toggle auto-music
                    auto_music = not auto_music
                    status = "ON" if auto_music else "OFF"
                    print(f"\n🎵 Auto-music: {status}\n")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
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
        print("\nUsage: python emotion_music_live.py [model_path]")
        print("Example: python emotion_music_live.py ../models/best_emotion_model.h5")
        sys.exit(1)

if __name__ == "__main__":
    main()
