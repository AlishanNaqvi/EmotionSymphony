"""
Quick Demo Script for Emotion Symphony
=======================================
This script demonstrates the music generation capabilities
without requiring the full CNN training.

Usage: python demo.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_generator import generate_emotion_music

def main():
    """Run demo of emotion-based music generation"""
    
    print("=" * 60)
    print("EMOTION SYMPHONY - Music Generation Demo")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = "../models"
    os.makedirs(output_dir, exist_ok=True)
    
    emotions = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral']
    
    print("Generating music for 6 different emotions...")
    print("This will create MIDI files you can play in any media player.\n")
    
    for emotion in emotions:
        print(f"🎵 Generating {emotion.upper()} music...")
        output_file = os.path.join(output_dir, f"{emotion}_music.mid")
        
        try:
            generate_emotion_music(emotion, duration_bars=16, output_file=output_file)
            print(f"   ✓ Saved to: {output_file}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Generated files are in the 'models/' directory.")
    print("Open them with any MIDI player (Windows Media Player, VLC, etc.)")
    print()
    print("Each file demonstrates different musical characteristics:")
    print("  - Happy: Fast tempo (140 BPM), C Major, upbeat")
    print("  - Sad: Slow tempo (70 BPM), A Minor, melancholic")
    print("  - Angry: Very fast (160 BPM), E Minor, aggressive")
    print("  - Fearful: Tense (90 BPM), F# Minor, diminished scale")
    print("  - Surprised: Lively (130 BPM), D Major, staccato")
    print("  - Neutral: Moderate (100 BPM), G Major, steady")
    print()
    print("Try listening to each one and notice the differences!")
    print()


if __name__ == "__main__":
    main()
