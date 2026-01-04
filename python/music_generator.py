"""
Advanced Music Generation System
=================================
Sophisticated algorithmic music composition based on emotional states.
Uses music theory principles, Markov chains, and generative algorithms.

Requirements:
pip install midiutil numpy --break-system-packages
"""

import random
import numpy as np
from midiutil import MIDIFile
from collections import defaultdict


class MusicTheory:
    """Music theory definitions and utilities"""
    
    # Scales (semitone intervals from root)
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'pentatonic_major': [0, 2, 4, 7, 9],
        'pentatonic_minor': [0, 3, 5, 7, 10],
        'blues': [0, 3, 5, 6, 7, 10],
        'whole_tone': [0, 2, 4, 6, 8, 10],
        'diminished': [0, 2, 3, 5, 6, 8, 9, 11]
    }
    
    # Chord progressions (roman numeral notation converted to scale degrees)
    PROGRESSIONS = {
        'classic': [[0, 3, 4, 0], [0, 4, 3, 0]],  # I-IV-V-I
        'pop': [[0, 5, 3, 4]],  # I-vi-IV-V
        'jazz': [[0, 3, 1, 4], [1, 4, 0]],  # ii-V-I
        'sad': [[0, 3, 5, 4]],  # i-iv-vi-V
        'dark': [[0, 5, 3, 2]]  # i-vi-iv-III
    }
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    @staticmethod
    def get_scale_notes(root_note, scale_type, octave=4):
        """Get MIDI note numbers for a scale"""
        root_midi = 60 + (octave - 4) * 12  # C4 = 60
        root_midi += MusicTheory.NOTE_NAMES.index(root_note)
        
        intervals = MusicTheory.SCALES[scale_type]
        return [root_midi + interval for interval in intervals]
    
    @staticmethod
    def get_chord(root_note, chord_type='major'):
        """Get MIDI notes for a chord"""
        if chord_type == 'major':
            intervals = [0, 4, 7]
        elif chord_type == 'minor':
            intervals = [0, 3, 7]
        elif chord_type == 'diminished':
            intervals = [0, 3, 6]
        elif chord_type == 'augmented':
            intervals = [0, 4, 8]
        elif chord_type == 'seventh':
            intervals = [0, 4, 7, 10]
        elif chord_type == 'minor_seventh':
            intervals = [0, 3, 7, 10]
        else:
            intervals = [0, 4, 7]
        
        return [root_note + i for i in intervals]


class MarkovMelodyGenerator:
    """Generate melodies using Markov chains"""
    
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(list)
        
    def train(self, sequences):
        """Train on existing melodic sequences"""
        for sequence in sequences:
            for i in range(len(sequence) - self.order):
                state = tuple(sequence[i:i + self.order])
                next_note = sequence[i + self.order]
                self.transitions[state].append(next_note)
    
    def generate(self, length, scale_notes, seed=None):
        """Generate a new melody"""
        if seed is None:
            seed = tuple(random.choices(scale_notes, k=self.order))
        
        melody = list(seed)
        current_state = seed
        
        for _ in range(length - self.order):
            if current_state in self.transitions and self.transitions[current_state]:
                next_note = random.choice(self.transitions[current_state])
            else:
                # Fallback to random note from scale
                next_note = random.choice(scale_notes)
            
            melody.append(next_note)
            current_state = tuple(melody[-self.order:])
        
        return melody


class EmotionBasedComposer:
    """Compose music based on emotional parameters"""
    
    def __init__(self, emotion):
        self.emotion = emotion
        self.params = self._get_emotion_parameters()
        
    def _get_emotion_parameters(self):
        """Map emotions to musical parameters"""
        params = {
            'happy': {
                'tempo': 140,
                'key': 'C',
                'scale': 'major',
                'rhythm_density': 0.8,
                'note_duration': 0.25,
                'octave_range': (4, 6),
                'chord_progression': 'pop',
                'dynamics': 'forte',
                'articulation': 'staccato'
            },
            'sad': {
                'tempo': 60,
                'key': 'A',
                'scale': 'minor',
                'rhythm_density': 0.4,
                'note_duration': 1.0,
                'octave_range': (3, 5),
                'chord_progression': 'sad',
                'dynamics': 'piano',
                'articulation': 'legato'
            },
            'angry': {
                'tempo': 160,
                'key': 'E',
                'scale': 'phrygian',
                'rhythm_density': 0.9,
                'note_duration': 0.125,
                'octave_range': (3, 5),
                'chord_progression': 'dark',
                'dynamics': 'fortissimo',
                'articulation': 'marcato'
            },
            'fearful': {
                'tempo': 90,
                'key': 'F#',
                'scale': 'diminished',
                'rhythm_density': 0.6,
                'note_duration': 0.5,
                'octave_range': (3, 5),
                'chord_progression': 'dark',
                'dynamics': 'mezzo-piano',
                'articulation': 'tremolo'
            },
            'surprised': {
                'tempo': 130,
                'key': 'D',
                'scale': 'lydian',
                'rhythm_density': 0.7,
                'note_duration': 0.25,
                'octave_range': (4, 6),
                'chord_progression': 'jazz',
                'dynamics': 'mezzo-forte',
                'articulation': 'staccato'
            },
            'neutral': {
                'tempo': 100,
                'key': 'G',
                'scale': 'major',
                'rhythm_density': 0.6,
                'note_duration': 0.5,
                'octave_range': (4, 5),
                'chord_progression': 'classic',
                'dynamics': 'mezzo-forte',
                'articulation': 'normal'
            }
        }
        
        return params.get(self.emotion, params['neutral'])
    
    def compose(self, duration_bars=16, output_file='emotion_music.mid'):
        """Compose a complete piece"""
        midi = MIDIFile(3)  # 3 tracks: melody, harmony, bass
        
        # Track 0: Melody
        track = 0
        channel = 0
        tempo = self.params['tempo']
        midi.addTempo(track, 0, tempo)
        
        # Get scale
        scale_notes = MusicTheory.get_scale_notes(
            self.params['key'],
            self.params['scale'],
            self.params['octave_range'][0]
        )
        
        # Extend scale to cover octave range
        octave_extension = []
        for octave in range(self.params['octave_range'][0], self.params['octave_range'][1] + 1):
            octave_extension.extend(
                MusicTheory.get_scale_notes(self.params['key'], self.params['scale'], octave)
            )
        scale_notes = sorted(list(set(octave_extension)))
        
        # Generate melody using Markov chain
        markov = MarkovMelodyGenerator(order=2)
        
        # Create training sequences with some musical patterns
        training_sequences = self._generate_training_sequences(scale_notes)
        markov.train(training_sequences)
        
        # Generate main melody
        beats_per_bar = 4
        total_beats = duration_bars * beats_per_bar
        melody = markov.generate(int(total_beats / self.params['note_duration']), scale_notes)
        
        # Add melody to MIDI
        time = 0
        for note in melody:
            if random.random() < self.params['rhythm_density']:
                duration = self.params['note_duration']
                velocity = self._get_velocity(self.params['dynamics'])
                midi.addNote(track, channel, note, time, duration, velocity)
            time += self.params['note_duration']
        
        # Track 1: Harmony (chords)
        track = 1
        channel = 1
        time = 0
        
        progression = MusicTheory.PROGRESSIONS[self.params['chord_progression']][0]
        for bar in range(duration_bars):
            chord_root_idx = progression[bar % len(progression)]
            chord_root = scale_notes[chord_root_idx]
            
            # Determine chord type based on scale
            if self.params['scale'] in ['major', 'lydian', 'mixolydian']:
                chord_type = 'major' if chord_root_idx in [0, 3, 4] else 'minor'
            else:
                chord_type = 'minor'
            
            chord_notes = MusicTheory.get_chord(chord_root, chord_type)
            
            for note in chord_notes:
                velocity = self._get_velocity(self.params['dynamics']) - 20
                midi.addNote(track, channel, note, time, beats_per_bar, velocity)
            
            time += beats_per_bar
        
        # Track 2: Bass
        track = 2
        channel = 2
        time = 0
        
        bass_octave = self.params['octave_range'][0] - 1
        bass_scale = MusicTheory.get_scale_notes(self.params['key'], self.params['scale'], bass_octave)
        
        for bar in range(duration_bars):
            chord_root_idx = progression[bar % len(progression)]
            bass_note = bass_scale[chord_root_idx]
            
            # Create bass pattern
            if self.emotion == 'angry':
                # Rapid bass notes
                for beat in range(beats_per_bar):
                    midi.addNote(track, channel, bass_note, time, 0.25, 90)
                    time += 0.25
                    midi.addNote(track, channel, bass_note + 12, time, 0.25, 85)
                    time += 0.25
                    midi.addNote(track, channel, bass_note, time, 0.25, 90)
                    time += 0.25
                    midi.addNote(track, channel, bass_note + 7, time, 0.25, 85)
                    time += 0.25
            else:
                # Standard bass
                midi.addNote(track, channel, bass_note, time, beats_per_bar, 80)
                time += beats_per_bar
        
        # Write MIDI file
        with open(output_file, 'wb') as f:
            midi.writeFile(f)
        
        print(f"Composition saved to {output_file}")
        print(f"Emotion: {self.emotion}")
        print(f"Tempo: {tempo} BPM")
        print(f"Key: {self.params['key']} {self.params['scale']}")
        print(f"Duration: {duration_bars} bars")
        
        return output_file
    
    def _generate_training_sequences(self, scale_notes):
        """Generate training sequences with musical patterns"""
        sequences = []
        
        # Ascending/descending scales
        sequences.append(scale_notes[:7])
        sequences.append(scale_notes[6::-1])
        
        # Arpeggios
        sequences.append([scale_notes[i] for i in [0, 2, 4, 2, 0]])
        sequences.append([scale_notes[i] for i in [0, 2, 4, 6, 4, 2, 0]])
        
        # Step patterns
        for start in range(len(scale_notes) - 4):
            sequences.append([scale_notes[start + i] for i in [0, 1, 0, 2, 1, 3]])
        
        return sequences
    
    def _get_velocity(self, dynamic):
        """Convert dynamic marking to MIDI velocity"""
        dynamics_map = {
            'pianissimo': 40,
            'piano': 60,
            'mezzo-piano': 70,
            'mezzo-forte': 85,
            'forte': 100,
            'fortissimo': 115
        }
        return dynamics_map.get(dynamic, 85)


class MultiEmotionComposer:
    """Compose music that transitions between emotions"""
    
    def __init__(self, emotion_sequence):
        """
        emotion_sequence: list of tuples (emotion, duration_bars)
        e.g., [('sad', 8), ('happy', 8), ('neutral', 4)]
        """
        self.emotion_sequence = emotion_sequence
    
    def compose(self, output_file='multi_emotion_music.mid'):
        """Compose a piece with emotional transitions"""
        midi = MIDIFile(3)
        
        total_time = 0
        
        for emotion, duration in self.emotion_sequence:
            composer = EmotionBasedComposer(emotion)
            
            # Add tempo marking
            if total_time == 0:
                midi.addTempo(0, total_time, composer.params['tempo'])
            
            # Compose section
            temp_file = f'temp_{emotion}.mid'
            composer.compose(duration, temp_file)
            
            # Note: In practice, you'd need to merge MIDI files properly
            # This is simplified - full implementation would require proper MIDI merging
            
            total_time += duration * 4  # assuming 4/4 time
        
        print(f"Multi-emotion composition created with {len(self.emotion_sequence)} sections")
        return output_file


def generate_emotion_music(emotion='happy', duration_bars=16, output_file='output.mid'):
    """Simple function to generate music for a given emotion"""
    composer = EmotionBasedComposer(emotion)
    return composer.compose(duration_bars, output_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        emotion = sys.argv[1]
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        output = sys.argv[3] if len(sys.argv) > 3 else f'{emotion}_music.mid'
    else:
        emotion = 'happy'
        duration = 16
        output = 'output.mid'
    
    print(f"Generating {duration}-bar composition for emotion: {emotion}")
    generate_emotion_music(emotion, duration, output)
    
    # Example of multi-emotion composition
    # multi_composer = MultiEmotionComposer([
    #     ('sad', 8),
    #     ('neutral', 4),
    #     ('happy', 8),
    #     ('surprised', 4)
    # ])
    # multi_composer.compose('emotional_journey.mid')
