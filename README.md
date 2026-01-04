# Emotion Symphony

**AI-Powered Real-Time Emotion Detection & Music Generation**

A multimodal AI system that detects facial emotions using a custom CNN and generates corresponding music in real-time. Built with TensorFlow, OpenCV, and Pygame.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

---

## 🌟 Features

- **Real-Time Emotion Detection**: Custom CNN trained on FER-2013 dataset (65% accuracy)
- **Live Music Generation**: Algorithmic composition based on detected emotions
- **Smooth Audio Transitions**: Professional-quality audio synthesis with harmonic richness
- **7 Emotion Classes**: Happy, Sad, Angry, Fearful, Surprised, Neutral, Disgust
- **Interactive UI**: Live webcam feed with confidence visualization
- **Multiple Modes**: Chord playback, full melodies, auto-music mode

---

## 🏗️ Architecture

### Emotion Detection (CNN)
- **Input**: 48x48 grayscale facial images
- **Architecture**: 4 convolutional blocks (64→128→256→512 filters)
- **Regularization**: BatchNormalization + Dropout
- **Parameters**: ~5.5M
- **Performance**: 65.44% test accuracy on FER-2013

### Music Generation
- **Method**: Rule-based composition with music theory
- **Features**: Markov chains, scale mapping, chord progressions
- **Audio**: Real-time synthesis with harmonics and ADSR envelopes
- **Transitions**: Smooth frequency gliding between emotions

### Emotion-to-Music Mapping

| Emotion | Tempo | Scale | Key | Character |
|---------|-------|-------|-----|-----------|
| Happy | 140 BPM | Major | C | Bright & Upbeat |
| Sad | 70 BPM | Minor | A | Melancholic & Slow |
| Angry | 160 BPM | Minor | E | Intense & Aggressive |
| Fearful | 90 BPM | Diminished | F# | Tense & Dark |
| Surprised | 130 BPM | Lydian | D | Playful & Bright |
| Neutral | 100 BPM | Major | G | Balanced & Steady |
| Disgust | 85 BPM | Minor | B | Dissonant & Uneasy |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Audio output device

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/emotion-symphony.git
cd emotion-symphony

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r python/requirements.txt
```

### Running the Application

#### Option 1: Web Demo (No Installation)
```bash
# Open in browser
open web/index.html
```

#### Option 2: Real-Time Detection + Music (Recommended)
```bash
cd python
python emotion_music_smooth.py
```

**Controls:**
- `SPACE` or `C` - Play chord for current emotion
- `M` - Play full melody
- `A` - Toggle auto-music mode
- `Q` - Quit

---

## 📊 Training Your Own Model

### 1. Download Dataset

Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in `data/archive/`

### 2. Train the Model
```bash
cd python
python train_from_images.py
```

Training takes 2-4 hours on GPU, 8-12 hours on CPU.

### 3. Evaluate
```bash
python emotion_model.py detect ../models/best_emotion_model.h5
```

---

## 📁 Project Structure
```
emotion-symphony/
├── README.md
├── LICENSE
├── web/
│   └── index.html              # Standalone web demo
├── python/
│   ├── emotion_model.py        # CNN training & detection
│   ├── music_generator.py      # Music composition engine
│   ├── train_from_images.py   # Image-based training script
│   ├── emotion_music_smooth.py # (main) Real-time app with audio
│   ├── demo.py                 # Quick demo
│   └── requirements.txt        # Python dependencies
├── models/
│   └── best_emotion_model.h5   # Trained CNN (not included)
├── data/
│   └── archive/                # FER-2013 dataset (not included)
└── docs/
    └── (additional documentation)
```

---

## 🛠️ Technologies Used

### Machine Learning
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision
- **NumPy** - Numerical computing

### Audio
- **Pygame** - Audio synthesis and playback
- **Wave generation** - Custom ADSR envelopes, harmonics

### Web
- **HTML/CSS/JavaScript** - Web interface
- **Tone.js** - Web Audio API

---

## 📈 Performance

- **Model Accuracy**: 65.44% on FER-2013 test set
- **Real-time FPS**: 25-30 FPS
- **Inference Time**: 15-30ms per frame
- **Audio Latency**: <100ms

---

## 🎓 Technical Highlights

### Computer Vision
- Custom CNN architecture from scratch
- Data augmentation (rotation, shift, zoom, flip)
- Real-time face detection with Haar Cascades
- Emotion smoothing with temporal averaging

### Music Generation
- Music theory-based composition (scales, chords, progressions)
- Real-time audio synthesis with harmonics
- Smooth transitions using frequency gliding
- ADSR envelope shaping for professional sound

### System Design
- Multi-threaded architecture for concurrent audio/video
- Efficient frame processing pipeline
- Low-latency audio synthesis

---

## 🐛 Known Issues

- Safari browser has limited camera support (use Chrome/Firefox)
- Model may misclassify subtle expressions in low light
- MIDI playback requires additional system setup

---

## 🔮 Future Enhancements

- [ ] LSTM-based emotion prediction for temporal smoothness
- [ ] Multi-instrument orchestration
- [ ] Style transfer for different musical genres
- [ ] Mobile app version
- [ ] Real-time collaboration mode
- [ ] Export to MP3/WAV/MusicXML
- [ ] Fine-tune on more diverse datasets

---

## 🙏 Acknowledgments

- **FER-2013 Dataset** - Facial Expression Recognition dataset
- **TensorFlow Team** - Deep learning framework
- **OpenCV Community** - Computer vision tools
- **Music Theory** - Traditional Western music theory principles

---

## ⭐ Show Your Support

If you found this project interesting or helpful, please consider giving it a ⭐!

---

**Built with ❤️ at the intersection of AI, music, and human emotion**
