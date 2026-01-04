# Emotion Symphony - Setup Guide for VSCode

## 📋 Quick Start Guide

This guide will help you set up and run the Emotion Symphony project in VSCode.

## 📁 Project Structure

```
emotion-symphony-project/
├── README.md                    # Main project documentation
├── SETUP.md                     # This file
├── web/
│   └── index.html              # Standalone web application (open in browser)
├── python/
│   ├── emotion_model.py        # CNN training & real-time detection
│   ├── music_generator.py      # Advanced music composition
│   ├── requirements.txt        # Python dependencies
│   └── demo.py                 # Quick demo script
├── models/
│   └── (trained models go here)
├── data/
│   └── (datasets go here)
└── docs/
    └── (additional documentation)
```

## 🚀 Getting Started

### Option 1: Run the Web App (Easiest - No Installation)

1. Open VSCode
2. Navigate to `web/index.html`
3. Right-click on `index.html` → "Open with Live Server" (if you have the Live Server extension)
   
   **OR**
   
   Simply double-click `index.html` in your file explorer to open in your browser

4. Click "Start Camera" and allow camera permissions
5. Click "Generate Music" to hear the AI compose!

**No installation needed!** The web app runs entirely in your browser.

---

### Option 2: Run Python Backend (Advanced Features)

#### Step 1: Install Python

Make sure you have Python 3.8+ installed:
```bash
python --version
# or
python3 --version
```

If not installed, download from: https://www.python.org/downloads/

#### Step 2: Create Virtual Environment (Recommended)

Open terminal in VSCode (`Ctrl+` ` or `View → Terminal`):

**Windows:**
```bash
cd emotion-symphony-project
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd emotion-symphony-project
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

This will install:
- TensorFlow (for CNN emotion detection)
- OpenCV (for computer vision)
- NumPy, Pandas (for data processing)
- Matplotlib (for visualization)
- MIDIUtil (for music file generation)

#### Step 4: Run the Demo

```bash
# Quick demo - generates music files for each emotion
python demo.py
```

---

## 🎓 Training Your Own Model

### Download Dataset

1. Go to Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Download the `fer2013.csv` file
3. Place it in the `data/` folder

### Train the Model

```bash
cd python
python emotion_model.py train ../data/fer2013.csv
```

Training will:
- Process ~35,000 facial images
- Train a custom CNN with data augmentation
- Save the best model to `../models/best_emotion_model.h5`
- Generate training plots
- Take 2-4 hours on GPU, longer on CPU

### Run Real-Time Detection

```bash
python emotion_model.py detect ../models/best_emotion_model.h5
```

This opens your webcam and detects emotions in real-time!

---

## 🎵 Generate Music Files

Create MIDI files for specific emotions:

```bash
cd python

# Generate happy music (16 bars)
python music_generator.py happy 16 output/happy_music.mid

# Generate sad music (32 bars)
python music_generator.py sad 32 output/sad_music.mid

# Available emotions: happy, sad, angry, fearful, surprised, neutral
```

---

## 🛠️ VSCode Extensions (Recommended)

Install these for the best experience:

1. **Live Server** - Run the web app locally
   - Extension ID: `ritwickdey.LiveServer`

2. **Python** - Python language support
   - Extension ID: `ms-python.python`

3. **Pylance** - Python IntelliSense
   - Extension ID: `ms-python.vscode-pylance`

4. **Jupyter** - For running notebooks (if you create any)
   - Extension ID: `ms-toolsai.jupyter`

To install:
- Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac)
- Search for extension name
- Click "Install"

---

## 🐛 Troubleshooting

### Camera Not Working in Web App

**Issue:** Camera doesn't start in browser

**Solutions:**
1. Use Chrome, Firefox, or Edge (Safari may have issues)
2. Make sure you click "Allow" when browser asks for camera permission
3. Check if another app is using your camera
4. Try HTTPS instead of HTTP (some browsers require secure connection)
5. Check browser console (F12) for errors

### Python Installation Issues

**Issue:** `pip install` fails

**Solutions:**
1. Upgrade pip: `python -m pip install --upgrade pip`
2. Use `--break-system-packages` flag: `pip install -r requirements.txt --break-system-packages`
3. Try Python 3.9 or 3.10 if TensorFlow fails on 3.12

**Issue:** TensorFlow installation fails

**Solutions:**
1. Make sure you have Python 3.8-3.11 (TensorFlow may not support 3.12 yet)
2. On Mac M1/M2: `pip install tensorflow-macos tensorflow-metal`
3. On Windows: Make sure Visual C++ redistributables are installed

**Issue:** OpenCV import error

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Music Not Playing in Web App

**Issue:** No sound when clicking "Generate Music"

**Solutions:**
1. Make sure your computer volume is up
2. Check browser isn't muted
3. Click anywhere on the page first (browsers require user interaction for audio)
4. Check browser console (F12) for errors

---

## 📚 Learning Resources

### Understanding the Code

**emotion_model.py:**
- Lines 1-100: CNN architecture definition
- Lines 101-200: Training loop with data augmentation
- Lines 201-300: Real-time detection class

**music_generator.py:**
- Lines 1-50: Music theory definitions (scales, chords)
- Lines 51-150: Markov chain melody generator
- Lines 151-300: Emotion-based composer

**index.html:**
- Lines 1-500: HTML structure and CSS styling
- Lines 501-end: JavaScript for emotion detection & music generation

### Next Steps

1. **Modify Emotions:** Add new emotions in `emotionToMusic` object
2. **Change Scales:** Edit the `SCALES` dictionary in `music_generator.py`
3. **Improve Model:** Try different CNN architectures
4. **Add Features:** Implement real-time emotion detection in web app

---

## 🎯 Project Goals & Uses

### Portfolio Project
- Showcases ML, web development, and creative AI
- Demonstrates full-stack capabilities
- Great conversation starter in interviews

### Learning Objectives
- Understand CNNs for computer vision
- Learn music theory in code
- Practice TensorFlow/Keras
- Web Audio API experience

### Potential Extensions
- Mobile app version
- Multiplayer emotion symphony
- VR/AR integration
- Spotify/streaming integration
- Real emotion detection in browser (TensorFlow.js)

---

## 💡 Tips for Success

1. **Start Simple:** Begin with the web app, then move to Python
2. **Read Documentation:** Check README.md for detailed explanations
3. **Experiment:** Change tempo, scales, try different emotions
4. **Debug Smartly:** Use print statements, check browser console
5. **Share Your Work:** Deploy to GitHub Pages, share with friends!

---

## 🆘 Getting Help

- Check `README.md` for detailed documentation
- Look at code comments for explanations
- Google error messages
- Stack Overflow for specific issues
- GitHub Issues (if you fork this project)

---

## 🎉 You're Ready!

Start with the web app (`web/index.html`), play around, then dive into the Python code. Have fun creating emotion-driven music!

**Good luck and happy coding!** 🎵🎨🤖
