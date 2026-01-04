# 🚀 QUICK START GUIDE

## Welcome to Emotion Symphony!

You've downloaded the complete codebase. Here's how to get started in 5 minutes.

---

## 📦 What You Downloaded

A complete ML project with:
- ✅ Web application (works instantly in browser)
- ✅ Python backend for ML training
- ✅ Music generation algorithms
- ✅ VSCode configuration
- ✅ Complete documentation

---

## 🏃 Fastest Path (Web App Only)

**No installation needed!**

1. Extract the zip file
2. Open `web/index.html` in Chrome/Firefox/Edge
3. Click "Start Camera" → Allow permissions
4. Click "Generate Music" → Enjoy!

⏱️ **Time: 1 minute**

---

## 🎵 Try Music Generation (Python)

**Requires Python 3.8+**

### Windows:
```batch
1. Extract the zip
2. Double-click install.bat
3. cd python
4. python demo.py
```

### Mac/Linux:
```bash
1. Extract the zip
2. bash install.sh
3. cd python
4. python demo.py
```

This generates 6 MIDI files (one for each emotion)!

⏱️ **Time: 5 minutes**

---

## 🧠 Full ML Pipeline (Advanced)

**For training your own model:**

1. Download FER-2013 dataset from Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013

2. Place `fer2013.csv` in the `data/` folder

3. Run training:
   ```bash
   cd python
   python emotion_model.py train ../data/fer2013.csv
   ```

4. Run real-time detection:
   ```bash
   python emotion_model.py detect ../models/best_emotion_model.h5
   ```

⏱️ **Time: 2-4 hours (training)**

---

## 📖 Documentation

- **SETUP.md** - Detailed setup for VSCode
- **PROJECT_README.md** - Full project documentation
- **README.md** (in python folder) - Technical details

---

## 🎯 Recommended Path

**Day 1:** Try the web app → Play with instant demo
**Day 2:** Run Python demo → Generate MIDI files
**Day 3:** Read documentation → Understand the code
**Day 4:** Train model → Advanced features

---

## 🆘 Need Help?

### Common Issues:

**Web app camera not working?**
- Use Chrome/Firefox (not Safari)
- Click "Allow" when prompted
- Check no other app is using camera

**Python installation fails?**
- Make sure Python 3.8-3.11 installed
- Run as administrator (Windows)
- Use virtual environment

**MIDI files won't play?**
- Use VLC Media Player
- Try online MIDI player
- Convert to MP3 online

### Check Documentation:
- See SETUP.md Section: "Troubleshooting"
- Read error messages carefully
- Google the specific error

---

## 🎓 What You'll Learn

✅ Computer Vision (CNNs, face detection)
✅ Music Theory (scales, chords, composition)  
✅ Web Audio (Tone.js, synthesis)
✅ Machine Learning (training, evaluation)
✅ Full-Stack Development

---

## 🌟 Pro Tips

1. **Start simple** - Web app first, then Python
2. **Read comments** - Code is heavily documented
3. **Experiment** - Change tempo, scales, emotions
4. **Share** - Show friends, add to portfolio
5. **Extend** - Add your own features!

---

## 🎉 You're Ready!

Pick a path above and dive in. Most importantly: **have fun creating emotion-driven music!**

Questions? Check the docs or experiment!

**Happy coding!** 🎵🎨🤖

---

**Quick Links:**
- Web App: `web/index.html`
- Python Demo: `python/demo.py`
- Documentation: `SETUP.md`
- Training: `python/emotion_model.py`
