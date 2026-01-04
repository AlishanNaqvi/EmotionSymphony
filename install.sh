#!/bin/bash

# Emotion Symphony - Master Installation Script
# This script sets up the entire project environment

echo "=========================================="
echo "   EMOTION SYMPHONY - Installation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Found Python $PYTHON_VERSION"
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "✓ Found Python $PYTHON_VERSION"
else
    echo "✗ Python not found! Please install Python 3.8+ first."
    exit 1
fi

echo ""

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Mac/Linux
    source venv/bin/activate
fi

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
echo "This may take a few minutes..."
cd python
pip install -r requirements.txt --break-system-packages
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "⚠ Some dependencies may have failed to install"
    echo "  Check the error messages above"
fi
cd ..

echo ""
echo "=========================================="
echo "   Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Quick Start (Web App):"
echo "   - Open web/index.html in your browser"
echo ""
echo "2. Run Music Demo:"
echo "   - cd python"
echo "   - python demo.py"
echo ""
echo "3. Train Model (requires dataset):"
echo "   - Download FER-2013 from Kaggle"
echo "   - python emotion_model.py train ../data/fer2013.csv"
echo ""
echo "4. Check SETUP.md for detailed instructions"
echo ""
echo "Happy coding! 🎵"
