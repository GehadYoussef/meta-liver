#!/bin/bash

# Meta Liver Setup Script
# This script sets up the Meta Liver application

set -e

echo "======================================"
echo "Meta Liver - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import streamlit; import plotly; import pandas; import openpyxl; import networkx" && echo "✓ All packages verified" || echo "✗ Package verification failed"
echo ""

echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To start the app, run:"
echo "  source venv/bin/activate"
echo "  streamlit run streamlit_app.py"
echo ""
echo "The app will be available at:"
echo "  http://localhost:8501"
echo ""
