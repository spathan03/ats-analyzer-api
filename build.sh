#!/bin/bash
echo "Starting build process..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

echo "Build completed successfully!"