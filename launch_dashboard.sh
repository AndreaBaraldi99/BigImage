#!/bin/bash
# Ship Detection Dashboard Launch Script

echo "🚢 Starting Ship Detection Dashboard..."
echo "=================================="

# Set environment variables to suppress PyTorch warnings
export TORCH_CLASSES_UNWRAP=1
export TORCH_CPP_LOG_LEVEL=2
export PYTHONWARNINGS="ignore::UserWarning"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Launch Streamlit app with optimized settings
echo "Launching dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless true \
    --browser.gatherUsageStats false \
    --global.suppressDeprecationWarnings true

echo "Dashboard stopped."
