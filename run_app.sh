#!/usr/bin/env bash
# ==============================================================================
# Helper script to launch the Streamlit app locally
# Handles Apple Silicon / macOS environment variables automatically.
# ==============================================================================

# Ensure Protobuf compatibility with TensorFlow on macOS
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# If the local conda tf_env exists, use its python/streamlit directly
if [ -f "/opt/anaconda3/envs/tf_env/bin/streamlit" ]; then
    echo "🚀 Launching Cast Metal Defect Detection App via conda 'tf_env'..."
    /opt/anaconda3/envs/tf_env/bin/streamlit run app/app.py "$@"
elif [ -d ".venv" ]; then
    echo "🚀 Launching via local virtual environment (.venv)..."
    source .venv/bin/activate
    streamlit run app/app.py "$@"
else
    echo "🚀 Launching via default streamlit..."
    streamlit run app/app.py "$@"
fi
