"""
run.py
──────
Entry point for the AI FastAPI server.

Usage:
    python run.py               # starts on http://127.0.0.1:8000
"""
# check

import os
import warnings

# Suppress TF/library noise before any heavy imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn

if __name__ == "__main__":
    # Ensure absolute imports (e.g. `backend.config`) resolve correctly
    # regardless of the working directory the script is launched from.
    os.environ.setdefault("PYTHONPATH", os.path.dirname(os.path.abspath(__file__)))

    print("Starting OncoScan AI Server on http://localhost:8000/")
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Set to True for local development only —
                        # reload=True spawns a file-watcher thread that
                        # doubles TF initialisation on each code change.
    )

