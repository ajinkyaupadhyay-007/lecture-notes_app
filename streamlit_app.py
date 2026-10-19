"""Root entry point for Streamlit Cloud deployment."""
import sys
from pathlib import Path

# Add project root to sys.path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Forward execution to the actual application
from lecturenotes.app.streamlit_app import *  # noqa: F401, F403
