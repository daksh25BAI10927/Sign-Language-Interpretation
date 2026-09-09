"""
Convenience root-level entry point.

Allows running the interpreter directly as:

    python main.py
    python main.py --no-preview

from the project root, without needing to remember `python -m backend.main`.
All real logic lives in `backend/main.py`.
"""

from backend.main import _parse_args, app, run_manual

__all__ = ["app", "run_manual"]

if __name__ == "__main__":
    args = _parse_args()
    run_manual(show_preview=not args.no_preview)

