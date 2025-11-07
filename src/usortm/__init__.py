"""uSort-M: Utilities for rapid and low-cost parsed protein library generation."""

__version__ = "0.1.0"

from pathlib import Path

def _apply_plot_style():
    """Automatically apply the default usortm plotting style."""
    try:
        import matplotlib.pyplot as plt
        style_path = Path(__file__).parent / "usortm.mplstyle"
        plt.style.use(style_path)
    except ImportError:
        # matplotlib not installed, skip silently
        pass

# Auto-apply style on import
_apply_plot_style()

# Optional: still expose for manual use if needed
def get_style_path():
    """Get the path to the default usortm matplotlib style file."""
    return Path(__file__).parent / "usortm.mplstyle"

__all__ = ['__version__', 'get_style_path']