from importlib.metadata import version

try:
    __version__ = version("appdaemon")
except ImportError:
    # Fallback for development/editable installs or if package not installed
    __version__ = "unknown"

__version_comments__ = ""
