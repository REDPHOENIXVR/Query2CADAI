"""Entrypoint for launching the unified Query2CAD Web UI."""

def main():
    # Ensure startup directories are present
    from src.utils import ensure_startup_dirs
    ensure_startup_dirs()
    # Launch the unified Gradio Web UI (handles Gradio import errors internally)
    from src.web_ui import launch_web_ui
    launch_web_ui()

if __name__ == "__main__":
    main()