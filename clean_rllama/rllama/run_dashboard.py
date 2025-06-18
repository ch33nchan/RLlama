#!/usr/bin/env python3
"""
Launch script for RLlama Dashboard
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "streamlit_app.py")
    
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard not found at {dashboard_path}")
        return
    
    print("🦙 Launching RLlama Dashboard...")
    print("Dashboard will open in your web browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🦙 Dashboard stopped.")

if __name__ == "__main__":
    main()