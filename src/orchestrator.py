import subprocess
import threading
import time
import sys
from pathlib import Path


class MLOpsOrchestrator:
    """Orchestrates the API and UI services for the skin cancer classification system"""
    
    def __init__(self, api_port: int = 8000, ui_port: int = 8501):
        self.api_port = api_port
        self.ui_port = ui_port
        self.api_process = None
        self.ui_process = None
        
    def start_api(self):
        """Start the FastAPI backend service"""
        print(f"Starting API on port {self.api_port}...")
        self.api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", str(self.api_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Wait for API to initialize
        
    def start_ui(self):
        """Start the Streamlit dashboard service"""
        print(f"Starting UI on port {self.ui_port}...")
        self.ui_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(self.ui_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
    def run(self):
        """Start all services"""
        print("=" * 50)
        print("MLOps Pipeline Starting...")
        print("=" * 50)
        
        # Start API in separate thread
        api_thread = threading.Thread(target=self.start_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Start UI
        self.start_ui()
        
        print("\nAll services running!")
        print(f"API: http://localhost:{self.api_port}")
        print(f"UI: http://localhost:{self.ui_port}")
        print("\nPress Ctrl+C to stop all services\n")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down services...")
            self.stop()
            
    def stop(self):
        """Stop all services"""
        if self.api_process:
            self.api_process.terminate()
        if self.ui_process:
            self.ui_process.terminate()
        print("All services stopped")
