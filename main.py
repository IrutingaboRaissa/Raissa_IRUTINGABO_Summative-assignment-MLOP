from src.orchestrator import MLOpsOrchestrator
import sys

if __name__ == "__main__":
    # Check command line arguments for specific service
    if len(sys.argv) > 1:
        service = sys.argv[1].lower()
        
        if service == "ui":
            print("🌐 Starting Streamlit UI only...")
            print("📍 UI will be available at: http://localhost:8501")
            import subprocess
            subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
        
        elif service == "api":
            print("🔌 Starting API only...")
            print("📍 API will be available at: http://localhost:8000")
            import subprocess
            subprocess.run(["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
        
        elif service == "locust":
            print("🐝 Starting Locust only...")
            print("📍 Locust UI will be available at: http://localhost:8089")
            import subprocess
            subprocess.run(["locust", "-f", "locustfile.py", "--host=http://localhost:8000"])
        
        else:
            print(f"❌ Unknown service: {service}")
            print("\nUsage:")
            print("  python main.py ui      - Start Streamlit UI")
            print("  python main.py api     - Start FastAPI")
            print("  python main.py locust  - Start Locust")
            print("  python main.py         - Start all services")
    
    else:
        # Start all services using orchestrator
        print("🚀 Starting all services...")
        orchestrator = MLOpsOrchestrator(api_port=8000, ui_port=8501)
        orchestrator.run()
