"""
Main entry point for the Skin Cancer Classification MLOps Pipeline
"""
from src.orchestrator import MLOpsOrchestrator

if __name__ == "__main__":
    orchestrator = MLOpsOrchestrator(api_port=8000, ui_port=8501)
    orchestrator.run()
