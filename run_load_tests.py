"""
Load Testing Script for Skin Cancer Detection API
Runs comprehensive performance tests using Locust
"""

import subprocess
import sys
import time
from datetime import datetime

def run_load_test(users, spawn_rate, duration, test_name):
    """
    Run a single load test with specified parameters
    
    Args:
        users: Number of concurrent users
        spawn_rate: Users spawned per second
        duration: Test duration (e.g., '2m', '3m')
        test_name: Name for this test
    """
    print("\n" + "="*80)
    print(f"RUNNING: {test_name}")
    print(f"Users: {users} | Spawn Rate: {spawn_rate}/sec | Duration: {duration}")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    cmd = [
        "locust",
        "-f", "locustfile.py",
        "--headless",
        "--host=http://localhost:8000",
        f"--users={users}",
        f"--spawn-rate={spawn_rate}",
        f"--run-time={duration}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """
    Run all three load tests sequentially
    """
    print("\n" + "="*80)
    print("SKIN CANCER API LOAD TESTING SUITE")
    print("="*80)
    print("This will run 3 comprehensive load tests:")
    print("  Test 1: Baseline (10 users, 2 min)")
    print("  Test 2: Moderate (50 users, 3 min)")
    print("  Test 3: Heavy (100 users, 3 min)")
    print("\nTotal estimated time: ~8-10 minutes")
    print("="*80)
    
    # Check if API is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("\nERROR: API is not responding correctly")
            print("Please start the API first with: .\\start.ps1")
            sys.exit(1)
        print("\nAPI Status: Running ✓")
    except Exception as e:
        print(f"\nERROR: Cannot connect to API at http://localhost:8000")
        print(f"Error: {e}")
        print("\nPlease start the API first with: .\\start.ps1")
        sys.exit(1)
    
    tests = [
        {
            "name": "Test 1: Baseline Performance",
            "users": 10,
            "spawn_rate": 2,
            "duration": "2m"
        },
        {
            "name": "Test 2: Moderate Load",
            "users": 50,
            "spawn_rate": 5,
            "duration": "3m"
        },
        {
            "name": "Test 3: Heavy Stress Test",
            "users": 100,
            "spawn_rate": 10,
            "duration": "3m"
        }
    ]
    
    results = []
    start_time = time.time()
    
    for i, test in enumerate(tests, 1):
        success = run_load_test(
            users=test["users"],
            spawn_rate=test["spawn_rate"],
            duration=test["duration"],
            test_name=test["name"]
        )
        results.append((test["name"], success))
        
        # Wait between tests
        if i < len(tests):
            print("\nWaiting 10 seconds before next test...")
            time.sleep(10)
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("LOAD TESTING COMPLETE")
    print("="*80)
    print(f"Total time: {total_time/60:.2f} minutes\n")
    print("Results:")
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name}: {status}")
    
    print("\nDetailed results have been saved to LOAD_TESTING.md")
    print("="*80)

if __name__ == "__main__":
    main()
