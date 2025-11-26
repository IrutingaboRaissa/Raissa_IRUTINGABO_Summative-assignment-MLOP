# 🐝 Locust Quick Start Guide

## ⚠️ IMPORTANT: How to Run Locust

**DON'T run with Python directly!** ❌
```powershell
# WRONG - Don't do this:
python locustfile.py
```

**DO run with the locust command!** ✅
```powershell
# CORRECT - Do this:
locust -f locustfile.py --host=http://localhost:8000
```

---

## Step-by-Step Instructions

### Step 1: Make Sure API is Running

```powershell
# Terminal 1 - Start the API first
python main.py api
```

Or if main.py doesn't work:
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Verify API is running by opening: http://localhost:8000/docs

---

### Step 2: Run Locust

```powershell
# Terminal 2 - Start Locust
locust -f locustfile.py --host=http://localhost:8000
```

**Or simply double-click:** `run_locust_simple.bat`

---

### Step 3: Open Locust Web UI

1. Open your browser
2. Go to: **http://localhost:8089**
3. You should see the Locust interface

---

### Step 4: Configure the Test

In the Locust web UI:

1. **Number of users (peak concurrency):** `10`
2. **Spawn rate (users per second):** `2`
3. **Host:** `http://localhost:8000` (should be pre-filled)
4. Click **"Start Swarming"** button

---

### Step 5: Watch the Results

You'll see real-time statistics:
- **RPS** (Requests Per Second)
- **Response Times** (min/avg/max)
- **Number of Users**
- **Failure Rate** (should be 0%)
- **Charts** showing performance over time

---

## If You Get Python/Module Errors

If you see errors like:
```
ModuleNotFoundError: No module named 'gevent._gevent_c_hub_local'
```

**Solution 1: Use Virtual Environment (Recommended)**

```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Now run locust
locust -f locustfile.py --host=http://localhost:8000
```

**Solution 2: Reinstall Locust**

```powershell
pip uninstall locust gevent greenlet -y
pip install locust --upgrade
```

**Solution 3: Use Python 3.9-3.11 (Not 3.13)**

Python 3.13 is very new and some packages aren't compatible yet.

```powershell
# Install Python 3.11 from python.org
# Then create venv with it:
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Commands Reference

```powershell
# Basic run
locust -f locustfile.py --host=http://localhost:8000

# Light load test
locust -f locustfile.py --host=http://localhost:8000 LightLoadUser --users 5 --spawn-rate 1

# Heavy load test
locust -f locustfile.py --host=http://localhost:8000 HeavyLoadUser --users 50 --spawn-rate 10

# Headless mode (no web UI, runs automatically)
locust -f locustfile.py --host=http://localhost:8000 --users 20 --spawn-rate 5 --run-time 2m --headless

# Generate HTML report
locust -f locustfile.py --host=http://localhost:8000 --users 20 --spawn-rate 5 --run-time 3m --html=locust_report.html --headless
```

---

## Troubleshooting

### Issue: "Connection refused"
- Make sure API is running: `python main.py api`
- Check API is at port 8000: http://localhost:8000/docs

### Issue: "Port 8089 already in use"
```powershell
# Use different port for Locust UI
locust -f locustfile.py --host=http://localhost:8000 --web-port=8090
# Then open: http://localhost:8090
```

### Issue: Module errors
- Make sure you're in virtual environment: `venv\Scripts\activate`
- Reinstall: `pip install locust --upgrade`

---

## What to Show in Your Video

1. ✅ Terminal showing Locust starting
2. ✅ Browser showing Locust UI (http://localhost:8089)
3. ✅ Configure: Users=10, Spawn rate=2
4. ✅ Click "Start Swarming"
5. ✅ Show the statistics table with all endpoints
6. ✅ Show the Charts tab with performance graphs
7. ✅ Explain the metrics (RPS, response time, etc.)
8. ✅ Show 0% failure rate
9. ✅ Stop the test

---

## Success Checklist

- ✅ API running on port 8000
- ✅ Locust command runs without errors
- ✅ Browser opens to http://localhost:8089
- ✅ Can start swarming
- ✅ See statistics for all 10 endpoints
- ✅ 0% failure rate
- ✅ Response times < 500ms

**You're ready for the video! 🎥**
