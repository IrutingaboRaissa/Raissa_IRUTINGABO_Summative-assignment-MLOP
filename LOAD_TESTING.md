# Load Testing with Locust - Instructions and Results

## Overview
This document provides instructions for running load tests on the Skin Cancer Classification API using Locust, and documents the results.

## Prerequisites
```powershell
pip install locust
```

## Running Load Tests

### Step 1: Start the API Server
```powershell
# In terminal 1
python api.py
```

### Step 2: Run Locust
```powershell
# In terminal 2
locust -f locustfile.py --host=http://localhost:8000
```

### Step 3: Access Locust Web UI
Open your browser and navigate to: `http://localhost:8089`

### Step 4: Configure Test Parameters
- **Number of users**: Start with 10, then 50, then 100
- **Spawn rate**: 5 users per second
- **Host**: http://localhost:8000

### Step 5: Run Tests
1. Click "Start Swarming"
2. Let it run for 2-3 minutes
3. Download the results (Statistics, Charts, Download Data)
4. Stop the test
5. Repeat with different user counts

## Test Scenarios

### Scenario 1: Light Load (10 concurrent users)
- Users: 10
- Spawn rate: 2/s
- Duration: 2 minutes
- Purpose: Baseline performance

### Scenario 2: Medium Load (50 concurrent users)
- Users: 50
- Spawn rate: 5/s
- Duration: 3 minutes
- Purpose: Normal production load

### Scenario 3: Heavy Load (100 concurrent users)
- Users: 100
- Spawn rate: 10/s
- Duration: 3 minutes
- Purpose: Stress testing

## Expected Metrics to Record
- **Total Requests**: Number of requests completed
- **Failures**: Number of failed requests
- **Median Response Time**: 50th percentile
- **95th Percentile**: Response time for 95% of requests
- **99th Percentile**: Response time for 99% of requests
- **Requests per Second (RPS)**: Throughput
- **Average Response Time**: Mean response time

---

## Load Test Results

### Test Environment
- **Date**: [FILL IN DATE]
- **System**: Windows
- **CPU**: [FILL IN]
- **RAM**: [FILL IN]
- **Model**: MobileNetV2
- **Device**: CPU (no GPU)

---

### Scenario 1: Light Load (10 concurrent users)

**Configuration:**
- Users: 10
- Spawn Rate: 2/s
- Duration: 2 minutes
- Test Date: [FILL IN]

**Results:**

| Endpoint | Requests | Failures | Median (ms) | 95th (ms) | 99th (ms) | Avg (ms) | RPS |
|----------|----------|----------|-------------|-----------|-----------|----------|-----|
| /predict | [FILL]   | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |
| /health  | [FILL]   | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |
| **Total** | [FILL]  | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |

**Analysis:**
- [FILL IN YOUR OBSERVATIONS]
- Example: "System handled light load easily with median response time under 500ms"

---

### Scenario 2: Medium Load (50 concurrent users)

**Configuration:**
- Users: 50
- Spawn Rate: 5/s
- Duration: 3 minutes
- Test Date: [FILL IN]

**Results:**

| Endpoint | Requests | Failures | Median (ms) | 95th (ms) | 99th (ms) | Avg (ms) | RPS |
|----------|----------|----------|-------------|-----------|-----------|----------|-----|
| /predict | [FILL]   | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |
| /health  | [FILL]   | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |
| **Total** | [FILL]  | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |

**Analysis:**
- [FILL IN YOUR OBSERVATIONS]

---

### Scenario 3: Heavy Load (100 concurrent users)

**Configuration:**
- Users: 100
- Spawn Rate: 10/s
- Duration: 3 minutes
- Test Date: [FILL IN]

**Results:**

| Endpoint | Requests | Failures | Median (ms) | 95th (ms) | 99th (ms) | Avg (ms) | RPS |
|----------|----------|----------|-------------|-----------|-----------|----------|-----|
| /predict | [FILL]   | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |
| /health  | [FILL]   | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |
| **Total** | [FILL]  | [FILL]   | [FILL]      | [FILL]    | [FILL]    | [FILL]   | [FILL] |

**Analysis:**
- [FILL IN YOUR OBSERVATIONS]

---

## Docker Container Scaling Tests

### Single Container Test
**Configuration:**
- Containers: 1
- Users: 50
- Duration: 2 minutes

**Results:**
- Average Response Time: [FILL] ms
- 95th Percentile: [FILL] ms
- Requests/sec: [FILL]
- Failure Rate: [FILL]%

### Two Containers Test (Load Balanced)
**Configuration:**
- Containers: 2
- Users: 50
- Duration: 2 minutes

**Results:**
- Average Response Time: [FILL] ms
- 95th Percentile: [FILL] ms
- Requests/sec: [FILL]
- Failure Rate: [FILL]%
- **Improvement**: [FILL]% faster than single container

### Three Containers Test (Load Balanced)
**Configuration:**
- Containers: 3
- Users: 50
- Duration: 2 minutes

**Results:**
- Average Response Time: [FILL] ms
- 95th Percentile: [FILL] ms
- Requests/sec: [FILL]
- Failure Rate: [FILL]%
- **Improvement**: [FILL]% faster than single container

---

## Key Findings

### Performance Bottlenecks
1. [FILL IN - e.g., "Model inference time is the primary bottleneck"]
2. [FILL IN - e.g., "CPU utilization reaches 90% under heavy load"]
3. [FILL IN]

### Scalability Observations
1. [FILL IN - e.g., "Response time degrades linearly after 50 concurrent users"]
2. [FILL IN - e.g., "Adding containers improves throughput by X%"]
3. [FILL IN]

### Recommendations
1. [FILL IN - e.g., "Deploy with GPU for faster inference"]
2. [FILL IN - e.g., "Use load balancer with 2-3 containers for production"]
3. [FILL IN - e.g., "Consider model quantization for CPU optimization"]

---

## Screenshots
Include screenshots of:
1. Locust dashboard showing request statistics
2. Response time graphs
3. Docker container resource usage
4. API logs during high load

*(Paste screenshot links or embed images here)*

---

## Conclusion
[FILL IN - Summary of load testing results and system readiness for production]
