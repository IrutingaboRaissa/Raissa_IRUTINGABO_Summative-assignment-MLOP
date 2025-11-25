# Load Testing Results

## Overview
This document presents comprehensive load testing results for the Skin Cancer Detection API using Locust. Three tests were conducted to evaluate system performance under varying concurrent user loads.

## Test Configuration

### Tool: Locust 2.42.5
- **Host**: http://localhost:8000
- **User Classes**:
  - `SkinCancerUser`: Predicts skin lesion images (5 sample images, 1-3s think time)
  - `LightLoadUser`: Health checks and metrics monitoring (2-5s think time)
  - `HeavyLoadUser`: Batch predictions with 5-10 images (5-10s think time)

### Test Scenarios

| Test | Users | Spawn Rate | Duration | Purpose |
|------|-------|------------|----------|---------|
| 1    | 10    | 2/sec      | 2 min    | Baseline performance |
| 2    | 50    | 5/sec      | 3 min    | Moderate load |
| 3    | 100   | 10/sec     | 3 min    | Heavy stress test |

### Test Environment
- **Date**: November 25, 2025
- **System**: Windows
- **Model**: MobileNetV2
- **Device**: CPU (no GPU)
- **Python**: 3.11
- **FastAPI**: Latest

---

## Test 1: Baseline Performance (10 Users)

### Summary
- **Total Requests**: 1,236
- **Failure Rate**: 0.00%
- **Request Rate**: 10.47 req/s
- **Average Response Time**: 173ms
- **Test Duration**: 120 seconds

### Response Times by Endpoint

| Endpoint | Requests | Avg (ms) | Min (ms) | Max (ms) | Median (ms) |
|----------|----------|----------|----------|----------|-------------|
| GET /health | 44 | 199 | 2 | 2,197 | 120 |
| GET /metrics | 32 | 166 | 3 | 1,133 | 120 |
| POST /predict | 1,151 | 170 | 40 | 2,342 | 120 |
| POST /predict/batch | 9 | 474 | 178 | 2,230 | 120 |
| **Aggregated** | **1,236** | **173** | **1** | **2,342** | **120** |

### Response Time Percentiles
- **50%**: 120ms
- **66%**: 150ms
- **75%**: 190ms
- **90%**: 310ms
- **95%**: 430ms
- **99%**: 1,400ms

### Analysis
✅ **Excellent baseline performance**
- Zero failures under light load
- Sub-200ms average response times across all endpoints
- 90th percentile under 350ms
- System handles 10 concurrent users effortlessly

---

## Test 2: Moderate Load (50 Users)

### Summary
- **Total Requests**: 2,576
- **Failure Rate**: 1.13% (29 failures)
- **Request Rate**: 14.56 req/s
- **Average Response Time**: 1,861ms
- **Test Duration**: 180 seconds

### Response Times by Endpoint

| Endpoint | Requests | Avg (ms) | Min (ms) | Max (ms) | Median (ms) |
|----------|----------|----------|----------|----------|-------------|
| GET /health | 198 | 1,793 | 215 | 6,124 | 1,700 |
| GET /metrics | 110 | 1,741 | 791 | 6,191 | 1,700 |
| POST /predict | 2,224 | 1,870 | 9 | 9,032 | 1,700 |
| POST /predict/batch | 44 | 1,993 | 788 | 4,922 | 1,700 |
| **Aggregated** | **2,576** | **1,861** | **9** | **9,032** | **1,700** |

### Response Time Percentiles
- **50%**: 1,700ms
- **66%**: 1,800ms
- **75%**: 2,100ms
- **90%**: 2,900ms
- **95%**: 3,400ms
- **99%**: 5,200ms

### Error Analysis
**Error Type**: `ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host')`
- **Occurrences**: 29
- **Affected Endpoint**: POST /predict
- **Pattern**: Windows socket error indicating server closed connections under stress

### Analysis
⚠️ **Significant performance degradation**
- Response times increased **10.7x** compared to Test 1 (173ms → 1,861ms)
- 1.13% failure rate due to connection resets
- Maximum response time reached 9.03 seconds
- System shows clear bottleneck at 50 concurrent users
- Connection pool or timeout configuration likely needs adjustment

---

## Test 3: Heavy Stress Test (100 Users)

### Summary
- **Total Requests**: 2,799
- **Failure Rate**: 2.04% (57 failures)
- **Request Rate**: 15.82 req/s
- **Average Response Time**: 4,540ms
- **Test Duration**: 177 seconds

### Response Times by Endpoint

| Endpoint | Requests | Avg (ms) | Min (ms) | Max (ms) | Median (ms) |
|----------|----------|----------|----------|----------|-------------|
| GET /health | 245 | 4,177 | 267 | 69,368 | 3,600 |
| GET /metrics | 129 | 4,131 | 435 | 40,113 | 3,500 |
| POST /predict | 2,369 | 4,599 | 358 | 68,355 | 3,700 |
| POST /predict/batch | 56 | 4,584 | 1,422 | 55,339 | 3,600 |
| **Aggregated** | **2,799** | **4,540** | **267** | **69,368** | **3,700** |

### Response Time Percentiles
- **50%**: 3,700ms
- **66%**: 4,100ms
- **75%**: 4,300ms
- **80%**: 4,500ms
- **90%**: 5,300ms
- **95%**: 6,100ms
- **98%**: 22,000ms
- **99%**: 44,000ms
- **99.9%**: 67,000ms

### Error Analysis
**Error Type**: `ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host')`
- **Occurrences**: 57
- **Affected Endpoint**: POST /predict
- **Pattern**: Nearly doubled error rate compared to Test 2

### Analysis
❌ **System under severe stress**
- Response times increased **26.2x** compared to Test 1 (173ms → 4,540ms)
- 2.04% failure rate (doubled from Test 2)
- Maximum response time: **69.4 seconds**
- Request rate only increased marginally (14.56 → 15.82 req/s)
- System at capacity limit with 100 concurrent users
- Tail latencies extremely high (99th percentile: 44 seconds)

---

## Performance Comparison

### Response Time Progression

| Test | Users | Avg Response (ms) | Multiplier vs Baseline |
|------|-------|-------------------|------------------------|
| 1    | 10    | 173               | 1.0x                   |
| 2    | 50    | 1,861             | 10.7x                  |
| 3    | 100   | 4,540             | 26.2x                  |

### Failure Rate Progression

| Test | Users | Total Requests | Failures | Failure Rate |
|------|-------|----------------|----------|--------------|
| 1    | 10    | 1,236          | 0        | 0.00%        |
| 2    | 50    | 2,576          | 29       | 1.13%        |
| 3    | 100   | 2,799          | 57       | 2.04%        |

### Request Rate Analysis

| Test | Users | Request Rate (req/s) | Efficiency |
|------|-------|---------------------|------------|
| 1    | 10    | 10.47               | 1.05 req/s per user |
| 2    | 50    | 14.56               | 0.29 req/s per user |
| 3    | 100   | 15.82               | 0.16 req/s per user |

**Observation**: Request rate shows sub-linear scaling, indicating the system reaches maximum throughput around 15-16 req/s regardless of user count.

---

## Key Findings

### 1. Performance Bottleneck
The system exhibits severe performance degradation beyond 10 concurrent users:
- **10 → 50 users**: 10.7x slower response times
- **50 → 100 users**: Additional 2.4x degradation
- Non-linear scaling suggests resource exhaustion (CPU, memory, or I/O)

### 2. Connection Reset Errors
Consistent `ConnectionResetError(10054)` pattern indicates:
- Server forcibly closing connections under load
- Possible causes:
  - Connection pool size too small
  - Keep-alive timeout misconfiguration
  - Backend server worker/thread limits
  - Resource exhaustion causing connection drops

### 3. Throughput Ceiling
Maximum throughput plateaus at ~16 req/s regardless of user count:
- **50 users**: 14.56 req/s
- **100 users**: 15.82 req/s
- Additional users only increase latency, not throughput
- Indicates a hard resource limit (likely model inference or single-threaded processing)

### 4. Tail Latency Issues
Test 3 shows extreme tail latencies:
- 99th percentile: 44 seconds
- 99.9th percentile: 67 seconds
- Maximum: 69.4 seconds
- Some requests take 400x longer than average

---

## Recommendations for Production

### Immediate Improvements

1. **Increase Worker Processes**
   ```python
   # Uvicorn configuration
   uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
   ```

2. **Configure Connection Pooling**
   ```python
   # Adjust connection limits
   app.add_middleware(
       CORSMiddleware,
       max_connections=500,
       keepalive_timeout=75
   )
   ```

3. **Add Request Timeout**
   ```python
   @app.middleware("http")
   async def timeout_middleware(request: Request, call_next):
       try:
           return await asyncio.wait_for(call_next(request), timeout=30.0)
       except asyncio.TimeoutError:
           return Response("Request timeout", status_code=504)
   ```

### Optimization Strategies

1. **Model Inference Optimization**
   - Batch inference for multiple simultaneous requests
   - Model quantization for faster inference
   - GPU utilization if available
   - Consider model serving frameworks (TorchServe, TensorFlow Serving)

2. **Caching Layer**
   - Cache predictions for identical images
   - Redis or Memcached for distributed caching
   - TTL-based cache invalidation

3. **Load Balancing**
   - Deploy multiple API instances
   - Nginx or HAProxy for load distribution
   - Horizontal scaling with container orchestration (Kubernetes)

4. **Asynchronous Processing**
   - Queue-based architecture for prediction requests
   - Celery or RabbitMQ for task distribution
   - Return job ID immediately, poll for results

### Monitoring & Alerting

1. **Metrics to Track**
   - Request latency (p50, p95, p99)
   - Error rate by endpoint
   - Active connections
   - CPU/memory utilization
   - Model inference time

2. **Alerting Thresholds**
   - Error rate > 1%
   - P95 latency > 3 seconds
   - Connection pool utilization > 80%

---

## Production Readiness Assessment

### Current State
| Criterion | Status | Notes |
|-----------|--------|-------|
| **Light Load (≤10 users)** | ✅ Pass | Excellent performance |
| **Moderate Load (≤50 users)** | ⚠️ Marginal | 1.13% error rate, degraded latency |
| **Heavy Load (≤100 users)** | ❌ Fail | 2% error rate, 4.5s avg latency |
| **Throughput** | ⚠️ Limited | Caps at ~16 req/s |
| **Reliability** | ❌ Needs Work | Connection resets under load |

### Verdict
The system is **NOT production-ready** for high-traffic scenarios. It performs well under light load but requires significant optimization for:
- Connection handling
- Concurrent request processing
- Resource management
- Error resilience

---

## Test Execution Commands

```powershell
# Test 1: Baseline
locust -f locustfile.py --headless --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 2m

# Test 2: Moderate Load
locust -f locustfile.py --headless --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 3m

# Test 3: Stress Test
locust -f locustfile.py --headless --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 3m
```

---

## Conclusion

This load testing exercise reveals a system that functions well at small scale but requires architectural improvements for production deployment. The primary bottleneck appears to be single-threaded model inference combined with limited connection handling. Implementing the recommended optimizations (multi-worker setup, async processing, caching) would significantly improve performance under load.

**Next Steps**: Address connection pool configuration, implement worker-based deployment, and consider asynchronous task processing for prediction requests.
