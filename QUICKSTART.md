# 🚀 Quick Start Guide

Get the Skin Cancer Detection app running in 5 minutes!

## Option 1: Docker Compose (Recommended - One Command!)

```bash
# Clone repository
git clone https://github.com/IrutingaboRaissa/Raissa_IRUTINGABO_Summative-assignment-MLOP.git
cd Raissa_IRUTINGABO_Summative-assignment-MLOP

# Start everything
docker-compose up
```

**That's it!** Now open:
- UI: http://localhost:8501
- API: http://localhost:8000/docs
- Locust: http://localhost:8089

## Option 2: Manual Setup (3 Terminals)

### Terminal 1 - API 🔌
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

### Terminal 2 - UI 🌐
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
streamlit run ui/app.py
```

### Terminal 3 - Locust 🐝
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
locust -f tests/locustfile.py --host=http://localhost:8000
```

## 🧪 Quick Test

### Test via UI
1. Go to http://localhost:8501
2. Upload image from `data/test/` folder
3. Click "Analyze Image"
4. See results!

### Test via API
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/test/sample_image.jpg"
```

### Load Test with Locust
1. Go to http://localhost:8089
2. Number of users: 10
3. Spawn rate: 2
4. Host: http://localhost:8000
5. Start swarming!

## 📊 What You'll See

### Streamlit UI
- Image upload interface
- Real-time predictions
- Confidence scores
- Visual feedback

### FastAPI Docs
- Interactive API documentation
- Try endpoints directly
- See request/response schemas

### Locust Dashboard
- Requests per second (RPS)
- Response times
- Failure rates
- Charts and graphs

## 🛑 Stop Everything

```bash
# Docker:
docker-compose down

# Manual: Press Ctrl+C in each terminal
```

## ⚡ Common Issues

**Port 8000 busy?**
```bash
# Kill process
lsof -ti:8000 | xargs kill -9  # Mac/Linux
# Or change port in docker-compose.yml
```

**Models missing?**
```bash
git lfs pull
ls -lh models/  # Should see *.pth files
```

**Docker out of memory?**
- Increase Docker memory to 6GB+
- Docker Desktop → Settings → Resources

## 🎯 Next Steps

1. ✅ Run the app (you're here!)
2. 📖 Read [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment
3. 🧪 Check [tests/](tests/) for more testing options
4. 📊 Review [docs/](docs/) for model details

## 💡 Tips

- Use Chrome/Firefox for best UI experience
- Upload images in JPG/PNG format
- Keep images under 10MB
- Check logs if something fails:
  ```bash
  docker-compose logs -f
  ```

## 🆘 Need Help?

- Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions
- Review [README.md](README.md) for project overview
- Open an issue on GitHub

---

**Ready to deploy to production?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for AWS, GCP, Azure, and Heroku guides!
