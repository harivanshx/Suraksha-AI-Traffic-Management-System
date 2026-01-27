# Deploy AI Traffic Management System on Render

## Complete Step-by-Step Deployment Guide

### Prerequisites
- ✓ GitHub account (to host your code)
- ✓ Render account (https://render.com)
- ✓ Project code ready (already have it!)
- ✓ Gemini API key (from https://aistudio.google.com/app/apikey)

---

## Step 1: Prepare Your Project

### 1.1 Make sure .env is in .gitignore
This prevents secrets from being committed:

```bash
# Check .gitignore exists
cat .gitignore
```

You should see:
```
.env
__pycache__/
*.pyc
.DS_Store
uploads/
flask_session/
logs/
```

### 1.2 Verify files are ready
All essential files needed:
- ✓ `app.py` - Main application
- ✓ `web_processor.py` - Traffic processor
- ✓ `src/` - Source modules
- ✓ `templates/` - HTML files
- ✓ `static/` - CSS & JS
- ✓ `requirements.txt` - Dependencies
- ✓ `render.yaml` - Render config
- ✓ `Procfile` - Process file
- ✓ `yolo11n.pt` - ML model

---

## Step 2: Create GitHub Repository

### 2.1 Initialize Git locally
```bash
cd "D:\E2E Projects\End to end Projects\Ai based"
git init
git add .
git commit -m "Initial commit: AI Traffic Management System"
```

### 2.2 Create repository on GitHub
1. Go to https://github.com/new
2. Repository name: `ai-traffic-management`
3. Description: "AI-powered traffic signal optimization system"
4. Choose: Public (easier for Render) or Private
5. Click "Create repository"

### 2.3 Push to GitHub
```bash
# Add remote (replace USERNAME and REPO)
git remote add origin https://github.com/USERNAME/ai-traffic-management.git

# Push code
git branch -M main
git push -u origin main

# Verify
git remote -v
```

---

## Step 3: Set Up on Render

### 3.1 Create New Web Service
1. Go to https://dashboard.render.com
2. Click "New" → "Web Service"
3. Select "Deploy an existing repository"
4. Click "Connect GitHub" (if not already connected)
5. Authorize Render to access GitHub
6. Find and select your repository: `ai-traffic-management`
7. Click "Connect"

### 3.2 Configure Service Settings

#### Basic Information
| Setting | Value |
|---------|-------|
| Name | `ai-traffic-management` |
| Environment | `Python 3` |
| Region | `Oregon` (or closest to you) |
| Branch | `main` |

#### Build Command
Should auto-detect from `render.yaml` or `Procfile`:
```
pip install -r requirements.txt
```

#### Start Command
```
gunicorn app:app
```

#### Instance Type
- **FREE** (first month free, sleeps after 15 min inactivity)
- **Paid** (recommended for production)

---

## Step 4: Configure Environment Variables

### 4.1 Add Environment Variables in Render

In the Render dashboard for your service, go to **Environment**:

```
FLASK_ENV=production
SECRET_KEY=your-random-secret-key-here-change-this
GEMINI_API_KEY=your-gemini-api-key-here
PYTHON_VERSION=3.11.0
```

**Getting each value:**

1. **GEMINI_API_KEY** (REQUIRED for chatbot)
   - Go to: https://aistudio.google.com/app/apikey
   - Click "Create API Key"
   - Copy the key (starts with `AIza...`)
   - Paste into Render environment

2. **SECRET_KEY** (for Flask sessions)
   - Generate random key in Python:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   # Copy output like: 3f4a8b9c2e1d6f5a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e
   ```
   - Paste into Render environment

3. **FLASK_ENV**
   - Set to `production`

### 4.2 Save Environment Variables
- Click "Save"
- Render will automatically redeploy

---

## Step 5: Deploy!

### 5.1 Manual Deploy
After saving environment variables, Render will auto-deploy:
1. Watch the deploy logs in dashboard
2. Should see:
   ```
   ✓ Building...
   ✓ Installing dependencies...
   ✓ Starting web service...
   ✓ Live on https://ai-traffic-management.onrender.com
   ```

### 5.2 Check Deployment Status
- Green dot = Running ✓
- Yellow dot = Building
- Red dot = Error

---

## Step 6: Access Your App

### 6.1 Open Live URL
Once deployed, Render gives you a URL like:
```
https://ai-traffic-management.onrender.com
```

Click the link or paste in browser.

### 6.2 Test the App
1. Go to: `https://your-app-name.onrender.com`
2. Should see landing page
3. Upload 4 traffic images
4. Click "Run Simulation"
5. Check results page with chatbot

---

## Troubleshooting

### Issue: "Build failed"
**Check logs:**
1. Go to Dashboard → Your Service
2. Click "Logs" tab
3. Look for error messages

**Common causes:**
- Missing `requirements.txt` → Add file with all dependencies
- Wrong Python version → Check render.yaml
- Syntax errors in code → Fix and push to GitHub

### Issue: "App not loading" (500 error)
**Check:**
1. Is GEMINI_API_KEY set in environment?
2. Are .env variables in Render dashboard (NOT in .env file)?
3. Check service logs for error details

### Issue: "Chatbot not working"
**Verify:**
- GEMINI_API_KEY is correct
- Key hasn't been revoked
- API is enabled in Google Cloud

### Issue: "Free tier keeps sleeping"
**Solution:**
- Upgrade to Paid plan ($7+/month)
- Or accept 15-minute inactivity timeout

### Issue: "File upload fails"
**Check:**
- Free tier has limited disk space (~100MB)
- Uploaded files are temporary
- Use image files, not large videos

---

## Important Notes for Production

### Security
- [ ] Change `SECRET_KEY` in Render environment
- [ ] Set `FLASK_ENV=production`
- [ ] Use HTTPS (automatic with Render)
- [ ] Rotate Gemini API key regularly
- [ ] Don't commit `.env` file

### Performance
- Render free tier: ~250MB RAM
- YOLO model: ~25MB
- Good for 1-2 concurrent users
- For more: Upgrade to paid plan

### Disk Space
- Render: 100GB available
- Each image: ~1-5MB
- Clean up old uploads regularly
- Or add cleanup script

### Limitations
- Free tier: Sleeps after 15 min
- Build time: ~5 minutes
- Cold start: ~30 seconds on free tier

---

## Recommended Render Plan

| Tier | Price | Best For | Startup | RAM |
|------|-------|----------|---------|-----|
| **Free** | $0 | Testing | ~30s | 512MB |
| **Starter** | $7/mo | Small | ~5s | 512MB |
| **Standard** | $25/mo | Production | ~2s | 1GB |

---

## Auto-Deployment from GitHub

Render automatically redeploys when you push to GitHub:

```bash
# Make changes locally
git add .
git commit -m "Fix chatbot styling"
git push origin main

# Render automatically:
# 1. Detects push
# 2. Builds new version
# 3. Deploys to live URL
# 4. Done! (within 5 minutes)
```

---

## Custom Domain (Optional)

To use your own domain:

1. Go to Render Dashboard → Settings
2. Add Custom Domain (e.g., `traffic-ai.com`)
3. Update DNS records at domain registrar
4. Render provides DNS instructions
5. SSL certificate: Automatic

---

## Monitoring & Logs

### View Logs
1. Dashboard → Your Service
2. Click "Logs" tab
3. See real-time application output

### Monitor Performance
1. Click "Metrics" tab
2. See CPU, RAM, requests
3. Plan upgrades if needed

### Alerts
- Set up email alerts (paid plan)
- Get notified of failures

---

## Updating Your App

### Deploy New Version
```bash
# Make changes
nano app.py  # or edit in VS Code

# Commit and push
git add .
git commit -m "Add new feature"
git push origin main

# Render auto-deploys within 5 minutes
```

### Zero Downtime Deploy
- Render handles automatically
- Old version runs until new one ready
- Users stay connected during deploy

---

## Production Checklist

Before deploying to production:

- [ ] All files committed to GitHub
- [ ] `.env` file NOT committed (in .gitignore)
- [ ] Environment variables set in Render dashboard
- [ ] GEMINI_API_KEY is valid
- [ ] Tested locally (works on http://localhost:5000)
- [ ] render.yaml has correct Python version
- [ ] requirements.txt has all dependencies
- [ ] No hardcoded secrets in code
- [ ] FLASK_ENV=production in Render
- [ ] SECRET_KEY is strong random value

---

## Example Render Dashboard Setup

```
Service: ai-traffic-management

General
├─ Name: ai-traffic-management
├─ Environment: Python 3
├─ Region: Oregon
├─ Branch: main
├─ Build Command: pip install -r requirements.txt
└─ Start Command: gunicorn app:app

Environment
├─ FLASK_ENV: production
├─ SECRET_KEY: 3f4a8b9c2e1d6f5a...
├─ GEMINI_API_KEY: AIzaSyCxxx...
└─ PYTHON_VERSION: 3.11.0

Settings
├─ Auto-Deploy: ON
├─ Health Check: /
└─ Instance Type: Free (or Starter)
```

---

## Getting Help

If deployment fails:

1. **Check Render Logs**
   - Dashboard → Logs tab
   - Look for error messages

2. **Common Issues**
   - Missing dependencies → Add to requirements.txt
   - Wrong Python version → Update render.yaml
   - API key invalid → Regenerate in Google Cloud
   - Out of memory → Upgrade to paid plan

3. **Test Locally First**
   ```bash
   python app.py
   # Should run on http://localhost:5000
   ```

4. **Render Documentation**
   - https://render.com/docs
   - Python deployment guide
   - Environment variables setup

5. **GitHub Issues**
   - Include deployment logs
   - Include error messages
   - Include render.yaml config

---

## Summary: Quick Deploy

1. ✓ Push code to GitHub
2. ✓ Go to https://render.com
3. ✓ New → Web Service → Select GitHub repo
4. ✓ Configure Python 3, gunicorn
5. ✓ Add Environment Variables (GEMINI_API_KEY, etc)
6. ✓ Click "Create Web Service"
7. ✓ Wait 5 minutes for build
8. ✓ Get live URL: `https://your-app-name.onrender.com`
9. ✓ Test the app!
10. ✓ Any changes: `git push` → Auto-deploys!

**That's it! Your app is live!** 🚀

---

## Costs

### Free Tier
- **First month**: Free
- **After**: $0.10/hour when running
- Sleeps after 15 min inactivity

### Paid Tiers
- **Starter**: $7/month (always on)
- **Standard**: $25/month (more power)
- **Custom**: Contact sales

**Estimate for this project:**
- Free tier: ~$7-15/month if always running
- Starter: $7/month (recommended)
- Good for 1-100 users

---

## Next Steps

1. Create GitHub account
2. Push project to GitHub
3. Create Render account
4. Deploy using this guide
5. Share live URL with others!

Your AI Traffic Management System will be live in ~10 minutes! 🎉
