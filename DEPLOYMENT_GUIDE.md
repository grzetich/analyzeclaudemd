# 🚀 Render.com Deployment Guide

## Prerequisites

1. **GitHub Personal Access Token**
   - Go to: https://github.com/settings/tokens
   - Generate new token (classic)
   - Only check `public_repo` scope
   - Copy the token

2. **GitHub Repository**
   - Push this code to a GitHub repository
   - Make sure all files are committed

## Render.com Deployment Steps

### 1. Create New Web Service
- Go to https://render.com/dashboard
- Click "New +" → "Web Service"
- Connect your GitHub repository

### 2. Configure Service Settings
```
Name: claude-topic-analyzer
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

### 3. Set Environment Variables
In Render dashboard, add:
```
GITHUB_PAT = your_github_token_here
```

### 4. Deploy
- Click "Create Web Service"
- Wait for build to complete (5-10 minutes)
- Your app will be available at: https://your-app-name.onrender.com

## Production Features

✅ **Daily Scheduled Analysis** - Runs at 3 AM GMT
✅ **GitHub API Integration** - Fetches real claude.md files
✅ **Historical Data Storage** - Tracks topics over time
✅ **Topic Evolution Analysis** - Shows trending patterns
✅ **Memory Management** - Optimized for free tier
✅ **Rate Limiting** - Handles GitHub API limits
✅ **Comprehensive Logging** - Full audit trail

## Free Tier Considerations

- **Memory**: 512MB (app optimized for this)
- **Compute**: Limited hours/month
- **Sleep**: Apps sleep after 15 min inactivity
- **Build Time**: ~5-10 minutes

## Monitoring

- Check logs in Render dashboard
- Analysis runs daily at 3 AM GMT
- Visit `/analyze` to trigger manual analysis
- Check `/api/topics` for current data

## Troubleshooting

**Build Fails**: Check Python version compatibility
**Analysis Fails**: Verify GITHUB_PAT is set correctly
**Memory Issues**: App includes automatic cleanup
**Rate Limits**: GitHub allows 5000 requests/hour with PAT

---

🎉 **Your production Claude.md Topic Analyzer will be live!**