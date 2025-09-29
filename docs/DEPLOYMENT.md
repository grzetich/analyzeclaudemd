# Render.com Deployment Guide

This application is ready for deployment on Render.com with persistent data storage.

## Prerequisites

1. GitHub repository with your code
2. Render.com account
3. GitHub Personal Access Token for API access

## Deployment Steps

### 1. PostgreSQL Database Setup

1. Log into Render.com dashboard
2. Create a new **PostgreSQL** service:
   - Name: `claude-analyzer-db` 
   - Database Name: `claude_analyzer`
   - User: `claude_user`
   - Note down the connection details

### 2. Web Service Setup

1. Create a new **Web Service**
2. Connect your GitHub repository
3. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: `Python 3.11+`

### 3. Environment Variables

Set the following environment variables in Render:

```bash
# Required: GitHub API access
GITHUB_PAT=your_github_personal_access_token_here

# Database (automatically set by Render when you link PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:port/database

# Optional: App configuration
PORT=10000  # Render sets this automatically
ANALYSIS_INTERVAL_HOURS=24  # How often to run analysis
```

### 4. PostgreSQL Database Connection

In your Web Service settings:
1. Go to "Environment" tab
2. Link to your PostgreSQL database
3. This automatically sets the `DATABASE_URL` environment variable

## Features in Production

### ✅ Persistent Storage
- **PostgreSQL Database**: All analysis runs and topic data stored permanently
- **Automatic Migration**: Existing JSON data migrated to database on first deploy
- **No Data Loss**: Survives service restarts and redeployments

### ✅ Downloadable Data
- **Export Data**: `/api/export-data` - Download all analysis data as JSON
- **Download Logs**: `/api/download-logs` - Download application logs as ZIP
- **Analysis Details**: `/api/analysis-run/<id>` - Get specific analysis run details

### ✅ Topic Evolution Analysis
- **Historical Tracking**: Compare topics across multiple analysis runs
- **Similarity Metrics**: Cosine similarity and Jaccard similarity calculations
- **Interactive Visualization**: Charts showing topic stability and evolution

## Environment Detection

The application automatically detects the environment:

- **Local Development**: Uses SQLite database (`analysis.db`)
- **Render.com Production**: Uses PostgreSQL (via `DATABASE_URL`)

## Rate Limits & Constraints

### GitHub API Limits
- **With PAT**: 5,000 requests/hour, 30 requests/minute for code search
- **Without PAT**: 60 requests/hour (very limited)
- **Collection Limit**: 500 files per analysis run

### Render.com Free Tier
- **Build Time**: 10 minutes max
- **Memory**: 512MB 
- **Disk**: Ephemeral (that's why we use PostgreSQL)
- **Sleep**: Services sleep after 15 minutes of inactivity

## Monitoring & Logs

### Application Logs
- Access via Render dashboard logs
- Download via `/api/download-logs` endpoint
- Individual analysis logs stored in database

### Health Monitoring
- **Database Stats**: `/api/database-stats`
- **Analysis History**: `/api/topic-evolution`
- **Application Status**: Main dashboard shows last analysis status

## Troubleshooting

### Database Issues
```bash
# Check if DATABASE_URL is set
echo $DATABASE_URL

# Verify PostgreSQL connection in Render logs
# Look for "Database initialized successfully (PostgreSQL)"
```

### GitHub API Issues
```bash
# Check if GITHUB_PAT is set
# Look for "GitHub PAT configured" in logs
# Without PAT: "WARNING: GITHUB_PAT environment variable not set"
```

### Build Issues
```bash
# Common fixes:
# 1. Ensure Python 3.11+ in Render settings
# 2. Check requirements.txt has all dependencies
# 3. psycopg2-binary (not psycopg2) for PostgreSQL
```

## Local Testing

Test production setup locally:

```bash
# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/testdb"
export GITHUB_PAT="your_token"

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## Security Considerations

- GitHub PAT is read-only for public repositories
- No sensitive data stored or logged
- Database connection uses SSL in production
- All downloads are server-generated (no user uploads)

## Scaling

For high-traffic deployments:
- Upgrade to Render Pro for more resources
- Consider Redis for caching
- Implement rate limiting for API endpoints
- Add monitoring with external services

## Support

- Check Render.com logs for deployment issues
- Use `/api/database-stats` to verify data persistence
- Download logs via `/api/download-logs` for debugging