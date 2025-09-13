@echo off
echo 🔧 Setting up development environment...

:: Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Docker found
    set DOCKER_AVAILABLE=true
) else (
    echo ⚠️  Docker not found - will use native Python setup
    set DOCKER_AVAILABLE=false
)

:: Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚡ Please edit .env and add your GITHUB_PAT token
)

:: Option 1: Docker setup (recommended)
if "%DOCKER_AVAILABLE%" == "true" (
    echo 🐳 Setting up Docker environment...
    
    :: Build and start containers
    docker-compose up --build -d
    
    echo ✅ Docker setup complete!
    echo 🌐 Application running at:
    echo    - Development: http://localhost:5000
    echo    - Production-like: docker-compose --profile prod up ^(port 5001^)
    echo.
    echo 📱 Useful commands:
    echo    - View logs: docker-compose logs -f
    echo    - Restart: docker-compose restart
    echo    - Stop: docker-compose down
    
) else (
    :: Option 2: Native Python setup
    echo 🐍 Setting up native Python environment...
    
    :: Check Python version
    python --version
    
    :: Create virtual environment
    if not exist venv (
        echo Creating virtual environment...
        python -m venv venv
    )
    
    :: Activate virtual environment
    call venv\Scripts\activate.bat
    
    :: Install dependencies
    echo Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    :: Download NLTK data
    echo Downloading NLTK data...
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
    
    echo ✅ Native Python setup complete!
    echo 🌐 To start the application:
    echo    venv\Scripts\activate
    echo    python app.py
)

echo.
echo 🎉 Development environment is ready!
pause