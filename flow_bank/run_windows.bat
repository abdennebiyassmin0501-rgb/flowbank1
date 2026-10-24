@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Flow Bank ===
if not exist venv (
    echo Creating virtual environment...
    py -m venv venv 2>nul || python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing requirements (first time only, may take a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if not exist .streamlit\secrets.toml (
    echo.
    echo No login account yet. Create one now:
    python set_password.py
)
streamlit run app.py
pause
