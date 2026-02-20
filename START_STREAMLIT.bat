@echo off
echo ========================================
echo   GridWorld RL Explainer 2026
echo   Lancement de l'application Streamlit
echo ========================================
echo.

REM Activer l'environnement virtuel si présent
if exist venv\Scripts\activate.bat (
    echo [INFO] Activation de l'environnement virtuel...
    call venv\Scripts\activate.bat
)

REM Vérifier que streamlit est installé
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [ERREUR] Streamlit n'est pas installé!
    echo [INFO] Installation de streamlit...
    pip install streamlit plotly
)

REM Lancer l'application
echo.
echo [INFO] Lancement de l'application...
echo [INFO] Accédez à l'application dans votre navigateur:
echo [INFO] http://localhost:8501
echo.
streamlit run app\streamlit_app.py

pause
