@echo off
REM Script de lancement de l'application mltp pour Windows
REM ==================================================================

echo.
echo ================================
echo   APPLICATION machine learning TP2
echo ================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    echo Telechargez Python depuis https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python detecte

REM Vérifier si Streamlit est installé
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Streamlit n'est pas installe
    echo [INFO] Installation des dependances en cours...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation
        pause
        exit /b 1
    )
)

echo [OK] Streamlit installe

REM Vérifier si les fichiers CSV existe
if not exist "auto-mpg.csv" (
    echo.
    echo [ATTENTION] Le fichier auto-mp.csv n'est pas trouve
    echo Assurez-vous qu'il est dans le meme dossier que ce script
    echo.
    set /p continue="Continuer quand meme? (o/n): "
    if /i not "%continue%"=="o" exit /b 1
)
if not exist "census-data2015.csv" (
    echo.
    echo [ATTENTION] Le fichier census-data2015.csv n'est pas trouve
    echo Assurez-vous qu'il est dans le meme dossier que ce script
    echo.
    set /p continue="Continuer quand meme? (o/n): "
    if /i not "%continue%"=="o" exit /b 1
)

echo.
echo ================================
echo   DEMARRAGE DE L'APPLICATION
echo ================================
echo.
echo [INFO] L'application va s'ouvrir dans votre navigateur
echo [INFO] URL: http://localhost:8502
echo.
echo [ASTUCE] Pour arreter: Fermez cette fenetre ou appuyez sur Ctrl+C
echo.

REM Lancer Streamlit
streamlit run mltp.py

pause