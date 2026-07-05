@echo off
echo ========================================================
echo Credit Card Approval Prediction System - Startup Script
echo ========================================================

echo.
echo [1/4] Installing requirements...
pip install -r requirements.txt

echo.
echo [2/4] Generating synthetic dataset...
python generate_data.py

echo.
echo [3/4] Preprocessing data and training Machine Learning Models...
python preprocess.py
python train_model.py

echo.
echo [4/4] Starting the Flask Web Application...
echo Please open your web browser and go to http://127.0.0.1:5000
python app.py

pause
