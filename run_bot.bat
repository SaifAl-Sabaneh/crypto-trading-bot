@echo off
:: Navigate to the trading bot project directory
cd /d "c:\Users\Asus\Desktop\Project crypto"

:: Run the trading bot orchestrator and append the console logs to a log file
python main.py >> execution_history.log 2>&1
