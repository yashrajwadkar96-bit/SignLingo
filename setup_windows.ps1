# Installs dependencies into the existing .venv and trains the alphabet CNN
# if its weights are not already on disk.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with the default Python..."
    py -3 -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip install -e .

if (-not (Test-Path ".\models\alphabet_cnn.keras")) {
    Write-Host "Training the fingerspelling CNN (a few minutes on CPU)..."
    & $py scripts\train_alphabet.py
}

Write-Host ""
Write-Host "Ready."
Write-Host "  Webcam : .\.venv\Scripts\python.exe scripts\demo_webcam.py"
Write-Host "  App    : .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8510"
