if (!(Test-Path -Path "documents")) {
    New-Item -ItemType Directory -Path "documents" | Out-Null
    Write-Host "Created 'documents' directory."
}
if (!(Test-Path -Path ".env")) {
    Set-Content -Path ".env" -Value "GEMINI_API_KEY=your_api_key_here`n"
    Write-Host "Created dummy '.env' file. Please edit it to add your real API key!"
}

Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

Write-Host "Installing/Verifying required pip packages..."
pip install -q google-genai python-dotenv chromadb tiktoken pypdf markitdown

Write-Host "Running pipeline..."
cd src
python test_pipeline.py
cd ..
