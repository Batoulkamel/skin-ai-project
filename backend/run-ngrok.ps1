# Tunnel local FastAPI to the internet (ngrok).
# One-time setup: https://dashboard.ngrok.com/signup then:
#   ngrok config add-authtoken YOUR_TOKEN
# Ensure the API is already running, e.g.:
#   .\.venv\Scripts\Activate.ps1
#   uvicorn main:app --host 127.0.0.1 --port 8000

$port = if ($env:NGROK_PORT) { $env:NGROK_PORT } else { "8000" }
Write-Host "Starting ngrok -> http://127.0.0.1:$port"
Write-Host "Local inspector: http://127.0.0.1:4040"
Write-Host "Live UI with this API: serve frontend then open index with ?api=PUBLIC_HTTPS_URL"
ngrok http "127.0.0.1:$port"
