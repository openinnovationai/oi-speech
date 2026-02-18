#!/bin/bash
set -e

# Function to log messages
log() {
    echo "[startup.sh] $1"
}

# 1. Check if ASR_BACKEND is explicitly set
if [ -n "$ASR_BACKEND" ]; then
    log "ASR_BACKEND is explicitly set to: $ASR_BACKEND"
else
    # 2. Heuristic based on model name
    # Check ASR_MODEL
    MODEL_NAME="${ASR_MODEL}"
    
    if [[ "$MODEL_NAME" == *"omni"* ]] || [[ "$MODEL_NAME" == *"Omni"* ]]; then
        log "Model name '$MODEL_NAME' implies Omnilingual backend."
        export ASR_BACKEND="omnilingual"
    else
        log "Defaulting to Faster Whisper backend."
        export ASR_BACKEND="faster_whisper"
    fi
fi

# 3. Validation (Optional but recommended)
if [ "$ASR_BACKEND" = "omnilingual" ]; then
    if ! python3 -c "import omnilingual_asr" 2>/dev/null; then
        log "WARNING: Omnilingual backend selected but 'omnilingual_asr' not found. Did you build with INSTALL_OMNILINGUAL=true?"
    fi
fi

# 4. Start the application
log "Starting application with ASR_BACKEND=$ASR_BACKEND..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8080
