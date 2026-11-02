# Quickstart Guide

Get the Granite Speech 5.0-2B demo running in under 10 minutes.

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.10–3.12 | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| (optional) Ollama | latest | [ollama.com](https://ollama.com) |

---

## Step 1 — Environment Setup

```bash
cd granite-speech5-test
bash scripts/setup.sh
```

This creates a `.venv` virtual environment and installs all dependencies.

---

## Step 2 — Configure

### Choose an LLM Backend

Open `config/config.yaml` and set the `llm.backend` field:

```yaml
llm:
  backend: "ollama"   # recommended for local-only, no API key needed
```

### Option A: Ollama (recommended for local use)

```bash
# Install and start Ollama
brew install ollama        # macOS
# or: https://ollama.com/download

# Pull a model
ollama pull granite3.3:8b
# or a smaller model:
ollama pull llama3.2:3b
```

### Option B: OpenAI / Anthropic (cloud)

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
nano .env
```

```env
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_BACKEND=openai
LLM_OPENAI_MODEL=gpt-4o-mini
```

---

## Step 3 — Launch

```bash
bash scripts/start.sh
```

The app starts in the background and prints:

```
✅  Application started  (PID 12345)

    ┌──────────────────────────────────────────────────────────────┐
    │  🔒  https://localhost:7860  (microphone enabled)         │
    └──────────────────────────────────────────────────────────────┘
```

Open **https://localhost:7860** in your browser.

> ⚠️ **Self-signed certificate warning** — on the first visit your browser will show
> a security warning ("Your connection is not private").
> Click **Advanced → Proceed to localhost (unsafe)** to continue.
> This is safe — the cert is only used to enable the microphone API locally.
> The warning appears only once; subsequent visits go straight to the app.

> If `setup.sh` could not generate the cert (openssl not found), the app falls
> back to HTTP and the microphone button will show "No microphone…".
> Install openssl and re-run `bash scripts/setup.sh` to fix this.

---

## Step 4 — Use the App

### Transcribe Tab
1. Click **Upload** or record from your microphone
2. Click **Transcribe**
3. The model is downloaded automatically on first use (~5 GB, ~2-5 min)

> ℹ️ Audio longer than ~2 minutes is automatically split into segments.

### Translate Tab
1. Upload audio
2. Select source and target languages
3. Click **Translate**

### Full Pipeline Tab
1. Upload audio
2. Optionally type an LLM instruction (e.g. "Summarise this meeting")
3. Click **Run Pipeline** — performs ASR then LLM processing in one click

### Configuration Tab
- View current runtime config
- Click **Reload** to apply changes from `config/config.yaml` without restarting

---

## Step 5 — Stop

```bash
bash scripts/stop.sh
```

---

## Troubleshooting

### "Model download is slow"
The model is ~5 GB. Ensure a stable internet connection.  
Set `HUGGINGFACE_HUB_CACHE` to a fast drive if needed.

### "CUDA out of memory"
Set `SPEECH_DTYPE=float16` in `.env` or try CPU:
```bash
SPEECH_DEVICE=cpu SPEECH_DTYPE=float32 bash scripts/start.sh
```

### "Ollama backend not reachable"
Check Ollama is running: `ollama serve` or `ollama ps`

### "Error: No such file or directory: audio.wav"
Place audio files in the `input/` directory. Supported formats: WAV, MP3, FLAC, OGG.

### `TorchCodec is required for load_with_torchcodec`

This error means `torchaudio≥2.9` is installed but `torchcodec` (its new FFmpeg-backed dependency) is not.

**Why it happens:** `torchaudio 2.9` (released Oct 2025) removed its bundled audio backends and made `torchaudio.load()` a thin alias to `load_with_torchcodec()`, which requires `torchcodec` + FFmpeg. The code detects this automatically and falls through to `soundfile`, but if `soundfile` is also missing you will see this error.

**Fix — pick ONE option:**

```bash
# Option A (recommended): install soundfile — no FFmpeg required
uv pip install soundfile

# Option B: install torchcodec + ensure FFmpeg 4-8 is in PATH
uv pip install torchcodec
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# Option C: pin torchaudio to a version that bundles its own I/O backends
uv pip install 'torchaudio>=2.0.0,<2.9'
```

> ⚠️ **A page refresh is NOT sufficient.** Python caches all imports at process startup. Newly installed packages are invisible to a running process — `sys.modules` is read once at startup and does not change when a Gradio page is refreshed. You must **fully restart the application** (`bash scripts/stop.sh && bash scripts/start.sh`) after installing a new package.

### "cert.pem blocked by IBM Vault Radar on git push"

IBM Vault Radar flags **any** X.509 certificate file (even public ones) as a medium-severity finding.
`certs/cert.pem` and `certs/key.pem` are excluded from git via `.gitignore` — they must never be committed.

If Vault Radar still blocks you, the cert was likely staged before `.gitignore` was updated.
Remove it from the index:

```bash
git rm --cached certs/cert.pem certs/key.pem 2>/dev/null || true
git commit -m "chore: remove TLS certs from git tracking"
```

Regenerate certs locally at any time with:

```bash
bash scripts/setup.sh   # idempotent — skips if certs already exist
```

---

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

All tests run without GPU or internet — heavy model calls are mocked.
