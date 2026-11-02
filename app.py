"""
IBM Granite Speech 5.0-2B — End-to-End Demo Application (Gradio UI).

Tabs
────
1. Transcribe   – upload or record audio → Granite Speech → text output
2. Translate    – speech translation between supported language pairs
3. LLM Interact – send a transcript to the configured LLM backend
4. Config       – inspect and switch settings at runtime
5. Hardware     – display host hardware profile

Run
───
    python app.py
    # or via scripts/start.sh
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from src.config_manager import get_config, reload_config
from src.hardware_detector import detect_hardware
from src.llm_client import build_llm_client
from src.speech_processor import (
    GraniteSpeechProcessor,
    PerfConfig,
    SpeechResult,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
# The speech processor is loaded lazily to avoid blocking app startup.
_speech_processor: Optional[GraniteSpeechProcessor] = None
_hardware_profile = None


def _get_speech_processor() -> GraniteSpeechProcessor:
    """Return the global speech processor, building it from config if needed."""
    global _speech_processor
    if _speech_processor is not None:
        return _speech_processor

    cfg = get_config()
    hw = _get_hardware_profile()

    # Resolve device
    device_cfg = cfg.speech.device
    if device_cfg == "auto":
        device = hw.recommended_device
    else:
        device = device_cfg

    # Resolve dtype
    import torch
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype_cfg = cfg.speech.dtype
    if dtype_cfg == "auto":
        dtype = dtype_map.get(hw.recommended_dtype, torch.bfloat16)
    else:
        dtype = dtype_map.get(dtype_cfg, torch.bfloat16)

    # Map config perf section → PerfConfig dataclass
    p = cfg.speech.perf
    perf = PerfConfig(
        max_new_tokens_short=p.max_new_tokens_short,
        short_clip_threshold_s=p.short_clip_threshold_s,
        num_beams=p.num_beams,
        use_cache=p.use_cache,
        int8_quantize=p.int8_quantize,
        compile=p.compile,
        profile=p.profile,
    )
    logger.info(
        "Initialising speech processor: device=%s  dtype=%s  "
        "int8=%s  compile=%s  num_beams=%d  max_tokens_short=%d",
        device, dtype,
        perf.int8_quantize, perf.compile, perf.num_beams, perf.max_new_tokens_short,
    )
    _speech_processor = GraniteSpeechProcessor(
        device=device,
        torch_dtype=dtype,
        model_name=cfg.speech.model_name,
        cache_dir=cfg.speech.cache_dir or None,
        perf=perf,
    )
    return _speech_processor


def _get_hardware_profile():
    global _hardware_profile
    if _hardware_profile is None:
        _hardware_profile = detect_hardware()
    return _hardware_profile


# ── Output helpers ─────────────────────────────────────────────────────────────

def _save_output(content: dict, prefix: str = "result") -> Path:
    """Save a JSON result to the output directory with a timestamp."""
    cfg = get_config()
    out_dir = Path(cfg.app.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{prefix}_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(content, fh, indent=2, ensure_ascii=False)
    return out_path


def _result_meta_str(result: SpeechResult) -> str:
    # Use full timing report when profiling is enabled, otherwise brief summary.
    if result.timing is not None:
        return result.timing_report()
    return (
        f"⏱ Audio: {result.audio_duration_s:.1f}s  |  "
        f"Processing: {result.processing_time_s:.1f}s  |  "
        f"RTFx: {result.realtime_factor:.1f}×  |  "
        f"Device: {result.device_used}"
    )


# ── Tab 1: Transcribe ─────────────────────────────────────────────────────────

def transcribe_audio(
    audio_file,
    verbatim: bool,
) -> Tuple[str, str]:
    """
    Gradio handler: transcribe uploaded or recorded audio.

    Returns (transcript_text, metadata_string)
    """
    if audio_file is None:
        return "", "⚠ Please upload or record audio first."
    try:
        processor = _get_speech_processor()
        cfg = get_config()
        result = processor.transcribe(
            audio_file,
            verbatim=verbatim,
            max_new_tokens=cfg.speech.max_new_tokens,
        )
        _save_output(
            {"task": "transcribe", "transcript": result.text, "meta": _result_meta_str(result)},
            prefix="transcribe",
        )
        return result.text, _result_meta_str(result)
    except Exception as exc:
        logger.exception("Transcription failed")
        return "", f"❌ Error: {exc}"


# ── Tab 2: Translate ──────────────────────────────────────────────────────────

def translate_audio(
    audio_file,
    source_language: str,
    target_language: str,
) -> Tuple[str, str]:
    """Gradio handler: translate audio from one language to another."""
    if audio_file is None:
        return "", "⚠ Please upload audio first."
    if source_language == target_language:
        return "", "⚠ Source and target languages must differ."
    try:
        processor = _get_speech_processor()
        cfg = get_config()
        result = processor.translate(
            audio_file,
            source_language=source_language,
            target_language=target_language,
            max_new_tokens=cfg.speech.max_new_tokens,
        )
        _save_output(
            {
                "task": "translate",
                "source_language": source_language,
                "target_language": target_language,
                "translation": result.text,
                "meta": _result_meta_str(result),
            },
            prefix="translate",
        )
        return result.text, _result_meta_str(result)
    except Exception as exc:
        logger.exception("Translation failed")
        return "", f"❌ Error: {exc}"


# ── Model discovery helpers ───────────────────────────────────────────────────

def fetch_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """
    Query Ollama's GET /api/tags endpoint and return a list of model name strings.
    Returns an empty list if Ollama is unreachable or returns an error.
    """
    try:
        url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.debug("fetch_ollama_models failed: %s", exc)
        return []


def fetch_llamacpp_models(base_url: str = "http://localhost:9931/v1") -> list[str]:
    """
    Query llama.cpp server's GET /v1/models endpoint.
    Returns an empty list if the server is unreachable.
    """
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception as exc:
        logger.debug("fetch_llamacpp_models failed: %s", exc)
        return []


def refresh_model_list(backend: str, base_url_override: str = "") -> Tuple[list, str]:
    """
    Gradio handler: fetch available models for the selected backend.
    Returns (gr.update for model dropdown, status string).
    """
    cfg = get_config()
    backend_l = backend.lower() if backend and backend != "default" else cfg.llm.backend
    entry = getattr(cfg.llm, backend_l, cfg.llm.active_backend_entry())
    base_url = base_url_override.strip() or entry.base_url

    if backend_l == "ollama":
        models = fetch_ollama_models(base_url)
        if not models:
            return gr.update(), "⚠️ Ollama unreachable or no models found. Is `ollama serve` running?"
        current = entry.model
        default = current if current in models else models[0]
        return gr.update(choices=models, value=default), f"✅ Found {len(models)} model(s)"
    elif backend_l == "llamacpp":
        models = fetch_llamacpp_models(base_url)
        if not models:
            return gr.update(), "⚠️ llama.cpp server unreachable or no models found."
        return gr.update(choices=models, value=models[0]), f"✅ Found {len(models)} model(s)"
    else:
        # For OpenAI / Anthropic / LM Studio, just show the configured model
        return gr.update(choices=[entry.model], value=entry.model), f"ℹ️ Using configured model: {entry.model}"


# ── Tab 3: LLM Interact ───────────────────────────────────────────────────────

def llm_interact(
    transcript: str,
    user_question: str,
    backend_override: str,
    model_override: str = "",
) -> Tuple[str, str]:
    """
    Send transcript + question to the configured (or overridden) LLM backend.
    Returns (llm_reply, metadata_string).
    """
    if not transcript.strip():
        return "", "⚠ Provide a transcript first (use the Transcribe tab)."
    if not user_question.strip():
        user_question = "Please summarise the above transcript concisely."

    cfg = get_config()
    backend = backend_override if backend_override and backend_override != "default" else cfg.llm.backend
    entry = cfg.llm.active_backend_entry() if backend == cfg.llm.backend else getattr(cfg.llm, backend.lower(), cfg.llm.active_backend_entry())

    # Use the UI-selected model if provided, otherwise fall back to config
    model = model_override.strip() if model_override and model_override.strip() else entry.model

    try:
        client = build_llm_client(
            backend=backend,
            model=model,
            base_url=entry.base_url or None,
        )
        messages = [
            {
                "role": "user",
                "content": f"Transcript:\n{transcript}\n\nInstruction: {user_question}",
            }
        ]
        t0 = time.perf_counter()
        resp = client.chat(
            messages,
            system_prompt=cfg.llm.system_prompt,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        meta = (
            f"Backend: {resp.backend}  |  Model: {resp.model}  |  "
            f"Latency: {resp.latency_s:.2f}s  |  "
            f"Tokens: {resp.input_tokens}→{resp.output_tokens}"
        )
        _save_output(
            {
                "task": "llm_interact",
                "backend": resp.backend,
                "model": resp.model,
                "transcript": transcript,
                "question": user_question,
                "reply": resp.text,
            },
            prefix="llm",
        )
        return resp.text, meta
    except Exception as exc:
        logger.exception("LLM call failed")
        return "", f"❌ Error: {exc}"


# ── Tab 4: Pipeline (ASR → LLM in one shot) ───────────────────────────────────

def full_pipeline(
    audio_file,
    verbatim: bool,
    user_question: str,
    backend_override: str,
    model_override: str = "",
) -> Tuple[str, str, str]:
    """
    Run ASR then immediately pass the transcript to the LLM.
    Returns (transcript, llm_reply, combined_metadata).
    """
    transcript, asr_meta = transcribe_audio(audio_file, verbatim)
    # Guard: if ASR returned an error message (prefixed with ❌), skip the LLM step.
    if transcript.startswith("❌"):
        return transcript, "", asr_meta

    llm_reply, llm_meta = llm_interact(transcript, user_question, backend_override, model_override)
    combined_meta = f"ASR → {asr_meta}\nLLM → {llm_meta}"
    return transcript, llm_reply, combined_meta


# ── Tab 5: Config Inspector ────────────────────────────────────────────────────

def get_current_config() -> str:
    try:
        cfg = get_config()
        hw = _get_hardware_profile()
        info = {
            "speech": {
                "model": cfg.speech.model_name,
                "device": cfg.speech.device,
                "dtype": cfg.speech.dtype,
                "max_new_tokens": cfg.speech.max_new_tokens,
                "resolved_device": hw.recommended_device,
                "resolved_dtype": hw.recommended_dtype,
            },
            "llm": {
                "backend": cfg.llm.backend,
                "model": cfg.llm.active_backend_entry().model,
                "base_url": cfg.llm.active_backend_entry().base_url,
                "temperature": cfg.llm.temperature,
                "max_tokens": cfg.llm.max_tokens,
            },
            "app": {
                "port": cfg.app.server_port,
                "output_dir": cfg.app.output_dir,
            },
        }
        return json.dumps(info, indent=2)
    except Exception as exc:
        return f"Error reading config: {exc}"


def do_reload_config() -> str:
    try:
        global _speech_processor
        _speech_processor = None  # force re-init with new config
        reload_config()
        return "✅ Configuration reloaded. Speech processor will re-initialise on next use."
    except Exception as exc:
        return f"❌ Reload failed: {exc}"


# ── Tab 6: Hardware Inspector ─────────────────────────────────────────────────

def get_hardware_info() -> str:
    try:
        hw = _get_hardware_profile()
        return hw.summary_str()
    except Exception as exc:
        return f"Error: {exc}"


# ── Transcript download helper ────────────────────────────────────────────────

def _prepare_transcript_download(text: str) -> gr.update:
    """
    Write transcript text to a temp .txt file and return a gr.update that
    makes the DownloadButton visible and points it at that file.

    Called via transcript_out.change() — fires whenever the transcript
    Textbox content changes (i.e. after a successful transcription).

    Returns gr.update(visible=True, value=path) when text is non-empty,
    or gr.update(visible=False) to hide the button when there is nothing
    to download.
    """
    import tempfile

    if not text or text.startswith("❌"):
        return gr.update(visible=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Write to a NamedTemporaryFile that Gradio can serve for download.
    # delete=False so the file persists until Gradio cleans its cache.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=f"transcript_{ts}_",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(text)
    tmp.close()
    return gr.update(visible=True, value=tmp.name)


# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    cfg = get_config()
    lang_choices = list(SUPPORTED_LANGUAGES)
    backend_choices = ["default", "ollama", "lmstudio", "llamacpp", "openai", "anthropic"]

    with gr.Blocks(
        title="IBM Granite Speech 5.0-2B Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# 🎙️ IBM Granite Speech 5.0-2B — Demo Application
Multilingual ASR and Speech Translation powered by [Granite Speech 5.0-2B](https://huggingface.co/ibm-granite/granite-speech-5.0-2b),
with configurable LLM backends for downstream processing.

> **Model licence:** Apache 2.0.  No usage restrictions for enterprise or research use.
"""
        )

        # ── Tab 1: Transcribe ─────────────────────────────────────────────────
        with gr.Tab("📝 Transcribe"):
            gr.Markdown("Upload or record audio to transcribe using Granite Speech 5.0-2B.")
            with gr.Row():
                with gr.Column():
                    audio_input_1 = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio input (mono 16 kHz recommended; longer files split automatically)",
                        # Remove the download button from the input widget — downloading
                        # audio you just uploaded is redundant and triggers a Gradio 6.x
                        # bug that shows "true" as the save-dialog filename.
                        buttons=["share"],
                    )
                    verbatim_cb = gr.Checkbox(
                        label="Verbatim mode (no punctuation normalisation)",
                        value=False,
                    )
                    btn_transcribe = gr.Button("Transcribe", variant="primary")
                with gr.Column():
                    transcript_out = gr.Textbox(
                        label="Transcript", lines=10, interactive=True
                    )
                    meta_out_1 = gr.Textbox(label="Metadata", lines=2, interactive=False)
                    # Proper transcript download — downloads the text as a .txt file
                    btn_dl_transcript = gr.DownloadButton(
                        label="⬇ Download transcript",
                        visible=False,
                        variant="secondary",
                        size="sm",
                    )
            btn_transcribe.click(
                transcribe_audio,
                inputs=[audio_input_1, verbatim_cb],
                outputs=[transcript_out, meta_out_1],
            )
            transcript_out.change(
                _prepare_transcript_download,
                inputs=[transcript_out],
                outputs=[btn_dl_transcript],
            )

        # ── Tab 2: Translate ──────────────────────────────────────────────────
        with gr.Tab("🌍 Translate"):
            gr.Markdown(
                "Translate speech between supported languages.  "
                "Supported: **en ↔ fr/de/es/pt/ja**, en → it/zh."
            )
            with gr.Row():
                with gr.Column():
                    audio_input_2 = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio input",
                        buttons=["share"],  # no download — avoids "true" filename bug
                    )
                    src_lang = gr.Dropdown(
                        choices=lang_choices, value="en", label="Source language"
                    )
                    tgt_lang = gr.Dropdown(
                        choices=lang_choices + ["it", "zh"],
                        value="fr",
                        label="Target language",
                    )
                    btn_translate = gr.Button("Translate", variant="primary")
                with gr.Column():
                    translation_out = gr.Textbox(
                        label="Translation", lines=10, interactive=True
                    )
                    meta_out_2 = gr.Textbox(label="Metadata", lines=2, interactive=False)
            btn_translate.click(
                translate_audio,
                inputs=[audio_input_2, src_lang, tgt_lang],
                outputs=[translation_out, meta_out_2],
            )

        # ── Tab 3: LLM Interact ───────────────────────────────────────────────
        with gr.Tab("🤖 LLM Interact"):
            gr.Markdown(
                "Send a transcript to your configured LLM backend.  "
                "Click **🔄 Refresh** to discover models from Ollama / llama.cpp."
            )
            with gr.Row():
                with gr.Column():
                    transcript_in_3 = gr.Textbox(
                        label="Transcript", lines=8, placeholder="Paste transcript here…"
                    )
                    question_in = gr.Textbox(
                        label="Instruction / Question",
                        placeholder="Summarise this transcript, or ask a question about it…",
                        lines=3,
                    )
                    with gr.Row():
                        backend_dd = gr.Dropdown(
                            choices=backend_choices,
                            value="default",
                            label=f"Backend (default: {cfg.llm.backend})",
                        )
                        btn_refresh_models = gr.Button("🔄 Refresh", scale=0)
                    model_dd = gr.Dropdown(
                        choices=[cfg.llm.active_backend_entry().model],
                        value=cfg.llm.active_backend_entry().model,
                        allow_custom_value=True,
                        label="Model (editable — type any name or pick from list)",
                    )
                    refresh_status = gr.Textbox(
                        label="", lines=1, interactive=False, show_label=False
                    )
                    btn_llm = gr.Button("Send to LLM", variant="primary")
                with gr.Column():
                    llm_reply_out = gr.Textbox(label="LLM reply", lines=12, interactive=False)
                    meta_out_3 = gr.Textbox(label="Metadata", lines=2, interactive=False)
            btn_refresh_models.click(
                refresh_model_list,
                inputs=[backend_dd],
                outputs=[model_dd, refresh_status],
            )
            backend_dd.change(
                refresh_model_list,
                inputs=[backend_dd],
                outputs=[model_dd, refresh_status],
            )
            btn_llm.click(
                llm_interact,
                inputs=[transcript_in_3, question_in, backend_dd, model_dd],
                outputs=[llm_reply_out, meta_out_3],
            )

        # ── Tab 4: Full Pipeline ──────────────────────────────────────────────
        with gr.Tab("⚡ Full Pipeline"):
            gr.Markdown(
                "One-click pipeline: upload audio → ASR transcription → LLM processing."
            )
            with gr.Row():
                with gr.Column():
                    audio_input_4 = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio input",
                        buttons=["share"],  # no download — avoids "true" filename bug
                    )
                    verbatim_cb_4 = gr.Checkbox(label="Verbatim ASR mode", value=False)
                    question_in_4 = gr.Textbox(
                        label="LLM instruction",
                        placeholder="Summarise this transcript concisely.",
                        lines=3,
                    )
                    with gr.Row():
                        backend_dd_4 = gr.Dropdown(
                            choices=backend_choices,
                            value="default",
                            label="Backend override",
                        )
                        btn_refresh_models_4 = gr.Button("🔄 Refresh", scale=0)
                    model_dd_4 = gr.Dropdown(
                        choices=[cfg.llm.active_backend_entry().model],
                        value=cfg.llm.active_backend_entry().model,
                        allow_custom_value=True,
                        label="Model",
                    )
                    refresh_status_4 = gr.Textbox(
                        label="", lines=1, interactive=False, show_label=False
                    )
                    btn_pipeline = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    pipeline_transcript = gr.Textbox(
                        label="Transcript", lines=6, interactive=False
                    )
                    pipeline_reply = gr.Textbox(
                        label="LLM reply", lines=6, interactive=False
                    )
                    pipeline_meta = gr.Textbox(
                        label="Metadata", lines=3, interactive=False
                    )
            btn_refresh_models_4.click(
                refresh_model_list,
                inputs=[backend_dd_4],
                outputs=[model_dd_4, refresh_status_4],
            )
            backend_dd_4.change(
                refresh_model_list,
                inputs=[backend_dd_4],
                outputs=[model_dd_4, refresh_status_4],
            )
            btn_pipeline.click(
                full_pipeline,
                inputs=[audio_input_4, verbatim_cb_4, question_in_4, backend_dd_4, model_dd_4],
                outputs=[pipeline_transcript, pipeline_reply, pipeline_meta],
            )

        # ── Tab 5: Configuration ──────────────────────────────────────────────
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown(
                "Inspect the current runtime configuration.  "
                "Edit `config/config.yaml` or set environment variables, then click **Reload**."
            )
            with gr.Row():
                btn_show_config = gr.Button("Show current config")
                btn_reload_config = gr.Button("Reload config from disk")
            config_out = gr.Code(language="json", label="Configuration")
            reload_status = gr.Textbox(label="Status", lines=2, interactive=False)
            btn_show_config.click(get_current_config, outputs=config_out)
            btn_reload_config.click(do_reload_config, outputs=reload_status)

        # ── Tab 6: Hardware ───────────────────────────────────────────────────
        with gr.Tab("💻 Hardware"):
            gr.Markdown("Host hardware capability report generated at startup.")
            btn_hw = gr.Button("Refresh hardware info")
            hw_out = gr.Textbox(
                label="Hardware profile",
                lines=14,
                interactive=False,
                value=get_hardware_info(),
            )
            btn_hw.click(get_hardware_info, outputs=hw_out)

        gr.Markdown(
            """
---
**Model:** [ibm-granite/granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) · Apache 2.0  
**Source:** [granite-speech-models](https://github.com/ibm-granite/granite-speech-models) · [granite-speech](https://github.com/ibm-granite/granite-speech)
"""
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cfg = get_config()

    # Ensure output directory exists
    Path(cfg.app.output_dir).mkdir(parents=True, exist_ok=True)

    # Warm up hardware detection
    logger.info("Detecting hardware …")
    hw = _get_hardware_profile()
    logger.info("\n%s", hw.summary_str())

    demo = build_ui()

    # Build SSL kwargs — when cert/key are present the app serves HTTPS,
    # which is required for the browser microphone API to work on localhost.
    ssl_kwargs: dict = {}
    root = Path(__file__).parent
    certfile = cfg.app.ssl_certfile or str(root / "certs" / "cert.pem")
    keyfile = cfg.app.ssl_keyfile or str(root / "certs" / "key.pem")
    if Path(certfile).exists() and Path(keyfile).exists():
        ssl_kwargs = {
            "ssl_certfile": certfile,
            "ssl_keyfile": keyfile,
            # ssl_verify=False tells Gradio's internal httpx health-check to
            # skip cert validation for its own loopback call — required when
            # using a self-signed certificate, otherwise Gradio crashes on
            # startup with CERTIFICATE_VERIFY_FAILED before the browser even
            # connects.
            "ssl_verify": False,
        }
        scheme = "https"
        logger.info("TLS enabled — serving over HTTPS (microphone will work in browser)")
    else:
        scheme = "http"
        logger.warning(
            "No TLS cert found at %s — microphone requires HTTPS.\n"
            "  Run: openssl req -x509 -newkey rsa:2048 -nodes "
            "-keyout certs/key.pem -out certs/cert.pem -days 3650 "
            '-subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"',
            certfile,
        )

    logger.info("App URL: %s://localhost:%d", scheme, cfg.app.server_port)
    demo.launch(
        server_port=cfg.app.server_port,
        share=cfg.app.share,
        server_name="0.0.0.0",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
