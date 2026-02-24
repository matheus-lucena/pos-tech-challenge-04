"""
Handlers for the real-time audio transcription section of the Gradio interface.

Extracted from gradio_interface.py so that:
- Each function can be tested independently.
- The interface builder stays focused on layout/wiring.
"""

import os
import time
import threading
import wave
from datetime import datetime
from typing import Generator, Optional, Tuple

import gradio as gr

from config.constants import (
    AUDIO_PLAYER_POLL_INTERVAL,
    SAMPLE_RATE,
    TEMP_AUDIO_DIR,
    TRANSCRIPT_POLL_INTERVAL,
    WAV_CHANNELS,
    WAV_SAMPLEWIDTH,
)
from ui.realtime_processor import RealtimeAudioProcessor, _realtime_processor

# Module-level state for the audio player loop (avoids function-attribute hacks).
_last_temp_audio_file: Optional[str] = None


def get_device_index(device_str: Optional[str]) -> Optional[int]:
    """Parse device index from a dropdown string like '0: Microphone Name'."""
    if not device_str or ":" not in device_str:
        return None
    try:
        return int(device_str.split(":")[0])
    except Exception:
        return None


def start_realtime(device_selected: Optional[str]) -> tuple:
    """Start microphone streaming and return initial Gradio component states."""
    if _realtime_processor.is_processing:
        return (
            "⚠️ Já existe uma transcrição em andamento.",
            None,
            "Aguardando transcrição...",
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            "",
            True,
            None,
        )

    device_index = get_device_index(device_selected)

    def _stream():
        try:
            _realtime_processor.start_microphone_streaming(device_index=device_index)
        except Exception as e:
            print(f"Stream error: {e}")

    threading.Thread(target=_stream, daemon=True).start()

    status_msg = (
        '<div style="padding: 15px; background: #d4edda; border-radius: 8px; '
        'margin-bottom: 15px; border-left: 4px solid #28a745;">'
        '<p style="margin: 0; color: #155724;"><strong>🎙️ Gravando...</strong> '
        "Comece a falar! A transcrição aparecerá em tempo real.</p></div>"
    )
    return (
        status_msg,
        None,
        "Aguardando transcrição...",
        gr.update(visible=True),
        gr.update(visible=False),
        None,
        "",
        True,
        None,
    )


def stop_realtime() -> tuple:
    """Stop microphone streaming and return final Gradio component states."""
    status = _realtime_processor.stop_transcription()
    transcript = _realtime_processor.get_current_transcript()
    audio_path = _realtime_processor.get_recorded_audio_path()

    status_msg = (
        '<div style="padding: 15px; background: #fff3cd; border-radius: 8px; '
        'margin-bottom: 15px; border-left: 4px solid #ffc107;">'
        f'<p style="margin: 0; color: #856404;"><strong>⏹️ {status}</strong></p></div>'
    )

    alert_md = _build_violence_alert_md(_realtime_processor.get_violence_alert())

    return (
        status_msg,
        audio_path or None,
        transcript if transcript else "Nenhuma transcrição capturada.",
        gr.update(visible=False),
        gr.update(visible=True),
        audio_path or None,
        alert_md,
        False,
        audio_path,
    )


def update_transcript_loop() -> Generator[Tuple[str, str], None, None]:
    """Generator that yields (transcript, alert_md) while streaming is active."""
    yield (
        _realtime_processor.get_current_transcript() or "Aguardando transcrição...",
        _build_violence_alert_md(_realtime_processor.get_violence_alert()),
    )
    while _realtime_processor.is_processing:
        time.sleep(TRANSCRIPT_POLL_INTERVAL)
        yield (
            _realtime_processor.get_current_transcript() or "Aguardando transcrição...",
            _build_violence_alert_md(_realtime_processor.get_violence_alert()),
        )
    final = _realtime_processor.get_current_transcript()
    yield (
        final if final else "Transcrição finalizada.",
        _build_violence_alert_md(_realtime_processor.get_violence_alert()),
    )


def update_audio_player_loop() -> Generator[str, None, None]:
    """Generator that periodically yields a WAV preview path while recording."""
    global _last_temp_audio_file

    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

    while _realtime_processor.is_processing:
        if _realtime_processor.recorded_audio_frames:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = os.path.join(TEMP_AUDIO_DIR, f"realtime_{timestamp}.wav")

                with wave.open(temp_file, "wb") as wf:
                    wf.setnchannels(WAV_CHANNELS)
                    wf.setsampwidth(WAV_SAMPLEWIDTH)
                    wf.setframerate(SAMPLE_RATE)
                    for frame in _realtime_processor.recorded_audio_frames:
                        wf.writeframes(frame)

                if _last_temp_audio_file and os.path.exists(_last_temp_audio_file):
                    try:
                        os.remove(_last_temp_audio_file)
                    except Exception:
                        pass
                _last_temp_audio_file = temp_file
                yield temp_file

            except Exception as e:
                print(f"Error creating temporary audio: {e}")
                yield gr.update()
        else:
            yield gr.update()

        time.sleep(AUDIO_PLAYER_POLL_INTERVAL)

    final_audio_path = _realtime_processor.get_recorded_audio_path()
    if final_audio_path:
        if _last_temp_audio_file and os.path.exists(_last_temp_audio_file):
            try:
                os.remove(_last_temp_audio_file)
            except Exception:
                pass
        yield final_audio_path
    else:
        yield gr.update()


def _build_violence_alert_md(alert: Optional[str]) -> str:
    """Return an HTML Markdown string for a violence alert, or empty string if none."""
    if not alert:
        return ""
    return (
        '<div style="padding: 12px; background: #f8d7da; border-radius: 8px; '
        'border-left: 4px solid #dc3545; margin-top: 8px;">'
        f"<strong>🚨 Alerta de violência detectado:</strong> {alert}</div>"
    )
