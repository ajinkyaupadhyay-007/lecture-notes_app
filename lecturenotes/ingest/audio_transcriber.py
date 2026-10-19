"""Audio transcription using faster-whisper (small model, CPU).

Produces a normalized list of AudioSegment objects, one per Whisper segment.
This module is the LAST milestone — it is intentionally isolated so the rest
of the pipeline can be built and tested without it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lecturenotes.ingest.models import AudioSegment

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[AudioSegment]:
    """Transcribe an audio or video file with faster-whisper.

    Args:
        audio_path: Path to the audio/video file (.mp3, .mp4, .wav, etc.).
        model_size: Whisper model variant. Defaults to ``"small"``.
        device: ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
        compute_type: Quantisation type. ``"int8"`` works on CPU.

    Returns:
        List of :class:`AudioSegment` objects in chronological order.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        ImportError: If faster-whisper is not installed.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.info(
        "Loading Whisper model '%s' on device='%s', compute_type='%s'",
        model_size,
        device,
        compute_type,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info("Transcribing: %s", path)
    whisper_segments, info = model.transcribe(str(path), beam_size=5)

    logger.info(
        "Detected language '%s' (probability %.2f)",
        info.language,
        info.language_probability,
    )

    segments: list[AudioSegment] = []
    for seg in whisper_segments:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            AudioSegment(text=text, start=round(seg.start, 3), end=round(seg.end, 3))
        )

    logger.info("Transcription complete: %d segments", len(segments))
    return segments
