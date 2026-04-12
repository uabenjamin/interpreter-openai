from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


@dataclass(slots=True)
class AppConfig:
    command: str
    openai_project: str | None
    realtime_session_model: str
    source_language: str
    target_language_label: str
    transcription_model: str
    turn_detection_type: str
    semantic_vad_eagerness: str
    max_turn_ms: int
    translation_buffer_silence_ms: int
    translation_buffer_max_ms: int
    translation_min_words: int
    translation_model: str
    translation_max_output_tokens: int
    tts_model: str
    tts_voice: str
    tts_instructions: str
    tts_speed: float
    capture_sample_rate_hz: int
    sample_rate_hz: int
    chunk_duration_ms: int
    vad_threshold: float
    vad_prefix_padding_ms: int
    vad_silence_ms: int
    noise_reduction_mode: str | None
    glossary_file: Path | None
    translation_notes_file: Path | None
    playback_drain_ms: int
    playback_target_rms: float
    playback_max_gain: float

    @property
    def chunk_frames(self) -> int:
        return int(self.sample_rate_hz * (self.chunk_duration_ms / 1000))

    @property
    def capture_chunk_frames(self) -> int:
        return int(self.capture_sample_rate_hz * (self.chunk_duration_ms / 1000))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interpreter-openai",
        description="Local macOS CLI English-to-Mandarin interpreter built with OpenAI APIs.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "doctor", "stop", "status"),
        help="Run the interpreter loop, verify local setup, or control a running instance.",
    )
    parser.add_argument(
        "--openai-project",
        default=os.getenv("OPENAI_PROJECT"),
        help="Optional OpenAI project ID.",
    )
    parser.add_argument(
        "--realtime-session-model",
        default=os.getenv(
            "INTERPRETER_OPENAI_REALTIME_SESSION_MODEL",
            "gpt-realtime",
        ),
        help="OpenAI Realtime session model used for the WebSocket connection.",
    )
    parser.add_argument(
        "--source-language",
        default=os.getenv("INTERPRETER_OPENAI_SOURCE_LANGUAGE", "en"),
        help="ISO-639-1 source language code for transcription.",
    )
    parser.add_argument(
        "--target-language-label",
        default=os.getenv(
            "INTERPRETER_OPENAI_TARGET_LANGUAGE_LABEL",
            "Mandarin Chinese (Simplified Chinese script)",
        ),
        help="Target language description used in translation prompting.",
    )
    parser.add_argument(
        "--transcription-model",
        default=os.getenv(
            "INTERPRETER_OPENAI_TRANSCRIPTION_MODEL",
            "gpt-4o-transcribe",
        ),
        help="Realtime transcription model.",
    )
    parser.add_argument(
        "--turn-detection-type",
        default=os.getenv("INTERPRETER_OPENAI_TURN_DETECTION_TYPE", "semantic_vad"),
        choices=("server_vad", "semantic_vad"),
        help=(
            "OpenAI Realtime turn detection mode. semantic_vad is usually better "
            "for fast, continuous speakers."
        ),
    )
    parser.add_argument(
        "--semantic-vad-eagerness",
        default=os.getenv("INTERPRETER_OPENAI_SEMANTIC_VAD_EAGERNESS", "low"),
        choices=("low", "medium", "high", "auto"),
        help=(
            "How quickly semantic_vad closes turns. low keeps larger chunks and is "
            "less likely to cut off fast speakers."
        ),
    )
    parser.add_argument(
        "--max-turn-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_MAX_TURN_MS", "6000")),
        help=(
            "Force-commit a long speech turn after this many milliseconds to bound "
            "translation delay. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--translation-buffer-silence-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_TRANSLATION_BUFFER_SILENCE_MS", "900")),
        help=(
            "Flush buffered finalized transcript fragments after this much idle time."
        ),
    )
    parser.add_argument(
        "--translation-buffer-max-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_TRANSLATION_BUFFER_MAX_MS", "9000")),
        help=(
            "Flush buffered finalized transcript fragments after this much total age "
            "even without punctuation."
        ),
    )
    parser.add_argument(
        "--translation-min-words",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_TRANSLATION_MIN_WORDS", "4")),
        help=(
            "Do not translate very short fragments unless punctuation or max buffer "
            "age forces a flush."
        ),
    )
    parser.add_argument(
        "--translation-model",
        default=os.getenv("INTERPRETER_OPENAI_TRANSLATION_MODEL", "gpt-4o"),
        help="Text model used for English-to-Mandarin translation.",
    )
    parser.add_argument(
        "--translation-max-output-tokens",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_TRANSLATION_MAX_OUTPUT_TOKENS", "192")),
        help="Maximum output tokens for the translation step.",
    )
    parser.add_argument(
        "--tts-model",
        default=os.getenv("INTERPRETER_OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        help="OpenAI text-to-speech model.",
    )
    parser.add_argument(
        "--tts-voice",
        default=os.getenv("INTERPRETER_OPENAI_TTS_VOICE", "marin"),
        help="OpenAI text-to-speech voice.",
    )
    parser.add_argument(
        "--tts-instructions",
        default=os.getenv(
            "INTERPRETER_OPENAI_TTS_INSTRUCTIONS",
            (
                "Speak in consistent Mandarin using the same speaker identity on every "
                "utterance. Keep the same timbre, persona, accent, pacing baseline, and "
                "overall delivery from clip to clip. Do not roleplay, do not change "
                "character, and do not vary age or personality. Use a calm, warm church "
                "interpretation style with clear diction and restrained emphasis."
            ),
        ),
        help="Speech style instructions for the TTS model.",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_TTS_SPEED", "1.15")),
        help="Speech rate for OpenAI TTS. Higher values reduce latency and spoken duration.",
    )
    parser.add_argument(
        "--capture-sample-rate-hz",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_CAPTURE_SAMPLE_RATE_HZ", "16000")),
        help="Local microphone capture sample rate in Hz. Audio is resampled to the OpenAI session rate.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_SAMPLE_RATE_HZ", "24000")),
        help="Audio sample rate used for OpenAI input and TTS playback.",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_CHUNK_DURATION_MS", "20")),
        help="Mic capture chunk size in milliseconds.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_VAD_THRESHOLD", "0.5")),
        help="OpenAI server VAD threshold.",
    )
    parser.add_argument(
        "--vad-prefix-padding-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_VAD_PREFIX_PADDING_MS", "300")),
        help="Server VAD prefix padding in milliseconds.",
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_VAD_SILENCE_MS", "450")),
        help="Server VAD silence duration in milliseconds.",
    )
    parser.add_argument(
        "--noise-reduction-mode",
        default=os.getenv("INTERPRETER_OPENAI_NOISE_REDUCTION_MODE", "near_field"),
        choices=("near_field", "far_field", "none"),
        help="OpenAI Realtime noise reduction mode.",
    )
    parser.add_argument(
        "--glossary-file",
        type=Path,
        default=_optional_path(os.getenv("INTERPRETER_OPENAI_GLOSSARY_FILE")),
        help="Optional CSV glossary for sermon terminology.",
    )
    parser.add_argument(
        "--translation-notes-file",
        type=Path,
        default=_optional_path(os.getenv("INTERPRETER_OPENAI_TRANSLATION_NOTES_FILE")),
        help="Optional text or markdown file with translation style notes.",
    )
    parser.add_argument(
        "--playback-drain-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_PLAYBACK_DRAIN_MS", "150")),
        help="Extra time to wait after playback to avoid truncation.",
    )
    parser.add_argument(
        "--playback-target-rms",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_PLAYBACK_TARGET_RMS", "0.14")),
        help="Target RMS loudness for translated playback normalization.",
    )
    parser.add_argument(
        "--playback-max-gain",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_PLAYBACK_MAX_GAIN", "1.8")),
        help="Maximum gain multiplier for translated playback normalization.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    noise_reduction_mode = None if args.noise_reduction_mode == "none" else args.noise_reduction_mode
    return AppConfig(
        command=args.command,
        openai_project=args.openai_project,
        realtime_session_model=args.realtime_session_model,
        source_language=args.source_language,
        target_language_label=args.target_language_label,
        transcription_model=args.transcription_model,
        turn_detection_type=args.turn_detection_type,
        semantic_vad_eagerness=args.semantic_vad_eagerness,
        max_turn_ms=args.max_turn_ms,
        translation_buffer_silence_ms=args.translation_buffer_silence_ms,
        translation_buffer_max_ms=args.translation_buffer_max_ms,
        translation_min_words=args.translation_min_words,
        translation_model=args.translation_model,
        translation_max_output_tokens=args.translation_max_output_tokens,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_instructions=args.tts_instructions,
        tts_speed=args.tts_speed,
        capture_sample_rate_hz=args.capture_sample_rate_hz,
        sample_rate_hz=args.sample_rate_hz,
        chunk_duration_ms=args.chunk_duration_ms,
        vad_threshold=args.vad_threshold,
        vad_prefix_padding_ms=args.vad_prefix_padding_ms,
        vad_silence_ms=args.vad_silence_ms,
        noise_reduction_mode=noise_reduction_mode,
        glossary_file=args.glossary_file,
        translation_notes_file=args.translation_notes_file,
        playback_drain_ms=args.playback_drain_ms,
        playback_target_rms=args.playback_target_rms,
        playback_max_gain=args.playback_max_gain,
    )
