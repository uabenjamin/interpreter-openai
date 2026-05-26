from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_tts_instructions(target_language_label: str) -> str:
    return (
        f"Speak in consistent {target_language_label} using the same speaker identity "
        "on every utterance. Keep the same timbre, persona, accent, pacing baseline, "
        "and overall delivery from clip to clip. Do not roleplay, do not change "
        "character, and do not vary age or personality. Use a calm, warm live "
        "interpretation style with clear diction and restrained emphasis."
    )


@dataclass(slots=True)
class AppConfig:
    command: str
    openai_project: str | None
    realtime_session_model: str
    source_language: str
    target_language_label: str
    input_audio_file: Path | None
    input_device: str | None
    output_device: str | None
    transcription_model: str
    turn_detection_type: str
    semantic_vad_eagerness: str
    max_turn_ms: int
    translation_buffer_silence_ms: int
    translation_buffer_max_ms: int
    translation_min_words: int
    translation_model: str
    translation_max_output_tokens: int
    enable_tts: bool
    tts_model: str
    tts_voice: str
    tts_instructions: str
    tts_speed: float
    capture_sample_rate_hz: int
    sample_rate_hz: int
    chunk_duration_ms: int
    speech_filter_mode: str
    speech_filter_highpass_hz: float
    speech_filter_lowpass_hz: float
    speech_filter_gate_threshold: float
    speech_filter_gate_floor: float
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
        description="Local macOS CLI English-to-target-language interpreter built with OpenAI APIs.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "doctor", "devices", "stop", "status"),
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--source-language",
        default=os.getenv("INTERPRETER_OPENAI_SOURCE_LANGUAGE", "en"),
        help="ISO-639-1 source language code for transcription.",
    )
    parser.add_argument(
        "--target-language",
        "--target-language-label",
        dest="target_language_label",
        default=os.getenv(
            "INTERPRETER_OPENAI_TARGET_LANGUAGE",
            os.getenv(
                "INTERPRETER_OPENAI_TARGET_LANGUAGE_LABEL",
                "Mandarin Chinese (Simplified Chinese script)",
            ),
        ),
        help=(
            "Target language used for translation and optional TTS, for example "
            "'Mandarin Chinese (Simplified Chinese script)' or 'Korean'."
        ),
    )
    parser.add_argument(
        "--input-device",
        default=os.getenv("INTERPRETER_OPENAI_INPUT_DEVICE"),
        help=(
            "Optional microphone device name or unique substring, for example "
            "'Maono'. Defaults to the system default input device."
        ),
    )
    parser.add_argument(
        "--input-audio-file",
        "--input-file",
        dest="input_audio_file",
        type=Path,
        default=_optional_path(os.getenv("INTERPRETER_OPENAI_INPUT_AUDIO_FILE")),
        help=(
            "Optional local WAV file to use instead of the microphone. "
            "Use this for repeatable tests with audio extracted from online videos."
        ),
    )
    parser.add_argument(
        "--output-device",
        default=os.getenv("INTERPRETER_OPENAI_OUTPUT_DEVICE"),
        help=(
            "Optional speaker device name or unique substring, for example "
            "'Maono'. Defaults to the system default output device."
        ),
    )
    parser.add_argument(
        "--transcription-model",
        default=os.getenv(
            "INTERPRETER_OPENAI_TRANSCRIPTION_MODEL",
            "gpt-realtime-whisper",
        ),
        help="Realtime transcription model.",
    )
    parser.add_argument(
        "--turn-detection-type",
        default=os.getenv("INTERPRETER_OPENAI_TURN_DETECTION_TYPE", "none"),
        choices=("none", "server_vad", "semantic_vad"),
        help=(
            "OpenAI Realtime turn detection mode. Use none for manual commits; "
            "gpt-realtime-whisper currently rejects server-side turn detection."
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
        help="Text model used for English-to-target-language translation.",
    )
    parser.add_argument(
        "--translation-max-output-tokens",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_TRANSLATION_MAX_OUTPUT_TOKENS", "320")),
        help="Maximum output tokens for the translation step.",
    )
    parser.add_argument(
        "--enable-tts",
        action="store_true",
        default=_env_flag("INTERPRETER_OPENAI_ENABLE_TTS", False),
        help="Enable translated text-to-speech playback. Disabled by default.",
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
        default=os.getenv("INTERPRETER_OPENAI_TTS_INSTRUCTIONS"),
        help=(
            "Speech style instructions for the TTS model. By default this is derived "
            "from the target language."
        ),
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
        help="Audio sample rate used for OpenAI input and optional TTS playback.",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=int(os.getenv("INTERPRETER_OPENAI_CHUNK_DURATION_MS", "20")),
        help="Mic capture chunk size in milliseconds.",
    )
    parser.add_argument(
        "--speech-filter-mode",
        default=os.getenv("INTERPRETER_OPENAI_SPEECH_FILTER_MODE", "off"),
        choices=("off", "voice_focus"),
        help=(
            "Optional local preprocessing before audio is sent to OpenAI. "
            "voice_focus applies a speech-band filter plus a light gate."
        ),
    )
    parser.add_argument(
        "--speech-filter-highpass-hz",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_SPEECH_FILTER_HIGHPASS_HZ", "120")),
        help="High-pass cutoff for the local voice_focus filter.",
    )
    parser.add_argument(
        "--speech-filter-lowpass-hz",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_SPEECH_FILTER_LOWPASS_HZ", "3600")),
        help="Low-pass cutoff for the local voice_focus filter.",
    )
    parser.add_argument(
        "--speech-filter-gate-threshold",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_SPEECH_FILTER_GATE_THRESHOLD", "0.015")),
        help="RMS threshold for the local voice_focus gate.",
    )
    parser.add_argument(
        "--speech-filter-gate-floor",
        type=float,
        default=float(os.getenv("INTERPRETER_OPENAI_SPEECH_FILTER_GATE_FLOOR", "0.15")),
        help="Residual gain floor when the local voice_focus gate attenuates audio.",
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
        help="Optional CSV glossary for interpretation terminology.",
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
    tts_instructions = args.tts_instructions or _default_tts_instructions(
        args.target_language_label
    )
    turn_detection_type = args.turn_detection_type
    if (
        args.transcription_model == "gpt-realtime-whisper"
        and turn_detection_type in {"server_vad", "semantic_vad"}
    ):
        turn_detection_type = "none"
    return AppConfig(
        command=args.command,
        openai_project=args.openai_project,
        realtime_session_model=args.realtime_session_model,
        source_language=args.source_language,
        target_language_label=args.target_language_label,
        input_audio_file=args.input_audio_file.expanduser() if args.input_audio_file else None,
        input_device=args.input_device,
        output_device=args.output_device,
        transcription_model=args.transcription_model,
        turn_detection_type=turn_detection_type,
        semantic_vad_eagerness=args.semantic_vad_eagerness,
        max_turn_ms=args.max_turn_ms,
        translation_buffer_silence_ms=args.translation_buffer_silence_ms,
        translation_buffer_max_ms=args.translation_buffer_max_ms,
        translation_min_words=args.translation_min_words,
        translation_model=args.translation_model,
        translation_max_output_tokens=args.translation_max_output_tokens,
        enable_tts=args.enable_tts,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_instructions=tts_instructions,
        tts_speed=args.tts_speed,
        capture_sample_rate_hz=args.capture_sample_rate_hz,
        sample_rate_hz=args.sample_rate_hz,
        chunk_duration_ms=args.chunk_duration_ms,
        speech_filter_mode=args.speech_filter_mode,
        speech_filter_highpass_hz=args.speech_filter_highpass_hz,
        speech_filter_lowpass_hz=args.speech_filter_lowpass_hz,
        speech_filter_gate_threshold=args.speech_filter_gate_threshold,
        speech_filter_gate_floor=args.speech_filter_gate_floor,
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
