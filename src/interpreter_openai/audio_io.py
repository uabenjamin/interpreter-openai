from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundcard as sc


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioDevices:
    default_microphone: str
    default_speaker: str


@dataclass(slots=True)
class VisibleAudioDevices:
    microphones: list[str]
    speakers: list[str]


class AudioUnavailableError(RuntimeError):
    """Raised when no usable default audio device is available."""


@dataclass(slots=True)
class SpeechFilterConfig:
    mode: str
    sample_rate_hz: int
    highpass_hz: float
    lowpass_hz: float
    gate_threshold: float
    gate_floor: float


class SpeechAudioPreprocessor:
    def __init__(self, config: SpeechFilterConfig) -> None:
        self._config = config
        self._enabled = config.mode == "voice_focus"
        self._highpass_prev_x = 0.0
        self._highpass_prev_y = 0.0
        self._lowpass_prev_y = 0.0
        self._smoothed_rms = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process(self, data: np.ndarray) -> np.ndarray:
        if not self._enabled or len(data) == 0:
            return data
        if data.ndim == 2 and data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        samples = data[:, 0].astype(np.float32, copy=True)
        band_limited = self._band_limit(samples)
        gated = self._apply_gate(band_limited)
        return gated.reshape(-1, 1)

    def _band_limit(self, samples: np.ndarray) -> np.ndarray:
        sample_rate = float(self._config.sample_rate_hz)
        dt = 1.0 / sample_rate

        hp_cutoff = max(20.0, min(self._config.highpass_hz, sample_rate / 3.0))
        hp_rc = 1.0 / (2.0 * np.pi * hp_cutoff)
        hp_alpha = hp_rc / (hp_rc + dt)

        highpassed = np.empty_like(samples)
        prev_x = self._highpass_prev_x
        prev_y = self._highpass_prev_y
        for index, x_value in enumerate(samples):
            y_value = hp_alpha * (prev_y + float(x_value) - prev_x)
            highpassed[index] = y_value
            prev_x = float(x_value)
            prev_y = y_value
        self._highpass_prev_x = prev_x
        self._highpass_prev_y = prev_y

        lp_cutoff = max(hp_cutoff + 50.0, min(self._config.lowpass_hz, sample_rate / 2.2))
        lp_rc = 1.0 / (2.0 * np.pi * lp_cutoff)
        lp_alpha = dt / (lp_rc + dt)

        lowpassed = np.empty_like(highpassed)
        prev_y = self._lowpass_prev_y
        for index, x_value in enumerate(highpassed):
            prev_y = prev_y + lp_alpha * (float(x_value) - prev_y)
            lowpassed[index] = prev_y
        self._lowpass_prev_y = prev_y
        return lowpassed

    def _apply_gate(self, samples: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        self._smoothed_rms = (self._smoothed_rms * 0.85) + (rms * 0.15)
        threshold = max(self._config.gate_threshold, 1e-5)
        floor = float(np.clip(self._config.gate_floor, 0.0, 1.0))
        if self._smoothed_rms >= threshold:
            gain = 1.0
        else:
            ratio = max(0.0, min(self._smoothed_rms / threshold, 1.0))
            gain = floor + ((1.0 - floor) * ratio)
        return samples * gain


def _all_microphones() -> list[object]:
    microphones = list(sc.all_microphones())
    if not microphones:
        raise AudioUnavailableError(
            "No microphone is visible. Check macOS microphone permissions for your "
            "terminal app and confirm an input device exists."
        )
    return microphones


def _all_speakers() -> list[object]:
    speakers = list(sc.all_speakers())
    if not speakers:
        raise AudioUnavailableError(
            "No speaker is visible. Confirm macOS has an active output device."
        )
    return speakers


def _default_microphone() -> object:
    _all_microphones()
    return sc.default_microphone()


def _default_speaker() -> object:
    _all_speakers()
    return sc.default_speaker()


def _describe_device(device: object, fallback: str) -> str:
    with contextlib.suppress(Exception):
        name = getattr(device, "name")
        if name:
            return str(name)
    with contextlib.suppress(Exception):
        device_id = getattr(device, "id")
        if device_id:
            return f"{fallback} #{device_id}"
    return fallback


def get_default_devices() -> AudioDevices:
    return AudioDevices(
        default_microphone=get_default_microphone_name(),
        default_speaker=get_default_speaker_name(),
    )


def get_default_microphone_name() -> str:
    microphone = _default_microphone()
    return _describe_device(microphone, "Default microphone")


def get_default_speaker_name() -> str:
    speaker = _default_speaker()
    return _describe_device(speaker, "Default speaker")


def _device_names(devices: list[object], kind: str) -> list[str]:
    return [
        _describe_device(device, f"{kind} #{index + 1}")
        for index, device in enumerate(devices)
    ]


def list_microphone_names() -> list[str]:
    return _device_names(_all_microphones(), "Microphone")


def list_speaker_names() -> list[str]:
    return _device_names(_all_speakers(), "Speaker")


def get_visible_devices() -> VisibleAudioDevices:
    return VisibleAudioDevices(
        microphones=list_microphone_names(),
        speakers=list_speaker_names(),
    )


def _resolve_device(
    requested_name: str | None,
    *,
    kind: str,
    devices: list[object],
    default_device_getter,
) -> object:
    if not requested_name or not requested_name.strip():
        return default_device_getter()

    selector = requested_name.strip().casefold()
    by_name: dict[str, list[object]] = {}
    for device in devices:
        device_name = _describe_device(device, kind).strip()
        by_name.setdefault(device_name.casefold(), []).append(device)

    exact_matches = by_name.get(selector, [])
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        names = ", ".join(
            _describe_device(match, kind) for match in exact_matches
        )
        raise AudioUnavailableError(
            f"Multiple {kind.lower()} devices match '{requested_name}': {names}. "
            "Use a more specific device name."
        )

    substring_matches = [
        device
        for device in devices
        if selector in _describe_device(device, kind).casefold()
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        names = ", ".join(
            _describe_device(match, kind) for match in substring_matches
        )
        raise AudioUnavailableError(
            f"Multiple {kind.lower()} devices match '{requested_name}': {names}. "
            "Use a more specific device name."
        )

    available = ", ".join(_device_names(devices, kind))
    raise AudioUnavailableError(
        f"No {kind.lower()} device matches '{requested_name}'. Available "
        f"{kind.lower()} devices: {available}"
    )


def _selected_microphone(requested_name: str | None) -> object:
    return _resolve_device(
        requested_name,
        kind="Microphone",
        devices=_all_microphones(),
        default_device_getter=_default_microphone,
    )


def _selected_speaker(requested_name: str | None) -> object:
    return _resolve_device(
        requested_name,
        kind="Speaker",
        devices=_all_speakers(),
        default_device_getter=_default_speaker,
    )


def get_selected_microphone_name(requested_name: str | None) -> str:
    return _describe_device(_selected_microphone(requested_name), "Microphone")


def get_selected_speaker_name(requested_name: str | None) -> str:
    return _describe_device(_selected_speaker(requested_name), "Speaker")


def float_audio_to_pcm16_bytes(data: np.ndarray) -> bytes:
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def pcm16_bytes_to_float_audio(audio_bytes: bytes) -> np.ndarray:
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    normalized = samples / 32768.0
    return normalized.reshape(-1, 1)


def resample_float_audio(
    data: np.ndarray,
    input_sample_rate_hz: int,
    output_sample_rate_hz: int,
) -> np.ndarray:
    if input_sample_rate_hz == output_sample_rate_hz:
        return data
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if len(data) == 0:
        return data

    source = data[:, 0].astype(np.float32, copy=False)
    source_positions = np.linspace(0.0, 1.0, num=len(source), endpoint=False)
    target_length = max(
        1,
        int(round(len(source) * output_sample_rate_hz / input_sample_rate_hz)),
    )
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, source).astype(np.float32)
    return resampled.reshape(-1, 1)


class MicrophoneCapture:
    def __init__(
        self,
        device_name: str | None,
        capture_sample_rate_hz: int,
        output_sample_rate_hz: int,
        capture_chunk_frames: int,
        speech_filter_config: SpeechFilterConfig | None = None,
    ) -> None:
        self._device_name = device_name
        self._capture_sample_rate_hz = capture_sample_rate_hz
        self._output_sample_rate_hz = output_sample_rate_hz
        self._capture_chunk_frames = capture_chunk_frames
        self._speech_preprocessor = (
            SpeechAudioPreprocessor(speech_filter_config)
            if speech_filter_config is not None
            else None
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._started_event = asyncio.Event()
        self._stop_event = threading.Event()
        self._task: asyncio.Task[None] | None = None
        self._overflow_warning_emitted = False
        self._silence_warning_emitted = False
        self._capture_started_at: float | None = None
        self._peak_rms_since_start = 0.0

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=128)
        self._stop_event.clear()
        self._overflow_warning_emitted = False
        self._silence_warning_emitted = False
        self._capture_started_at = None
        self._peak_rms_since_start = 0.0
        self._started_event.set()
        self._task = asyncio.create_task(asyncio.to_thread(self._capture_loop))

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            with contextlib.suppress(Exception):
                await self._task
        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)

    async def chunks(self) -> AsyncIterator[bytes]:
        await self._started_event.wait()
        if self._queue is None:
            raise RuntimeError("MicrophoneCapture.start() must be called first.")
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk

    def _capture_loop(self) -> None:
        if self._loop is None or self._queue is None:
            raise RuntimeError("Capture loop started before initialization.")
        microphone = _selected_microphone(self._device_name)
        try:
            LOGGER.info(
                "Microphone capture started from %s at %s Hz, resampling to %s Hz.",
                _describe_device(microphone, "Microphone"),
                self._capture_sample_rate_hz,
                self._output_sample_rate_hz,
            )
            if self._speech_preprocessor is not None and self._speech_preprocessor.enabled:
                LOGGER.info(
                    "Applying local voice_focus filter: highpass=%sHz lowpass=%sHz gate=%.4f floor=%.2f",
                    self._speech_preprocessor._config.highpass_hz,
                    self._speech_preprocessor._config.lowpass_hz,
                    self._speech_preprocessor._config.gate_threshold,
                    self._speech_preprocessor._config.gate_floor,
                )
            with microphone.recorder(
                samplerate=self._capture_sample_rate_hz,
                channels=1,
                blocksize=self._capture_chunk_frames,
            ) as recorder:
                while not self._stop_event.is_set():
                    frames = recorder.record(numframes=self._capture_chunk_frames)
                    frames = resample_float_audio(
                        frames,
                        input_sample_rate_hz=self._capture_sample_rate_hz,
                        output_sample_rate_hz=self._output_sample_rate_hz,
                    )
                    self._track_input_level(frames, microphone)
                    if self._speech_preprocessor is not None:
                        frames = self._speech_preprocessor.process(frames)
                    chunk = float_audio_to_pcm16_bytes(frames)
                    self._loop.call_soon_threadsafe(self._queue_chunk, chunk)
        finally:
            self._loop.call_soon_threadsafe(self._queue_chunk, None)

    def _track_input_level(self, frames: np.ndarray, microphone: object) -> None:
        now = time.monotonic()
        if self._capture_started_at is None:
            self._capture_started_at = now

        if frames.size:
            samples = frames[:, 0] if frames.ndim == 2 else frames
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
            self._peak_rms_since_start = max(self._peak_rms_since_start, rms)

        if self._silence_warning_emitted:
            return
        if now - self._capture_started_at < 8.0:
            return
        if self._peak_rms_since_start >= 0.002:
            return

        LOGGER.warning(
            "Input from %s is nearly silent. Check the mixer send, Maono gain, "
            "macOS input selection, and whether the pastor mic is routed to this USB input.",
            _describe_device(microphone, "Microphone"),
        )
        self._silence_warning_emitted = True

    def _queue_chunk(self, chunk: bytes | None) -> None:
        if self._queue is None:
            return
        if chunk is None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(None)
            return
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            if not self._overflow_warning_emitted:
                LOGGER.warning(
                    "Dropping microphone audio because the realtime transcription "
                    "stream is not draining fast enough."
                )
                self._overflow_warning_emitted = True
        self._queue.put_nowait(chunk)


class AudioFileCapture:
    def __init__(
        self,
        path: Path,
        output_sample_rate_hz: int,
        output_chunk_frames: int,
        speech_filter_config: SpeechFilterConfig | None = None,
        realtime: bool = True,
    ) -> None:
        self._path = path.expanduser()
        self._output_sample_rate_hz = output_sample_rate_hz
        self._output_chunk_frames = output_chunk_frames
        self._speech_preprocessor = (
            SpeechAudioPreprocessor(speech_filter_config)
            if speech_filter_config is not None
            else None
        )
        self._realtime = realtime
        self._stop = False
        self._started_event = asyncio.Event()

    async def start(self) -> None:
        if not self._path.exists():
            raise AudioUnavailableError(f"Input audio file not found: {self._path}")
        if not self._path.is_file():
            raise AudioUnavailableError(f"Input audio path is not a file: {self._path}")
        self._stop = False
        self._started_event.set()

    async def stop(self) -> None:
        self._stop = True

    async def chunks(self) -> AsyncIterator[bytes]:
        await self._started_event.wait()
        try:
            with wave.open(str(self._path), "rb") as audio_file:
                channels = audio_file.getnchannels()
                sample_width = audio_file.getsampwidth()
                input_sample_rate_hz = audio_file.getframerate()
                if sample_width != 2:
                    raise AudioUnavailableError(
                        "Only 16-bit PCM WAV files are supported for --input-audio-file. "
                        "Convert the file to WAV with 16-bit PCM first."
                    )

                input_chunk_frames = max(
                    1,
                    int(
                        round(
                            self._output_chunk_frames
                            * input_sample_rate_hz
                            / self._output_sample_rate_hz
                        )
                    ),
                )
                LOGGER.info(
                    "Reading audio file %s at %s Hz, %s channel(s), streaming as %s Hz PCM.",
                    self._path,
                    input_sample_rate_hz,
                    channels,
                    self._output_sample_rate_hz,
                )

                while not self._stop:
                    raw = audio_file.readframes(input_chunk_frames)
                    if not raw:
                        break

                    data = self._pcm16_bytes_to_float_audio(raw, channels)
                    data = resample_float_audio(
                        data,
                        input_sample_rate_hz=input_sample_rate_hz,
                        output_sample_rate_hz=self._output_sample_rate_hz,
                    )
                    if self._speech_preprocessor is not None:
                        data = self._speech_preprocessor.process(data)
                    yield float_audio_to_pcm16_bytes(data)

                    if self._realtime:
                        await asyncio.sleep(
                            max(0.0, len(data) / self._output_sample_rate_hz)
                        )
        except wave.Error as exc:
            raise AudioUnavailableError(
                f"Could not read WAV input audio file {self._path}: {exc}"
            ) from exc

    def _pcm16_bytes_to_float_audio(self, audio_bytes: bytes, channels: int) -> np.ndarray:
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            complete_samples = len(samples) - (len(samples) % channels)
            samples = samples[:complete_samples].reshape(-1, channels).mean(axis=1)
        return samples.reshape(-1, 1)


class SpeakerPlayback:
    def __init__(
        self,
        device_name: str | None,
        sample_rate_hz: int,
        drain_ms: int,
        target_rms: float,
        max_gain: float,
    ) -> None:
        self._device_name = device_name
        self._sample_rate_hz = sample_rate_hz
        self._drain_ms = drain_ms
        self._target_rms = target_rms
        self._max_gain = max_gain
        self._play_lock = asyncio.Lock()

    async def play_pcm16(self, audio_bytes: bytes) -> None:
        async with self._play_lock:
            await asyncio.to_thread(self._play_pcm16_blocking, audio_bytes)

    def _play_pcm16_blocking(self, audio_bytes: bytes) -> None:
        audio = pcm16_bytes_to_float_audio(audio_bytes)
        speaker = _selected_speaker(self._device_name)
        normalized = self._normalize_for_playback(audio)
        speaker.play(normalized, samplerate=self._sample_rate_hz)
        if self._drain_ms > 0:
            time.sleep(self._drain_ms / 1000)

    def play_pcm16_stream_blocking(self, audio_chunks) -> None:
        speaker = _selected_speaker(self._device_name)
        block_frames = max(1, self._sample_rate_hz // 10)
        previous_block: np.ndarray | None = None

        try:
            with speaker.player(
                samplerate=self._sample_rate_hz,
                channels=1,
            ) as player:
                for block in self._iter_stream_blocks(audio_chunks, block_frames):
                    if previous_block is not None:
                        player.play(previous_block, wait=False)
                    previous_block = block

                if previous_block is not None:
                    player.play(previous_block, wait=True)
        except Exception as exc:
            LOGGER.warning(
                "Streaming playback is unavailable on this device. Falling back to "
                "buffered playback: %s",
                exc,
            )
            buffered = b"".join(audio_chunks)
            if buffered:
                self._play_pcm16_blocking(buffered)
                return

        if self._drain_ms > 0:
            time.sleep(self._drain_ms / 1000)

    def _iter_stream_blocks(self, audio_chunks, block_frames: int):
        pending = b""
        block_bytes = block_frames * 2

        for chunk in audio_chunks:
            if not chunk:
                continue
            pending += chunk
            complete_bytes = len(pending) - (len(pending) % 2)
            while complete_bytes >= block_bytes:
                block = pending[:block_bytes]
                pending = pending[block_bytes:]
                complete_bytes -= block_bytes
                yield self._prepare_stream_block(block)

        complete_bytes = len(pending) - (len(pending) % 2)
        if complete_bytes > 0:
            yield self._prepare_stream_block(pending[:complete_bytes])

    def _prepare_stream_block(self, audio_bytes: bytes) -> np.ndarray:
        audio = pcm16_bytes_to_float_audio(audio_bytes)
        return np.clip(audio, -0.97, 0.97)

    def _normalize_for_playback(self, audio: np.ndarray) -> np.ndarray:
        peak_limit = 0.97
        activity_threshold = 0.008
        target_percentile = 0.58
        normalized = np.clip(audio, -1.0, 1.0)
        analysis = np.abs(normalized[:, 0])
        active_mask = analysis >= activity_threshold
        active = normalized[active_mask] if np.any(active_mask) else normalized
        if active.size == 0:
            return normalized

        active_samples = active[:, 0]
        rms = float(np.sqrt(np.mean(np.square(active_samples), dtype=np.float64)))
        if rms <= 1e-5:
            return normalized

        p95 = float(np.percentile(np.abs(active_samples), 95))
        rms_gain = self._target_rms / rms
        percentile_gain = target_percentile / max(p95, 1e-4)
        gain = min(np.sqrt(rms_gain * percentile_gain), self._max_gain)
        peak = float(np.max(np.abs(normalized)))
        if peak > 1e-5:
            gain = min(gain, peak_limit / peak)

        adjusted = normalized * gain
        post_peak = float(np.max(np.abs(adjusted)))
        if post_peak > peak_limit:
            adjusted = np.tanh(adjusted / peak_limit) * peak_limit
        return np.clip(adjusted, -peak_limit, peak_limit)
