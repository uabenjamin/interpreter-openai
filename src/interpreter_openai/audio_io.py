from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import soundcard as sc


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioDevices:
    default_microphone: str
    default_speaker: str


class AudioUnavailableError(RuntimeError):
    """Raised when no usable default audio device is available."""


def _default_microphone() -> object:
    microphones = sc.all_microphones()
    if not microphones:
        raise AudioUnavailableError(
            "No default microphone is visible. Check macOS microphone permissions "
            "for your terminal app and confirm a default input device exists."
        )
    return sc.default_microphone()


def _default_speaker() -> object:
    speakers = sc.all_speakers()
    if not speakers:
        raise AudioUnavailableError(
            "No default speaker is visible. Confirm macOS has an active default "
            "output device."
        )
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
    microphone = _default_microphone()
    speaker = _default_speaker()
    return AudioDevices(
        default_microphone=_describe_device(microphone, "Default microphone"),
        default_speaker=_describe_device(speaker, "Default speaker"),
    )


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
        capture_sample_rate_hz: int,
        output_sample_rate_hz: int,
        capture_chunk_frames: int,
    ) -> None:
        self._capture_sample_rate_hz = capture_sample_rate_hz
        self._output_sample_rate_hz = output_sample_rate_hz
        self._capture_chunk_frames = capture_chunk_frames
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._started_event = asyncio.Event()
        self._stop_event = threading.Event()
        self._task: asyncio.Task[None] | None = None
        self._overflow_warning_emitted = False

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=128)
        self._stop_event.clear()
        self._overflow_warning_emitted = False
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
        microphone = _default_microphone()
        try:
            LOGGER.info(
                "Microphone capture started at %s Hz, resampling to %s Hz.",
                self._capture_sample_rate_hz,
                self._output_sample_rate_hz,
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
                    chunk = float_audio_to_pcm16_bytes(frames)
                    self._loop.call_soon_threadsafe(self._queue_chunk, chunk)
        finally:
            self._loop.call_soon_threadsafe(self._queue_chunk, None)

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


class SpeakerPlayback:
    def __init__(
        self,
        sample_rate_hz: int,
        drain_ms: int,
        target_rms: float,
        max_gain: float,
    ) -> None:
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
        speaker = _default_speaker()
        normalized = self._normalize_for_playback(audio)
        speaker.play(normalized, samplerate=self._sample_rate_hz)
        if self._drain_ms > 0:
            time.sleep(self._drain_ms / 1000)

    def play_pcm16_stream_blocking(self, audio_chunks) -> None:
        speaker = _default_speaker()
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
