# Interpreter OpenAI MVP

Local macOS CLI that listens to English speech on a configurable macOS input
device, transcribes it with the OpenAI Realtime API, translates the transcript
into a configurable target language with a text model, and can optionally play
translated speech through a configurable output device with OpenAI
text-to-speech.

This version keeps listening continuously while translated speech is queued
for playback when TTS is enabled. The pipeline is:

`microphone -> OpenAI Realtime transcription -> OpenAI text translation -> optional OpenAI TTS -> optional speaker`

## Why this architecture

This is intentionally not a single speech-to-speech prompt loop. For
interpreting, it is useful to keep these stages separate:

- realtime transcription stays low-latency and stable
- translation stays promptable, so sermon-specific terminology can be injected
- speech output can be tuned independently

That makes it easier to improve translation accuracy for church use without
rewriting the audio plumbing.

## Local setup

### 1. Save your OpenAI API key

Add your key to your shell profile, for example `~/.zshrc`:

```bash
export OPENAI_API_KEY="sk-..."
```

Optional, if your account uses a specific project:

```bash
export OPENAI_PROJECT="proj_..."
```

Then reload your shell:

```bash
source ~/.zshrc
```

Do not save API keys in this repository.

### 2. macOS permissions

Grant microphone access to your terminal app in System Settings:

`Privacy & Security > Microphone`

### 3. Install dependencies

```bash
.venv/bin/pip install -e .
```

### 4. Verify the environment

```bash
.venv/bin/python -m interpreter_openai doctor
```

`doctor` checks:

- selected microphone visibility
- selected speaker visibility when `--enable-tts` is set
- OpenAI API key presence
- a small text-model translation probe
- an optional TTS probe when `--enable-tts` is set

To see the exact device names that the app can open through `soundcard`, run:

```bash
.venv/bin/python -m interpreter_openai devices
```

The current CoreAudio default device is marked with `*`.

### 5. Run the interpreter

```bash
.venv/bin/python -m interpreter_openai run
```

This default run mode does not play translated audio. It only prints:

- finalized English sent to translation
- finalized translated output in the target language

If you want speech playback, enable it explicitly:

```bash
.venv/bin/python -m interpreter_openai run --enable-tts
```

You can also launch the desktop GUI:

```bash
.venv/bin/python -m interpreter_openai gui
```

The GUI lets you choose the input source, starts with a Maono device selected
when one is visible, otherwise prefers the built-in MacBook microphone, and
then falls back to the system default input. Click `Start` to begin
transcription and translation. The same button becomes `Stop` while the
interpreter is running. Closing the window or clicking `Quit` asks for
confirmation if transcription is still active.

The GUI also supports a temporary sermon reference before you click `Start`.
Use `Paste`, `Upload`, or drag and drop a file onto the reference box. Supported
reference formats are `.txt`, `.md`, `.markdown`, `.rtf`, `.docx`, and `.pdf`.
The reference can be summarized once into a compact translation context, or used
as a raw excerpt. Summarized mode is recommended for live use because it keeps
translation requests smaller and avoids adding latency to every sentence.

Sermon references are used for translation context only. They help with sermon
topic, Scripture references, church names, announcements, key terms, and obvious
ASR correction hints. The current realtime transcription model does not support
prompting with a sermon draft. References are held in memory and forgotten when
the app quits.

By default, the target language is Mandarin Chinese. You can change it, for
example to Korean:

```bash
.venv/bin/python -m interpreter_openai run --target-language Korean
```

If your mixer or interface is not being picked up correctly through the macOS
default device, pin the device explicitly by name or a unique substring:

```bash
.venv/bin/python -m interpreter_openai run --input-device Maono
```

For repeatable testing from a recorded service or online video, first extract
the audio to a local 16-bit PCM WAV file, then use `--input-audio-file`:

```bash
.venv/bin/python -m interpreter_openai run \
  --input-audio-file ~/Downloads/service-test.wav \
  --glossary-file resources/sermon_glossary.sample.csv \
  --translation-notes-file resources/sermon_translation_notes.sample.md
```

The app currently reads local WAV files directly. It does not download online
videos itself. If you have permission to use the video, use a downloader such
as `yt-dlp` and convert/extract with `ffmpeg`, for example:

```bash
yt-dlp -x --audio-format wav -o ~/Downloads/service-test.%(ext)s "VIDEO_URL"
ffmpeg -i ~/Downloads/service-test.wav -ac 1 -ar 24000 -sample_fmt s16 ~/Downloads/service-test-24k.wav
```

Then pass `~/Downloads/service-test-24k.wav` to `--input-audio-file`.

If you want to play an online video in the browser and have the app listen to
that playback live, route macOS system audio into a virtual input device. On
macOS, a common free option is BlackHole 2ch; a paid option with a simpler app
UI is Rogue Amoeba Loopback.

Recommended BlackHole setup:

1. Install BlackHole 2ch.
2. Open `Audio MIDI Setup`.
3. Create a `Multi-Output Device` that includes your headphones/speakers and
   `BlackHole 2ch`.
4. Set macOS sound output to that `Multi-Output Device`.
5. Play the online video.
6. Run the interpreter with BlackHole as the input:

```bash
.venv/bin/python -m interpreter_openai run \
  --input-device "BlackHole 2ch" \
  --glossary-file resources/sermon_glossary.sample.csv \
  --translation-notes-file resources/sermon_translation_notes.sample.md
```

Use `.venv/bin/python -m interpreter_openai devices` to confirm the exact
virtual device name. This path captures whatever macOS is playing, so it is
good for quick testing but less controlled than `--input-audio-file`.

If you also want translated speech to go back out through the interface:

```bash
.venv/bin/python -m interpreter_openai run \
  --input-device Maono \
  --output-device Maono \
  --enable-tts
```

If the incoming feed contains some background bed or room rumble, you can also
enable OpenAI far-field noise reduction plus a local speech-focused prefilter
before audio is sent to OpenAI:

```bash
.venv/bin/python -m interpreter_openai run \
  --input-device Maono \
  --noise-reduction-mode far_field \
  --speech-filter-mode voice_focus \
  --max-turn-ms 3000
```

This is a limited mitigation, not true source separation. If speech and music
are already mixed together on the same bus, no simple filter can fully remove
the music. The best live setup is a dedicated mixer send that contains the
pastor mic and excludes the music channels. If possible, send a post-EQ direct
out or aux mix with only the pastor's microphone into the Maono USB input.

To stop it from the terminal, use `Control-C`.

On macOS terminals, `Command-C` usually copies text and does not interrupt the
running process.

If you lose track of a running instance, use:

```bash
.venv/bin/python -m interpreter_openai status
.venv/bin/python -m interpreter_openai stop
```

## Current defaults

- Realtime transcription model: `gpt-realtime-whisper`
- Target language: `Mandarin Chinese (Simplified Chinese script)`
- Input device: system default unless `--input-device` is set
- Output device: system default unless `--output-device` is set
- Turn detection: `none` for `gpt-realtime-whisper`
- Manual commit interval: `6000 ms`
- Translation buffer silence: `900 ms`
- Translation buffer max age: `9000 ms`
- Translation minimum words: `4`
- Translation model: `gpt-4o`
- Translation max output tokens: `320`
- TTS enabled: `false`
- TTS model: `gpt-4o-mini-tts`
- TTS voice: `marin`
- TTS speed: `1.15`
- OpenAI audio format: `24 kHz` mono PCM
- Local microphone capture: `16 kHz`, resampled to `24 kHz`
- Local speech filter: `off`

These are configurable with CLI flags.

The live `gpt-realtime-whisper` endpoint currently rejects server-side turn
detection, so the app omits `turn_detection` for that model and manually
commits the current audio buffer every `6000 ms`. If an old environment
variable still requests `server_vad` or `semantic_vad` with
`gpt-realtime-whisper`, the app coerces it to `none` before opening the session.

To reduce translation delay, lower the manual commit interval with
`--max-turn-ms`. Shorter values produce faster but more fragmented transcripts.

Finalized English fragments are now buffered before translation. Instead of
translating every committed fragment immediately, the app waits for sentence
punctuation, a short idle gap, or a max buffer age, which produces more
coherent translated output when the speaker talks in long flowing clauses.
The buffer also holds obvious incomplete clauses a little longer, such as
segments ending with "for", "to", "if you are interested", or "both for", so
announcements are less likely to be translated mid-sentence.

For another realtime transcription model that supports server-side turn
detection, you can run:

```bash
.venv/bin/python -m interpreter_openai run \
  --turn-detection-type server_vad \
  --vad-threshold 0.35 \
  --vad-prefix-padding-ms 500 \
  --vad-silence-ms 800
```

If you want even faster translation handoff for live preaching, lower the cap:

```bash
.venv/bin/python -m interpreter_openai run --max-turn-ms 4000
```

If translation still feels too fragmentary, raise the buffer settings:

```bash
.venv/bin/python -m interpreter_openai run \
  --translation-buffer-silence-ms 1200 \
  --translation-buffer-max-ms 12000
```

This project uses OpenAI's GA realtime transcription session shape. The app
sends a `session.update` event with `session.type = "transcription"`, 24 kHz
mono PCM input, and the streaming transcription model `gpt-realtime-whisper`.

OpenAI removed the older Realtime beta interface on May 12, 2026, so older
beta-shaped session payloads are no longer valid.

Note: `gpt-4o-transcribe` is still useful for file/request-response
transcription workflows, but OpenAI's current realtime transcription guide
uses `gpt-realtime-whisper` for live streaming deltas.

When TTS is enabled, playback starts speaking as TTS audio bytes arrive instead
of waiting for the full clip to download first. This lowers perceived latency
even if total synthesis time is still around one second.

The app already pins a single named TTS voice on every request. To reduce
occasional drift in delivery, the default TTS instructions also tell the model
to keep the same speaker identity, timbre, persona, and pacing from clip to
clip.

The Realtime session also supports OpenAI-side input noise reduction through
`input_audio_noise_reduction`, and this project passes that setting through as
`near_field`, `far_field`, or `none`. That helps with general noise, but it is
not a vocal-isolation feature.

## Sermon-specific translation quality

The translation step uses a dedicated prompt for church interpretation and can
inject a glossary file directly into the model instructions. It also sends a
short rolling sermon context with each translation request so split Bible
quotes, pronouns, and repeated theological terms are translated more
consistently. Only the current segment is translated.

The translation prompt also tells the model that the English text is live ASR
and may contain obvious homophones or misheard church/Bible terms. It should
correct only clear context errors before translating, while avoiding invented
details.

The sample glossary is:

`resources/sermon_glossary.sample.csv`

Use it like this:

```bash
.venv/bin/python -m interpreter_openai run \
  --glossary-file resources/sermon_glossary.sample.csv \
  --translation-notes-file resources/sermon_translation_notes.sample.md
```

The sample format is:

```csv
source,target,notes
Gospel,福音,Prefer the Christian term.
grace,恩典,Do not translate as elegance or favor.
```

For best Sunday worship quality, copy the sample glossary and notes, then tune
them with your church's preferred Bible translation vocabulary, pastor names,
ministry names, sermon series terms, and recurring Scripture passages.

You can also run with only a free-form style notes file:

```bash
.venv/bin/python -m interpreter_openai run \
  --translation-notes-file resources/sermon_translation_notes.sample.md
```

## Important notes

- OpenAI's TTS voices are optimized for English, but the official docs say the
  model generally follows Whisper language coverage and supports Chinese.
- If you use built-in speakers instead of headphones, some playback may still
  bleed back into the microphone.
- This scaffold is meant to get the local OpenAI version running first. The
  next step after live validation is tuning the sermon glossary and translation
  instructions on real worship audio.

## References

This MVP was designed against the official OpenAI docs:

- Realtime transcription:
  https://developers.openai.com/api/docs/guides/realtime-transcription
- Realtime WebSocket connections:
  https://developers.openai.com/api/docs/guides/realtime-websocket
- Using realtime models:
  https://developers.openai.com/api/docs/guides/realtime-models-prompting
- Text to speech:
  https://platform.openai.com/docs/guides/text-to-speech
- Responses API:
  https://platform.openai.com/docs/api-reference/responses
