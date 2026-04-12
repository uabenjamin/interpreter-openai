# Interpreter OpenAI MVP

Local macOS CLI that listens to English speech on the system default
microphone, transcribes it with the OpenAI Realtime API, translates the
transcript to Mandarin with a text model, and plays Mandarin speech through the
system default speaker with OpenAI text-to-speech.

This version keeps listening continuously while translated Mandarin is queued
for playback. The pipeline is:

`microphone -> OpenAI Realtime transcription -> OpenAI text translation -> OpenAI TTS -> speaker`

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

- default microphone and speaker visibility
- OpenAI API key presence
- a small text-model translation probe
- a small TTS probe

### 5. Run the interpreter

```bash
.venv/bin/python -m interpreter_openai run
```

To stop it from the terminal, use `Control-C`.

On macOS terminals, `Command-C` usually copies text and does not interrupt the
running process.

If you lose track of a running instance, use:

```bash
.venv/bin/python -m interpreter_openai status
.venv/bin/python -m interpreter_openai stop
```

## Current defaults

- Realtime transcription model: `gpt-4o-transcribe`
- Realtime session model: `gpt-realtime`
- Turn detection: `semantic_vad`
- Semantic VAD eagerness: `low`
- Max turn duration: `6000 ms`
- Translation buffer silence: `900 ms`
- Translation buffer max age: `9000 ms`
- Translation minimum words: `4`
- Translation model: `gpt-4o`
- Translation max output tokens: `192`
- TTS model: `gpt-4o-mini-tts`
- TTS voice: `marin`
- TTS speed: `1.15`
- OpenAI audio format: `24 kHz` mono PCM
- Local microphone capture: `16 kHz`, resampled to `24 kHz`

These are configurable with CLI flags.

For fast speakers, the app now defaults to `semantic_vad` with low eagerness.
OpenAI's current Realtime docs describe `semantic_vad` as less likely to chunk
the transcript before the speaker is done, while `server_vad` chunks purely on
periods of silence.

To prevent a long sermon segment from sitting open for too long, the app also
force-commits the current audio buffer after `6000 ms` of continuous speech by
default. This bounds translation delay even when VAD is being conservative.

Finalized English fragments are now buffered before translation. Instead of
translating every committed fragment immediately, the app waits for sentence
punctuation, a short idle gap, or a max buffer age, which produces more
coherent Mandarin output when the speaker talks in long flowing clauses.

If you want the older silence-based behavior, run:

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

Important: the Realtime WebSocket session model and the transcription model are
not the same thing in this app. The WebSocket connects with a realtime model
such as `gpt-realtime`, and the session then enables input transcription with
`gpt-4o-transcribe`.

Note: OpenAI's docs currently list `gpt-4o-transcribe-latest` as a supported
Realtime transcription model, but this project defaults to `gpt-4o-transcribe`
because some live endpoints currently reject the `-latest` alias.

The playback path now starts speaking as TTS audio bytes arrive instead of
waiting for the full clip to download first. This lowers perceived latency even
if total synthesis time is still around one second.

The app already pins a single named TTS voice on every request. To reduce
occasional drift in delivery, the default TTS instructions also tell the model
to keep the same speaker identity, timbre, persona, and pacing from clip to
clip.

## Sermon-specific translation quality

The translation step uses a dedicated prompt for church interpretation and can
inject a glossary file directly into the model instructions.

The sample glossary is:

`resources/sermon_glossary.sample.csv`

Use it like this:

```bash
.venv/bin/python -m interpreter_openai run \
  --glossary-file resources/sermon_glossary.sample.csv
```

The sample format is:

```csv
source,target,notes
Gospel,福音,Prefer the Christian term.
grace,恩典,Do not translate as elegance or favor.
```

You can also provide an optional free-form style notes file:

```bash
.venv/bin/python -m interpreter_openai run \
  --glossary-file resources/sermon_glossary.sample.csv \
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
