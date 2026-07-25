import io
import os
import queue
import re
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import cv2
import mss
import numpy
import pygetwindow
import requests
import sounddevice
import soundfile
import zxingcpp
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from openai import OpenAI

# Load environment variables from the .env file located in the project root
load_dotenv('.env')

# Load environment variables from the .env.local file located in the project root
load_dotenv('.env.local')

# TTS_ENGINE selects which voice backend to use: 'openai' (default), 'xtts'
# (a local/remote xtts-api-server instance, see https://github.com/daswer123/xtts-api-server)
# or 'elevenlabs'.
TTS_ENGINE = (os.getenv('TTS_ENGINE') or 'openai').strip().lower()

# Base URL of the xtts-api-server instance, only used when TTS_ENGINE=xtts.
XTTS_API_URL = (os.getenv('XTTS_API_URL') or 'http://localhost:8020').rstrip('/')

# Language passed to xtts-api-server, only used when TTS_ENGINE=xtts.
XTTS_LANGUAGE = os.getenv('XTTS_LANGUAGE') or 'ru'

# ElevenLabs API key, only used when TTS_ENGINE=elevenlabs.
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Model passed to ElevenLabs, only used when TTS_ENGINE=elevenlabs. Defaults to
# a model that supports an explicit language_code (see ELEVENLABS_LANGUAGE below) -
# most other ElevenLabs models auto-detect language from the text instead.
ELEVENLABS_MODEL = os.getenv('ELEVENLABS_MODEL') or 'eleven_flash_v2_5'

# Language enforced via language_code, only used when TTS_ENGINE=elevenlabs.
# Only honored by language_code-capable models (eleven_turbo_v2_5, eleven_flash_v2_5);
# ignored by others.
ELEVENLABS_LANGUAGE = os.getenv('ELEVENLABS_LANGUAGE') or 'ru'

# Comma-separated voice names/IDs to drop from the pool entirely, regardless of
# TTS_ENGINE - useful to blacklist a voice that turned out to sound wrong for
# its assigned gender, without waiting on the provider to fix its own labels.
EXCLUDED_VOICES = {name.strip() for name in (os.getenv('EXCLUDED_VOICES') or '').split(',') if name.strip()}

if TTS_ENGINE == 'openai' and not os.getenv("OPENAI_API_KEY"):
    print("[CRITICAL] OPENAI_API_KEY not found in .env file or environment variables!")

if TTS_ENGINE == 'elevenlabs' and not ELEVENLABS_API_KEY:
    print("[CRITICAL] ELEVENLABS_API_KEY not found in .env file or environment variables!")

# DEBUG=True in .env enables verbose scan/detection logging. Any other value
# (or missing key) keeps output to essential/critical messages only.
DEBUG = os.getenv("DEBUG") == 'True'


def debug_print(message):
    if DEBUG:
        print(message)


# Initialize OpenAI client (only needed when TTS_ENGINE=openai)
client = OpenAI() if TTS_ENGINE == 'openai' else None

# Initialize ElevenLabs client (only needed when TTS_ENGINE=elevenlabs)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if TTS_ENGINE == 'elevenlabs' else None

# QR/barcode detector: zxing-cpp (zxingcpp.read_barcodes) is primary.
# cv2.QRCodeDetector is kept as a fallback for when zxing finds nothing.
qr_detector_fallback = cv2.QRCodeDetector()

# Thread-safe queue to pass tasks from the screen scanner to the coordinator thread
voice_queue = queue.Queue()

# Default worker count differs per engine: OpenAI's API handles real parallel
# requests, but a local xtts-api-server instance serializes on one GPU/model,
# so extra workers there just queue up instead of speeding anything up.
# ElevenLabs caps concurrent requests by subscription tier (as low as 2-3 on
# free/starter plans), so it defaults conservatively; raise via TTS_WORKERS
# if your plan allows more.
if TTS_ENGINE == 'xtts':
    DEFAULT_TTS_WORKERS = 1
elif TTS_ENGINE == 'elevenlabs':
    DEFAULT_TTS_WORKERS = 3
else:
    DEFAULT_TTS_WORKERS = 15
TTS_WORKERS = int(os.getenv('TTS_WORKERS') or DEFAULT_TTS_WORKERS)

# Thread pool for parallel TTS requests to the configured TTS_ENGINE
executor = ThreadPoolExecutor(max_workers=TTS_WORKERS)

# Thread-safe cancellation event to instantly kill ongoing voiceover threads/playbacks
cancel_event = threading.Event()

# How many times to retry a single failed OpenAI TTS request before giving up
TTS_MAX_RETRIES = 5
TTS_RETRY_DELAY_SEC = 0.5

TTS_MODEL = os.getenv('OPENAI_TTS_MODEL') or 'gpt-4o-mini-tts'

# Sentences shorter than this (in characters) get merged with a neighboring
# sentence instead of being sent to the TTS engine as their own request -
# avoids wasting a whole request/playback cycle on something like "Да."
MIN_SENTENCE_CHARS = int(os.getenv('MIN_SENTENCE_CHARS', '20'))

GENDER_MALE = 'Male'
GENDER_FEMALE = 'Female'
GENDER_UNKNOWN = 'Unknown'

OPENAI_VOICES = [
    {'name': 'alloy', 'gender': GENDER_FEMALE},
    {'name': 'echo', 'gender': GENDER_MALE},
    {'name': 'fable', 'gender': GENDER_FEMALE},
    {'name': 'onyx', 'gender': GENDER_MALE},
    {'name': 'nova', 'gender': GENDER_FEMALE},
    {'name': 'shimmer', 'gender': GENDER_FEMALE},
]


def _classify_xtts_gender(speaker_name):
    # xtts-api-server doesn't expose a gender field, so it's inferred from the
    # speaker name itself (e.g. "Zoltan_Male", "Anna_Female").
    return GENDER_FEMALE if 'female' in speaker_name.lower() else GENDER_MALE


def _fetch_xtts_voices():
    try:
        response = requests.get(f"{XTTS_API_URL}/speakers_list", timeout=10)
        response.raise_for_status()
        speaker_names = response.json()
    except Exception as e:
        print(f"[CRITICAL] Could not fetch speaker list from xtts-api-server at {XTTS_API_URL}: {e}")
        return []

    return [{'name': name, 'gender': _classify_xtts_gender(name)} for name in speaker_names]


def _fetch_elevenlabs_voices():
    try:
        voices = elevenlabs_client.voices.get_all().voices
    except Exception as e:
        print(f"[CRITICAL] Could not fetch voice list from ElevenLabs: {e}")
        return []

    # Voices without an explicit labels.gender from ElevenLabs are skipped
    # rather than guessed - defaulting them to a gender risks silently mixing
    # an actually-female voice into the male pool (or vice versa).
    result = []
    for voice in voices:
        gender_label = (voice.labels or {}).get('gender')
        if gender_label == 'female':
            result.append({'name': voice.voice_id, 'gender': GENDER_FEMALE})
        elif gender_label == 'male':
            result.append({'name': voice.voice_id, 'gender': GENDER_MALE})

    return result


if TTS_ENGINE == 'xtts':
    VOICES = _fetch_xtts_voices()
elif TTS_ENGINE == 'elevenlabs':
    VOICES = _fetch_elevenlabs_voices()
else:
    VOICES = OPENAI_VOICES

VOICES = [voice for voice in VOICES if voice['name'] not in EXCLUDED_VOICES]


def _print_voices_by_gender(voices):
    male_names = [voice['name'] for voice in voices if voice['gender'] == GENDER_MALE]
    female_names = [voice['name'] for voice in voices if voice['gender'] == GENDER_FEMALE]
    print(f"[VOICES] Male ({len(male_names)}): {', '.join(male_names) if male_names else '-'}")
    print(f"[VOICES] Female ({len(female_names)}): {', '.join(female_names) if female_names else '-'}")


_print_voices_by_gender(VOICES)

VOICES_CACHE = {}


def _build_qr_variants(img_bgr):
    """
    Builds a few alternative renderings of the same frame to maximize the
    chance that OpenCV's QR detector actually finds a code. The native
    detector is quite sensitive to small size, low contrast and anti-aliasing
    artifacts, which is why some in-game QR codes were being missed.
    """
    variants = [("original", img_bgr)]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Plain grayscale often decodes better than the raw BGR capture
    variants.append(("grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    # Upscaled version helps with QR codes that render small on screen
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("upscaled_2x", cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)))

    # Adaptive threshold helps when the QR sits on a busy/gradient background
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    variants.append(("adaptive_threshold", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

    return variants


def _scan_with_cv2_fallback(img, seen):
    """
    Last-resort pass using cv2.QRCodeDetector, only invoked when zxing-cpp
    found nothing at all this tick. Kept around because the two detectors
    occasionally succeed on different frames/borderline cases.
    """
    debug_print("[DEBUG] zxing found nothing this tick, trying cv2.QRCodeDetector fallback...")

    fallback_texts = []

    for variant_name, variant_img in _build_qr_variants(img):
        try:
            retval, decoded_texts, points, straight_qrcode = qr_detector_fallback.detectAndDecodeMulti(variant_img)
        except Exception as e:
            print(f"[QR ERROR] cv2 fallback failed on variant '{variant_name}': {e}")
            continue

        if not retval:
            debug_print(f"[DEBUG] cv2 fallback variant '{variant_name}': no QR pattern detected.")
            continue

        non_empty = [t.strip() for t in decoded_texts if t.strip()]
        debug_print(f"[DEBUG] cv2 fallback variant '{variant_name}': detector fired, {len(non_empty)} readable QR(s).")

        for text in non_empty:
            if text not in seen:
                seen.add(text)
                fallback_texts.append(text)

    return fallback_texts


# A real ESO window should be at least this big. Filters out stray hidden
# helper/tooltip/notification windows that merely happen to match the title
# substring search (pygetwindow matches substrings, not exact titles).
MIN_WINDOW_WIDTH = 300
MIN_WINDOW_HEIGHT = 200


def _find_eso_window():
    """
    Returns the first window matching the ESO title that actually looks like
    the game's real window - on-screen, not minimized, and of a plausible
    size - or None if nothing qualifies this tick.
    """
    windows = pygetwindow.getWindowsWithTitle("Elder Scrolls Online")
    if not windows:
        return None

    for win in windows:
        try:
            if win.isMinimized:
                continue
            if not win.isActive:
                continue
            if win.left <= -10000 or win.top <= -10000:
                continue
            if win.width < MIN_WINDOW_WIDTH or win.height < MIN_WINDOW_HEIGHT:
                continue
        except Exception as e:
            print(f"[CAPTURE WARN] Could not inspect a candidate ESO window: {e}")
            continue

        return win

    return None


def read_multiple_eso_qrs(sct):
    """
    Finds the ESO window, captures its content safely, and scans for any
    visible QR codes. Tries several image variants to reduce missed reads,
    and tolerates a minimized/closed/moved window without crashing.
    """
    debug_print("[DEBUG] Scanning screen for the ESO window...")

    eso_win = _find_eso_window()
    if eso_win is None:
        debug_print(
            "[DEBUG] No suitable ESO window found this tick (not running, not focused, minimized, or too small/off-screen).")
        return []

    try:
        width = eso_win.width
        height = eso_win.height
        left = eso_win.left
        top = eso_win.top
    except Exception as e:
        # Window handle can go stale between the lookup and reading its geometry
        print(f"[CAPTURE WARN] Could not read ESO window geometry: {e}")
        return []

    monitor = {"top": top, "left": left, "width": width, "height": height}
    debug_print(f"[DEBUG] Capturing region: {monitor}")

    try:
        screenshot = sct.grab(monitor)
    except Exception as e:
        print(f"[CAPTURE ERROR] Screen grab failed: {e}")
        return []

    img = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_BGRA2BGR)

    found_texts = []
    seen = set()

    for variant_name, variant_img in _build_qr_variants(img):
        try:
            results = zxingcpp.read_barcodes(variant_img)
        except Exception as e:
            print(f"[QR ERROR] Detection failed on variant '{variant_name}': {e}")
            continue

        if not results:
            debug_print(f"[DEBUG] Variant '{variant_name}': no QR pattern detected.")
            continue

        non_empty = [r.text.strip() for r in results if r.text and r.text.strip()]
        debug_print(f"[DEBUG] Variant '{variant_name}': detector fired, {len(non_empty)} readable QR(s).")

        for text in non_empty:
            if text not in seen:
                seen.add(text)
                found_texts.append(text)

    if not found_texts:
        found_texts.extend(_scan_with_cv2_fallback(img, seen))

    debug_print(f"[DEBUG] Total unique raw QR payloads found this tick: {len(found_texts)}")

    return found_texts


def parse_qr_content(raw_text):
    """
    Parses the internal line structure of the QR data payload.
    """
    lines = [line.strip() for line in raw_text.split('\n')]
    if len(lines) < 3:
        return None

    event_header = lines[0]
    npc_name = lines[1]
    gender_id = lines[2]

    gender = GENDER_UNKNOWN
    if gender_id == "1":
        gender = GENDER_FEMALE
    elif gender_id == "2":
        gender = GENDER_MALE

    speech_text = " ".join(lines[3:])

    if '-' in event_header:
        event_type, event_id_str = event_header.split('-', 1)
        try:
            event_id = int(event_id_str)
            return {
                "type": event_type,
                "id": event_id,
                "npc_name": npc_name,
                "gender": gender,
                "text": speech_text
            }
        except ValueError:
            return None

    return None


ROUND_ROBIN_CURSOR = {GENDER_MALE: 0, GENDER_FEMALE: 0}


def select_voice(name, gender):
    gender = gender if gender != GENDER_UNKNOWN else GENDER_MALE

    if not VOICES_CACHE.get(name):
        filtered_names = [voice['name'] for voice in VOICES if voice['gender'] == gender]

        cursor = ROUND_ROBIN_CURSOR[gender]
        VOICES_CACHE[name] = filtered_names[cursor % len(filtered_names)]
        ROUND_ROBIN_CURSOR[gender] += 1

    return VOICES_CACHE[name]


def split_into_sentences(text):
    """
    Splits a raw text block into individual clean sentences using a regex pattern.
    """
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]


def merge_short_sentences(sentences, min_chars=MIN_SENTENCE_CHARS):
    """
    Greedily merges consecutive short sentences into a single chunk so tiny
    fragments (e.g. "Да.", "Хорошо.") don't each get their own TTS request
    and playback slot. Any leftover short tail gets folded into the last
    chunk rather than sent on its own.
    """
    if not sentences:
        return sentences

    merged = []
    buffer = ""

    for sentence in sentences:
        buffer = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(buffer) >= min_chars:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return merged


def _fetch_openai_audio_bytes(text, voice_name):
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice_name,
        input=text,
        response_format="wav",
    )
    return response.content


def _fetch_xtts_audio_bytes(text, voice_name):
    # Uses /tts_to_audio/ rather than /tts_stream: the streaming endpoint
    # sends a WAV header with an unknown/placeholder data size, which
    # soundfile/libsndfile reads as empty audio (silent playback) even
    # though a browser happily plays the same chunked response.
    #
    # A bare trailing "." makes the model vocalize it as a garbled sound,
    # so it's stripped here since punctuation isn't needed for pronunciation.
    clean_text = text.rstrip('.')

    response = requests.post(
        f"{XTTS_API_URL}/tts_to_audio/",
        json={"text": clean_text, "speaker_wav": f"{voice_name}.wav", "language": XTTS_LANGUAGE},
        timeout=60,
    )
    response.raise_for_status()
    return response.content


# Sample rate requested from ElevenLabs' raw pcm_<rate> output_format.
ELEVENLABS_SAMPLE_RATE = 24000


def _pcm16_to_wav_bytes(pcm_bytes, samplerate, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _fetch_elevenlabs_audio_bytes(text, voice_name):
    # ElevenLabs has no native WAV output_format, so raw 16-bit PCM is
    # requested and wrapped in a WAV header locally - soundfile/libsndfile
    # can't parse headerless PCM on its own. convert() returns the audio as
    # an iterator of byte chunks rather than a single blob.
    audio_chunks = elevenlabs_client.text_to_speech.convert(
        voice_id=voice_name,
        text=text,
        model_id=ELEVENLABS_MODEL,
        language_code=ELEVENLABS_LANGUAGE,
        output_format=f"pcm_{ELEVENLABS_SAMPLE_RATE}",
    )
    pcm_bytes = b"".join(audio_chunks)
    return _pcm16_to_wav_bytes(pcm_bytes, ELEVENLABS_SAMPLE_RATE)


def fetch_single_sentence_audio(text, voice_name, current_cancel_event):
    """
    Worker task dispatched to the ThreadPoolExecutor.
    Fetches the speech binary data for exactly one sentence from the
    configured TTS_ENGINE (openai or xtts). Requests an explicit WAV
    response, since the default mp3 response is not reliably decodable by
    soundfile/libsndfile on every machine, and retries a couple of times on
    transient network/API errors.
    """
    if current_cancel_event.is_set():
        return None

    last_error = None
    for attempt in range(1, TTS_MAX_RETRIES + 2):
        if current_cancel_event.is_set():
            return None

        try:
            if TTS_ENGINE == 'xtts':
                audio_bytes = _fetch_xtts_audio_bytes(text, voice_name)
            elif TTS_ENGINE == 'elevenlabs':
                audio_bytes = _fetch_elevenlabs_audio_bytes(text, voice_name)
            else:
                audio_bytes = _fetch_openai_audio_bytes(text, voice_name)

            if current_cancel_event.is_set():
                return None

            audio_buffer = io.BytesIO(audio_bytes)
            data, samplerate = soundfile.read(audio_buffer)
            return data, samplerate

        except Exception as e:
            last_error = e
            print(f"[FETCH ERROR] Attempt {attempt} failed for sentence: {e}")
            if attempt <= TTS_MAX_RETRIES and not current_cancel_event.is_set():
                time.sleep(TTS_RETRY_DELAY_SEC)

    print(f"[FETCH ERROR] Giving up on sentence after retries: {last_error}")
    return None


def process_voiceover_sequence(text, voice_name, current_cancel_event):
    """
    Coordinates sentence slicing, dispatches parallel futures to the thread pool,
    and handles ordered audio rendering with safe pacing pauses.
    """
    sentences = split_into_sentences(text)
    sentences = merge_short_sentences(sentences)
    if not sentences:
        return

    print(f"[COORDINATOR] Splitting speech block into {len(sentences)} individual sentences...")

    futures = []
    for sentence in sentences:
        future = executor.submit(fetch_single_sentence_audio, sentence, voice_name, current_cancel_event)
        futures.append(future)

    for idx, future in enumerate(futures):
        if current_cancel_event.is_set():
            break

        try:
            result = future.result()
            if result is None or current_cancel_event.is_set():
                continue

            audio_data, samplerate = result

            print(f" -> Speaking sentence {idx + 1}/{len(sentences)}: '{sentences[idx][:30]}...'")

            sounddevice.play(audio_data, samplerate)

            while sounddevice.get_stream().active:
                if current_cancel_event.is_set():
                    sounddevice.stop()
                    return
                time.sleep(0.05)

            if idx < len(futures) - 1 and not current_cancel_event.is_set():
                time.sleep(0.4)

        except Exception as e:
            print(f"[COORDINATOR ERROR] Sequencer failure on item index {idx}: {e}")


def voiceover_worker():
    """
    Persistent background worker thread managing sequential dialogue/subtitle events.
    """
    print("[THREAD] Background Voiceover Sequencer Worker running.")
    while True:
        task = voice_queue.get()
        if task is None:
            voice_queue.task_done()
            break

        text, voice_name, current_cancel_event = task
        process_voiceover_sequence(text, voice_name, current_cancel_event)
        voice_queue.task_done()


def main_loop():
    global cancel_event
    print("================================================================")
    print(" ESO QR Bard Active. Parallel Sentence Futures Mode Engaged.     ")
    print("================================================================")

    worker_thread = threading.Thread(target=voiceover_worker, daemon=True)
    worker_thread.start()

    last_dialogue_id = -1
    last_subtitle_id = -1

    # Reuse a single mss instance instead of recreating it every tick
    with mss.MSS() as sct:
        try:
            while True:
                try:
                    raw_qrs = read_multiple_eso_qrs(sct)
                    detected_events = {}

                    for raw_text in raw_qrs:
                        debug_print(f"[DEBUG] RAW text: {raw_qrs}")
                        parsed_data = parse_qr_content(raw_text)
                        if parsed_data:
                            debug_print(
                                f"[DEBUG] Parsed OK -> type={parsed_data['type']} id={parsed_data['id']} npc={parsed_data['npc_name']}")
                            detected_events[parsed_data["type"]] = parsed_data
                        else:
                            debug_print(f"[DEBUG] Failed to parse raw QR payload: {raw_text!r}")

                    target_event = None

                    if "s" in detected_events and detected_events["s"]["id"] != last_subtitle_id:
                        target_event = detected_events["s"]
                        last_subtitle_id = target_event["id"]

                    elif "d" in detected_events and detected_events["d"]["id"] != last_dialogue_id:
                        target_event = detected_events["d"]
                        last_dialogue_id = target_event["id"]

                    if target_event:
                        print(
                            f"[NEW EVENT CHOSEN] Type: {target_event['type'].upper()} | ID: {target_event['id']} | NPC: {target_event['npc_name']}")

                        cancel_event.set()
                        sounddevice.stop()

                        while not voice_queue.empty():
                            try:
                                voice_queue.get_nowait()
                                voice_queue.task_done()
                            except queue.Empty:
                                break

                        cancel_event = threading.Event()
                        chosen_voice = select_voice(target_event['npc_name'], target_event["gender"])
                        print(f"[VOICE] NPC '{target_event['npc_name']}' ({target_event['gender']}) -> {chosen_voice}")

                        voice_queue.put((target_event["text"], chosen_voice, cancel_event))

                except Exception as e:
                    print(f"[MAIN LOOP ERROR] Core clock tick failure: {e}")

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Stopping ESO QR Bard...")
            cancel_event.set()
            sounddevice.stop()
            voice_queue.put(None)
            executor.shutdown(wait=False, cancel_futures=True)
            worker_thread.join(timeout=2)
            print("[SHUTDOWN] Done.")


if __name__ == "__main__":
    main_loop()
