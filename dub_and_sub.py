import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from openai import OpenAI


# -----------------------------
# Utilities
# -----------------------------
def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it's on PATH.")

def srt_timestamp(seconds: float) -> str:
    # SRT: HH:MM:SS,mmm
    ms = int(round(seconds * 1000))
    hh = ms // 3_600_000
    ms %= 3_600_000
    mm = ms // 60_000
    ms %= 60_000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:max_len] if s else "segment"

def write_srt(segments: List[Dict[str, Any]], out_path: Path, text_key: str = "text") -> None:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = srt_timestamp(float(seg["start"]))
        end = srt_timestamp(float(seg["end"]))
        text = str(seg.get(text_key, "")).strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def extract_audio(video_path: Path, wav_path: Path) -> None:
    # 16kHz mono WAV is a clean transcription format
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(wav_path)
    ])

def burn_subtitles(video_path: Path, srt_path: Path, out_path: Path) -> None:
    # Hard-burn subtitles into video
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles={str(srt_path)}",
        "-c:a", "copy",
        str(out_path)
    ])

def mux_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    # Replace audio track with dubbed audio (keep video stream)
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_path)
    ])

def attach_soft_subs(video_path: Path, srt_path: Path, out_path: Path) -> None:
    # Attach subtitles as a soft track (mov_text works well for MP4 players)
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(srt_path),
        "-map", "0",
        "-map", "1",
        "-c", "copy",
        "-c:s", "mov_text",
        str(out_path)
    ])

def make_silence_wav(duration_sec: float, out_path: Path, sample_rate: int = 48000) -> None:
    # Generate silent audio of duration_sec
    if duration_sec <= 0:
        # create a tiny silence to avoid concat issues
        duration_sec = 0.01
    run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{duration_sec:.3f}",
        "-q:a", "9",
        str(out_path)
    ])

def concat_wavs(list_file: Path, out_path: Path) -> None:
    # Concat using ffmpeg concat demuxer
    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path)
    ])


# -----------------------------
# OpenAI steps
# -----------------------------
def transcribe_with_timestamps(client: OpenAI, wav_path: Path, source_lang: Optional[str]) -> Dict[str, Any]:
    with wav_path.open("rb") as f:
        # We ask for verbose JSON + segment timestamps.
        # The docs call this timestamp_granularities[] (segment/word) in the transcriptions API.
        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=source_lang if source_lang else None,
        )
    return json.loads(resp.model_dump_json())

def translate_segments(client: OpenAI, segments: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """
    Translates segment text while keeping the same timing.
    """
    translated: List[Dict[str, Any]] = []
    for seg in tqdm(segments, desc=f"Translating -> {target_lang}"):
        text = str(seg.get("text", "")).strip()
        if not text:
            translated.append({**seg, "text_translated": ""})
            continue

        # Keep it short and subtitle-friendly.
        prompt = (
            f"Translate the following to {target_lang}. "
            f"Make it natural, concise, and suitable for subtitles. "
            f"Do not add explanations.\n\nTEXT:\n{text}"
        )
        r = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        out = (r.output_text or "").strip()
        translated.append({**seg, "text_translated": out})
    return translated

def tts_segment(client: OpenAI, text: str, out_mp3: Path, voice: str) -> None:
    """
    Generate TTS audio (mp3) for a single segment.
    """
    # The TTS endpoint is /v1/audio/speech
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    # SDK returns bytes-like content; write to file
    out_mp3.write_bytes(audio.read())

def mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int = 48000) -> None:
    run([
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        str(wav_path)
    ])


# -----------------------------
# Main orchestration
# -----------------------------
def build_dub_track(
    client: OpenAI,
    segments: List[Dict[str, Any]],
    workdir: Path,
    voice: str,
    sample_rate: int = 48000,
) -> Path:
    """
    Creates a single WAV track aligned to segment start times by inserting silence.
    Uses translated text if present as text_translated; otherwise falls back to text.
    """
    parts_dir = workdir / "dub_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    concat_list = parts_dir / "concat.txt"
    lines = []

    current_t = 0.0
    for i, seg in enumerate(tqdm(segments, desc="TTS + aligning audio")):
        start = float(seg["start"])
        end = float(seg["end"])
        text = (seg.get("text_translated") or seg.get("text") or "").strip()
        if not text:
            # If no text, just keep timing by inserting silence over its duration
            gap = max(0.0, end - current_t)
            sil = parts_dir / f"{i:05d}_sil.wav"
            make_silence_wav(gap, sil, sample_rate=sample_rate)
            lines.append(f"file '{sil.as_posix()}'")
            current_t = end
            continue

        # Insert silence until this segment's start
        gap = max(0.0, start - current_t)
        if gap > 0.01:
            sil = parts_dir / f"{i:05d}_gap.wav"
            make_silence_wav(gap, sil, sample_rate=sample_rate)
            lines.append(f"file '{sil.as_posix()}'")
            current_t = start

        # TTS for this segment
        mp3 = parts_dir / f"{i:05d}_{safe_filename(text)}.mp3"
        wav = parts_dir / f"{i:05d}_{safe_filename(text)}.wav"
        tts_segment(client, text, mp3, voice=voice)
        mp3_to_wav(mp3, wav, sample_rate=sample_rate)
        lines.append(f"file '{wav.as_posix()}'")

        # Move current time forward to at least segment end (keeps overall pacing reasonable)
        current_t = max(current_t, end)

    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_wav = workdir / "dubbed_track.wav"
    concat_wavs(concat_list, out_wav)
    return out_wav


def main():
    ensure_ffmpeg()

    ap = argparse.ArgumentParser(description="Local video -> translated subtitles + optional dubbed audio -> final MP4")
    ap.add_argument("--input", required=True, help="Path to input video (mp4)")
    ap.add_argument("--target-lang", default="pt-BR", help="Target language, e.g. pt-BR, pt-PT, es, fr")
    ap.add_argument("--source-lang", default=None, help="Optional source language code, e.g. en. Leave blank to auto-detect.")
    ap.add_argument("--voice", default="alloy", help="TTS voice (e.g., alloy, ash, nova, shimmer...)")
    ap.add_argument("--dub", action="store_true", help="Generate dubbed audio in target language")
    ap.add_argument("--subs", choices=["none", "srt", "burn", "soft"], default="srt",
                    help="Subtitle output: srt only, burn into video, attach as soft track, or none")
    ap.add_argument("--out", default=None, help="Output mp4 path. Default: <input>_OUT.mp4")
    args = ap.parse_args()

    video_path = Path(args.input).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_path = Path(args.out).expanduser().resolve() if args.out else video_path.with_name(video_path.stem + "_OUT.mp4")

    workdir = video_path.with_name(video_path.stem + "_work")
    workdir.mkdir(exist_ok=True)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 1) Extract audio
    wav_path = workdir / "audio_16k_mono.wav"
    extract_audio(video_path, wav_path)

    # 2) Transcribe with timestamps
    tr = transcribe_with_timestamps(client, wav_path, args.source_lang)
    segments = tr.get("segments") or []
    if not segments:
        raise RuntimeError("No segments returned from transcription. Check API key / model / audio.")

    # Save raw transcript JSON for debugging
    (workdir / "transcription_verbose.json").write_text(json.dumps(tr, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Translate segments
    translated_segments = translate_segments(client, segments, args.target_lang)

    # 4) Write SRT
    srt_path = workdir / f"subs_{args.target_lang}.srt"
    write_srt(translated_segments, srt_path, text_key="text_translated")
    print(f"Subtitles written: {srt_path}")

    # Start from original video
    temp_video = video_path

    # 5) Dub track + mux (optional)
    if args.dub:
        dub_wav = build_dub_track(client, translated_segments, workdir, voice=args.voice)
        dubbed_mp4 = workdir / "video_with_dub.mp4"
        mux_audio(video_path, dub_wav, dubbed_mp4)
        temp_video = dubbed_mp4
        print(f"Dubbed audio muxed: {dubbed_mp4}")

    # 6) Subtitles output mode
    if args.subs == "none":
        shutil.copyfile(temp_video, out_path)
    elif args.subs == "srt":
        # Keep video as-is; just output mp4 + srt side-by-side
        shutil.copyfile(temp_video, out_path)
        print("Note: subtitles are in the .srt file (not embedded).")
    elif args.subs == "burn":
        burn_subtitles(temp_video, srt_path, out_path)
    elif args.subs == "soft":
        attach_soft_subs(temp_video, srt_path, out_path)

    print(f"\nDONE ✅ Output video: {out_path}")
    if args.subs in ("srt", "burn", "soft"):
        print(f"Subs file: {srt_path}")


if __name__ == "__main__":
    main()