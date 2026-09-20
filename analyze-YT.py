#!/usr/bin/env python3
"""
Send a video link to a Nemotron VL NIM (or any OpenAI-compatible VLM endpoint)
and get structured play-by-play JSON back.

Usage (run on the Brev instance, next to the NIM):
    python3 analyze.py "https://www.youtube.com/watch?v=..." --out plays.json
    python3 analyze.py "https://example.com/game.mp4" --send-url   # short direct links only

Requires: pip install openai yt-dlp    and    ffmpeg (sudo apt-get install -y ffmpeg)
"""
import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.getenv("NIM_MODEL", "nvidia/nemotron-nano-12b-v2-vl")

DEFAULT_PROMPT = """You are analyzing a short clip of a basketball game or practice.
List every distinct play you can see, in order. Respond with ONLY a JSON array
(no prose, no markdown fences). Each item must have these keys:
  "time_sec": approximate seconds from the START OF THIS CLIP (number),
  "player": short description (jersey number or color if visible, else "unknown"),
  "action": one of "shot", "pass", "dribble", "rebound", "steal", "block", "turnover", "foul", "other",
  "outcome": short description (e.g. "made 2-pointer", "missed", "intercepted"),
  "notes": optional extra detail.
If nothing notable happens, return []."""


def run(cmd):
    subprocess.run(cmd, check=True)


def download(url, workdir):
    """Download the video (YouTube or direct link) to workdir/source.mp4 using yt-dlp."""
    template = str(Path(workdir) / "source.%(ext)s")
    run([
        "yt-dlp",
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", template,
        url,
    ])
    matches = sorted(Path(workdir).glob("source.*"))
    if not matches:
        raise RuntimeError("yt-dlp finished but no video file was found")
    return matches[0]


def make_clips(src, workdir, seconds, height, fps):
    """Re-encode small and cut into fixed-length clips so each request stays small."""
    pattern = str(Path(workdir) / "clip_%03d.mp4")
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"scale=-2:{height}", "-r", str(fps),
        "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-an",
        "-force_key_frames", f"expr:gte(t,n_forced*{seconds})",
        "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1",
        pattern,
    ])
    return sorted(Path(workdir).glob("clip_*.mp4"))


def parse_json_list(text):
    """Models sometimes wrap JSON in fences or add chatter; pull out the array."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, list) else None


def ask_model(client, model, video_url, prompt, retries=1):
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                }],
                max_tokens=1024,
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:  # network hiccup, 500 from a bad URL, etc.
            last_err = e
    raise RuntimeError(f"Model request failed: {last_err}")


def analyze_link(url, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                 clip_seconds=20, height=480, fps=8, send_url=False,
                 prompt=DEFAULT_PROMPT, log=print):
    """Return a list of {clip, start_sec, plays, raw} dicts."""
    client = OpenAI(base_url=base_url, api_key=os.getenv("NIM_API_KEY", "not-used"))

    # Fast path: hand the link straight to the NIM (must be a direct, public video file).
    if send_url:
        log(f"Sending link directly: {url}")
        raw = ask_model(client, model, url, prompt)
        return [{"clip": 0, "start_sec": 0, "plays": parse_json_list(raw) or [], "raw": raw}]

    results = []
    with tempfile.TemporaryDirectory() as workdir:
        log("Downloading video...")
        src = download(url, workdir)
        log("Cutting into clips...")
        clips = make_clips(src, workdir, clip_seconds, height, fps)
        log(f"{len(clips)} clip(s)")

        for i, clip in enumerate(clips):
            b64 = base64.b64encode(clip.read_bytes()).decode()
            log(f"Analyzing clip {i + 1}/{len(clips)} ({len(b64) // 1024} KB)...")
            raw = ask_model(client, model, f"data:video/mp4;base64,{b64}", prompt)
            plays = parse_json_list(raw)
            offset = i * clip_seconds
            for p in plays or []:
                if isinstance(p.get("time_sec"), (int, float)):
                    p["game_time_sec"] = round(offset + p["time_sec"], 1)
            results.append({
                "clip": i,
                "start_sec": offset,
                "plays": plays if plays is not None else [],
                "raw": raw if plays is None else None,  # keep raw text only if parsing failed
            })
    return results


def main():
    ap = argparse.ArgumentParser(description="Analyze a video link with a Nemotron VL NIM")
    ap.add_argument("url", help="YouTube link or direct video URL")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--clip-seconds", type=int, default=20)
    ap.add_argument("--height", type=int, default=480, help="clip height in pixels")
    ap.add_argument("--fps", type=int, default=8, help="clip frame rate")
    ap.add_argument("--send-url", action="store_true",
                    help="pass the URL to the NIM as-is (direct, public video file only)")
    ap.add_argument("--out", default="plays.json")
    args = ap.parse_args()

    results = analyze_link(
        args.url, base_url=args.base_url, model=args.model,
        clip_seconds=args.clip_seconds, height=args.height, fps=args.fps,
        send_url=args.send_url,
    )
    Path(args.out).write_text(json.dumps(results, indent=2))
    total = sum(len(r["plays"]) for r in results)
    print(f"Done: {total} plays across {len(results)} clip(s) -> {args.out}")


if __name__ == "__main__":
    main()