"""
Simplest version: send ONE short video to the NIM and print what it says.

    python analyze_simple.py clip.mp4                      # a file you uploaded
    python analyze_simple.py https://example.com/clip.mp4  # a direct, public video link

Needs only:  pip install openai
"""
import base64
import sys

from openai import OpenAI

source = sys.argv[1]

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

if source.startswith("http"):
    video = source  # the NIM fetches the link itself
else:
    with open(source, "rb") as f:
        video = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

prompt = (
    "List each play in this basketball clip, in order. For each one give the "
    "approximate time in seconds, the player, the action, and the outcome. "
    "Answer as a JSON array."
)

response = client.chat.completions.create(
    model="nvidia/nemotron-nano-12b-v2-vl",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video}},
        ],
    }],
    max_tokens=1024,
    extra_body={"media_io_kwargs": {"video": {"fps": 2.0}}},
)

print(response.choices[0].message.content)