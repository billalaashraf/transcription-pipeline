"""
Generate a synthetic test clip (tone bursts separated by silence) so the pipeline
can be exercised without a real recording. Whisper will produce little or no text
for pure tones, which is itself a useful check of the no-speech / needs_review path.

    python examples/make_sample_audio.py sample.wav --seconds 30
"""

import argparse

from pydub import AudioSegment
from pydub.generators import Sine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--rate", type=int, default=44100)
    p.add_argument("--channels", type=int, default=2)
    args = p.parse_args()

    tone = Sine(440, sample_rate=args.rate).to_audio_segment(duration=3000).apply_gain(-6)
    gap = AudioSegment.silent(duration=1000, frame_rate=args.rate)
    out = AudioSegment.silent(duration=0, frame_rate=args.rate)
    while len(out) < args.seconds * 1000:
        out += tone + gap
    out = out[: int(args.seconds * 1000)].set_channels(args.channels)
    fmt = args.output.rsplit(".", 1)[-1].lower()
    out.export(args.output, format=fmt)
    print(f"wrote {args.output}: {len(out) / 1000:.1f}s, {args.rate} Hz, {args.channels}ch")


if __name__ == "__main__":
    main()
