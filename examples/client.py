"""
Minimal API client: submit a file, poll until done, print the transcript.

    python examples/client.py meeting.mp3 --format srt --base-url http://localhost:8000
"""

import argparse
import sys
import time

import httpx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--format", default="txt", choices=["json", "srt", "vtt", "txt"])
    p.add_argument("--language", default=None)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        with open(args.file, "rb") as fh:
            r = client.post(
                "/v1/transcriptions",
                files={"file": (args.file, fh)},
                params={"language": args.language} if args.language else None,
            )
        if r.status_code != 202:
            print(f"submit failed: {r.status_code} {r.text}", file=sys.stderr)
            return 1
        job = r.json()
        print(f"job {job['job_id']} queued", file=sys.stderr)

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            status = client.get(job["status_url"]).json()
            if status["status"] in ("done", "dead_letter", "failed"):
                break
            time.sleep(2)
        else:
            print("timed out", file=sys.stderr)
            return 1

        print(f"status: {status['status']}", file=sys.stderr)
        if status["status"] == "failed":
            print(status["error"], file=sys.stderr)
            return 1
        if status.get("needs_review"):
            print("warning: low confidence, flagged for review", file=sys.stderr)

        t = client.get(f"{job['status_url']}/transcript", params={"format": args.format})
        print(t.text)
        return 0


if __name__ == "__main__":
    sys.exit(main())
