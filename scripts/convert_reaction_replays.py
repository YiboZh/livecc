#!/usr/bin/env python
"""Convert cleaned reaction descriptions into LiveCC-style conversations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Iterable, List

try:
    import decord
except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "decord is required to validate video usability. Please install it with `pip install decord`."
        ) from exc


@dataclass
class ConversionStats:
    total_ids: int = 0
    converted: int = 0
    skipped_missing_description: int = 0
    skipped_missing_video: int = 0
    skipped_empty_rows: int = 0
    skipped_unusable_video: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-list", required=True, type=Path, help="Path to the filtered ID list (one per line).")
    parser.add_argument("--video-dir", required=True, type=Path, help="Directory that stores <id>.mp4 stimuli videos.")
    parser.add_argument(
        "--description-dir", required=True, type=Path, help="Directory that stores <id>.csv description files."
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL path (will include seek footer).")
    parser.add_argument(
        "--emotion-fallback",
        default="neutral",
        help="Fallback emotion text when the CSV field is empty.",
    )
    parser.add_argument(
        "--reaction-fallback",
        default="neutral reaction",
        help="Fallback reaction text when the CSV field is empty.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1e-3,
        help="Minimum segment duration (seconds); rows shorter than this are dropped.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_id_list(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                ids.append(stripped)
    return ids


def wrap(tag: str, text: str) -> str:
    return f"<{tag}>{text}</{tag}>"


def video_has_frames(video_path: Path, *, min_frames: int = 2) -> bool:
    """Return True when the video is readable and contains enough frames."""
    try:
        vr = decord.VideoReader(str(video_path), num_threads=1)
    except Exception as exc:
        logging.warning("Skipping %s: unable to read video (%s)", video_path, exc)
        return False
    total_frames = len(vr)
    if total_frames < min_frames:
        logging.warning("Skipping %s: only %d frame(s) detected", video_path, total_frames)
        return False
    return True


def build_conversation(
    id_str: str,
    video_path: Path,
    desc_path: Path,
    emotion_fallback: str,
    reaction_fallback: str,
    *,
    min_duration: float,
) -> list[dict] | None:
    conversation: list[dict] = []
    with desc_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                start = float(row.get("start_time_s", "0") or 0)
                end = float(row.get("end_time_s", "0") or 0)
            except ValueError:
                logging.debug("%s: failed to parse timestamps in row=%s", id_str, row)
                continue
            duration = end - start
            if duration < min_duration:
                continue
            emotion = (row.get("emotion_description") or emotion_fallback).strip() or emotion_fallback
            reaction = (row.get("reaction_description") or reaction_fallback).strip() or reaction_fallback

            user_turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Time={start:.2f}-{end:.2f}s"},
                    {
                        "type": "video",
                        "video": str(video_path),
                        "video_start": start,
                        "video_end": end,
                    },
                ],
            }
            assistant_turn = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"{wrap('emotion', emotion)}{wrap('reaction', reaction)}",
                    }
                ],
            }
            conversation.extend([user_turn, assistant_turn])

    return conversation or None


def write_jsonl(conversations: Iterable[list[dict]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths: List[int] = []
    count = 0
    with output_path.open("w") as f:
        for conversation in conversations:
            line = json.dumps(conversation, ensure_ascii=True)
            f.write(line + "\n")
            lengths.append(len(line) + 1)  # include newline for accurate seeking
            count += 1

        if lengths:
            cumulative = list(accumulate(lengths[:-1]))
            seeks = [0] + cumulative
        else:
            seeks = [0]
        f.write(json.dumps(seeks))
    return count


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    ids = load_id_list(args.id_list)
    stats = ConversionStats(total_ids=len(ids))

    def generate_conversations() -> Iterable[list[dict]]:
        for id_str in ids:
            video_path = args.video_dir / f"{id_str}.mp4"
            if not video_path.exists():
                stats.skipped_missing_video += 1
                logging.debug("Skipping %s: video not found", id_str)
                continue
            if not video_has_frames(video_path):
                stats.skipped_unusable_video += 1
                continue
            desc_path = args.description_dir / f"{id_str}.csv"
            if not desc_path.exists():
                stats.skipped_missing_description += 1
                logging.debug("Skipping %s: description not found", id_str)
                continue
            conversation = build_conversation(
                id_str,
                video_path,
                desc_path,
                args.emotion_fallback,
                args.reaction_fallback,
                min_duration=args.min_duration,
            )
            if not conversation:
                stats.skipped_empty_rows += 1
                logging.debug("Skipping %s: empty conversation", id_str)
                continue
            stats.converted += 1
            yield conversation

    converted = write_jsonl(generate_conversations(), args.output)
    logging.info("Wrote %d conversations to %s", converted, args.output)
    logging.info("Stats: %s", json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
