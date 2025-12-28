#!/usr/bin/env python3
"""Data preparation utility - converts various stroke formats to unified format."""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def convert_to_unified_format(data):
    """
    Convert various stroke data formats to unified format.

    Unified format:
    {
        "strokes": [
            {
                "points": [
                    {"x": float, "y": float, "t": float, "pen_down": bool},
                    ...
                ]
            },
            ...
        ],
        "label": "text"
    }
    """
    if isinstance(data, list):
        return [convert_to_unified_format(item) for item in data]

    result = {"strokes": [], "label": ""}

    # Extract label
    result["label"] = data.get("label", data.get("text", data.get("transcription", "")))

    # Extract strokes - handle various formats
    strokes = data.get("strokes", data.get("stroke_data", data.get("ink", [])))

    if not strokes:
        # Maybe strokes are at top level with x, y arrays
        if "x" in data and "y" in data:
            strokes = [data]

    for stroke in strokes:
        unified_stroke = {"points": []}

        if isinstance(stroke, dict):
            # Format 1: {"points": [{"x": ..., "y": ...}, ...]}
            if "points" in stroke:
                for p in stroke["points"]:
                    point = {
                        "x": float(p.get("x", p.get("X", 0))),
                        "y": float(p.get("y", p.get("Y", 0))),
                        "t": float(p.get("t", p.get("time", p.get("timestamp", 0)))),
                        "pen_down": p.get("pen_down", p.get("penDown", True))
                    }
                    unified_stroke["points"].append(point)

            # Format 2: {"x": [...], "y": [...], "t": [...]}
            elif "x" in stroke and "y" in stroke:
                x_vals = stroke["x"]
                y_vals = stroke["y"]
                t_vals = stroke.get("t", stroke.get("time", list(range(len(x_vals)))))
                pen_vals = stroke.get("pen_down", [True] * len(x_vals))

                for i in range(len(x_vals)):
                    point = {
                        "x": float(x_vals[i]),
                        "y": float(y_vals[i]),
                        "t": float(t_vals[i]) if i < len(t_vals) else float(i),
                        "pen_down": pen_vals[i] if i < len(pen_vals) else True
                    }
                    unified_stroke["points"].append(point)

        # Format 3: [[x, y, t], [x, y, t], ...] or [[x, y], ...]
        elif isinstance(stroke, list) and stroke:
            for p in stroke:
                if isinstance(p, (list, tuple)):
                    point = {
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "t": float(p[2]) if len(p) > 2 else 0.0,
                        "pen_down": True
                    }
                    unified_stroke["points"].append(point)
                elif isinstance(p, dict):
                    point = {
                        "x": float(p.get("x", p.get("X", 0))),
                        "y": float(p.get("y", p.get("Y", 0))),
                        "t": float(p.get("t", p.get("time", 0))),
                        "pen_down": p.get("pen_down", True)
                    }
                    unified_stroke["points"].append(point)

        if unified_stroke["points"]:
            result["strokes"].append(unified_stroke)

    return result


def process_file(input_path, output_path):
    """Process a single JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    converted = convert_to_unified_format(data)

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    return len(converted) if isinstance(converted, list) else 1


def process_directory(input_dir, output_dir):
    """Process all JSON files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for json_file in input_path.glob("*.json"):
        output_file = output_path / json_file.name
        count = process_file(str(json_file), str(output_file))
        total_samples += count
        print(f"Processed {json_file.name}: {count} samples")

    print(f"\nTotal: {total_samples} samples")


def merge_to_single_file(input_dir, output_file):
    """Merge all JSON files into a single file."""
    all_data = []
    input_path = Path(input_dir)

    for json_file in sorted(input_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)

        converted = convert_to_unified_format(data)

        if isinstance(converted, list):
            all_data.extend(converted)
        else:
            all_data.append(converted)

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Merged {len(all_data)} samples into {output_file}")


def split_data(input_file, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets."""
    import random

    with open(input_file, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    # Shuffle
    random.seed(42)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train.json': data[:train_end],
        'val.json': data[train_end:val_end],
        'test.json': data[val_end:]
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename, split_data in splits.items():
        with open(output_path / filename, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"{filename}: {len(split_data)} samples")


def main():
    parser = argparse.ArgumentParser(description='Prepare stroke data for training')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert to unified format')
    convert_parser.add_argument('input', help='Input file or directory')
    convert_parser.add_argument('output', help='Output file or directory')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge files into single file')
    merge_parser.add_argument('input_dir', help='Input directory')
    merge_parser.add_argument('output_file', help='Output file')

    # Split command
    split_parser = subparsers.add_parser('split', help='Split into train/val/test')
    split_parser.add_argument('input', help='Input file')
    split_parser.add_argument('output_dir', help='Output directory')
    split_parser.add_argument('--train', type=float, default=0.8, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.1, help='Validation ratio')

    args = parser.parse_args()

    if args.command == 'convert':
        if os.path.isfile(args.input):
            process_file(args.input, args.output)
        else:
            process_directory(args.input, args.output)

    elif args.command == 'merge':
        merge_to_single_file(args.input_dir, args.output_file)

    elif args.command == 'split':
        split_data(args.input, args.output_dir, args.train, args.val)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
