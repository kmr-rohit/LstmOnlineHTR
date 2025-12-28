#!/usr/bin/env python3
"""Data inspection utility for understanding stroke data formats."""

import os
import sys
import json
import argparse
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def inspect_json_file(filepath):
    """Inspect a single JSON file."""
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print('='*60)

    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"Type: List with {len(data)} items")
        if data:
            print(f"\nFirst item structure:")
            inspect_item(data[0], indent=2)

            # Collect statistics
            labels = [item.get('label', item.get('text', '')) for item in data]
            print(f"\nStatistics:")
            print(f"  Total samples: {len(data)}")
            print(f"  Unique labels: {len(set(labels))}")

            if labels:
                lengths = [len(l) for l in labels]
                print(f"  Label length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

                # Character frequency
                all_chars = ''.join(labels)
                char_freq = Counter(all_chars)
                print(f"  Unique characters: {len(char_freq)}")
                print(f"  Top 10 chars: {dict(char_freq.most_common(10))}")
    else:
        print(f"Type: Single object")
        inspect_item(data, indent=2)


def inspect_item(item, indent=0):
    """Inspect a single data item."""
    prefix = ' ' * indent

    if isinstance(item, dict):
        print(f"{prefix}Keys: {list(item.keys())}")
        for key, value in item.items():
            if key in ['label', 'text']:
                print(f"{prefix}  {key}: '{value}'")
            elif key in ['strokes', 'stroke_data', 'points']:
                if isinstance(value, list):
                    print(f"{prefix}  {key}: List with {len(value)} items")
                    if value:
                        first = value[0]
                        if isinstance(first, dict):
                            print(f"{prefix}    First item keys: {list(first.keys())}")
                            # Check for nested points
                            if 'points' in first:
                                points = first['points']
                                if points:
                                    print(f"{prefix}      Points: {len(points)} items")
                                    print(f"{prefix}      Point keys: {list(points[0].keys()) if isinstance(points[0], dict) else 'values'}")
                            else:
                                # Show sample values
                                sample_keys = list(first.keys())[:5]
                                for k in sample_keys:
                                    v = first[k]
                                    if isinstance(v, list):
                                        print(f"{prefix}      {k}: list of {len(v)} items")
                                    else:
                                        print(f"{prefix}      {k}: {v}")
                        elif isinstance(first, list):
                            print(f"{prefix}    First stroke: {len(first)} points")
                            if first:
                                print(f"{prefix}    Point format: {type(first[0])}")
                                if isinstance(first[0], dict):
                                    print(f"{prefix}    Point keys: {list(first[0].keys())}")
            else:
                if isinstance(value, (str, int, float, bool)):
                    print(f"{prefix}  {key}: {value}")
                elif isinstance(value, list):
                    print(f"{prefix}  {key}: List[{len(value)}]")
                elif isinstance(value, dict):
                    print(f"{prefix}  {key}: Dict with keys {list(value.keys())}")


def inspect_directory(dirpath):
    """Inspect all JSON files in a directory."""
    json_files = [f for f in os.listdir(dirpath) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {dirpath}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Inspect first few files
    for f in json_files[:3]:
        inspect_json_file(os.path.join(dirpath, f))

    if len(json_files) > 3:
        print(f"\n... and {len(json_files) - 3} more files")


def main():
    parser = argparse.ArgumentParser(description='Inspect stroke data format')
    parser.add_argument('path', type=str, help='Path to JSON file or directory')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        inspect_json_file(args.path)
    elif os.path.isdir(args.path):
        inspect_directory(args.path)
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
