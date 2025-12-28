#!/usr/bin/env python3
"""Parse IAM-OnDB dataset and convert to training format."""

import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_stroke_xml(xml_path):
    """
    Parse IAM-OnDB stroke XML file.

    Returns:
        List of strokes, each stroke is a dict with 'points' list
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

    strokes = []

    # Find all Stroke elements
    for stroke_elem in root.iter('Stroke'):
        points = []

        for point_elem in stroke_elem.findall('Point'):
            x = float(point_elem.get('x', 0))
            y = float(point_elem.get('y', 0))
            t = float(point_elem.get('time', 0))

            points.append({
                'x': x,
                'y': y,
                't': t,
                'pen_down': True
            })

        if points:
            strokes.append({'points': points})

    return strokes if strokes else None


def parse_mlf_labels(mlf_path):
    """
    Parse HTK Master Label File (MLF) format.

    Returns:
        Dict mapping file_id to text label
    """
    labels = {}
    current_id = None
    current_chars = []

    with open(mlf_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line == '#!MLF!#':
                continue

            # New file entry
            if line.startswith('"') and line.endswith('"'):
                # Save previous entry
                if current_id and current_chars:
                    text = chars_to_text(current_chars)
                    labels[current_id] = text

                # Extract file ID from path like "/scratch/.../a02-050-01.lab"
                path = line.strip('"')
                filename = os.path.basename(path)
                current_id = filename.replace('.lab', '')
                current_chars = []
            else:
                # Character label
                current_chars.append(line)

        # Don't forget last entry
        if current_id and current_chars:
            text = chars_to_text(current_chars)
            labels[current_id] = text

    return labels


def chars_to_text(chars):
    """Convert character list to text string."""
    text = []

    for char in chars:
        if char == 'sp':
            text.append(' ')
        elif char == 'ga':
            # 'ga' seems to be garbage/special - skip or use a placeholder
            continue
        elif char == '.':
            text.append('.')
        elif len(char) == 1:
            text.append(char)
        else:
            # Multi-char tokens - could be special symbols
            text.append(char)

    return ''.join(text).strip()


def parse_split_file(split_path):
    """Parse train/test split file containing form IDs."""
    ids = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def find_stroke_files(linestrokes_dir, form_id):
    """Find all stroke XML files for a given form ID."""
    files = []

    # Form ID format: a01-020x -> look in a01/a01-020/a01-020x-*.xml
    parts = form_id.split('-')
    if len(parts) >= 2:
        writer = parts[0]  # e.g., 'a01'
        form = '-'.join(parts[:2])  # e.g., 'a01-020'

        # Try different directory structures
        possible_dirs = [
            os.path.join(linestrokes_dir, writer, form),
            os.path.join(linestrokes_dir, writer, form_id),
            os.path.join(linestrokes_dir, writer),
        ]

        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    if f.startswith(form_id) and f.endswith('.xml'):
                        files.append(os.path.join(dir_path, f))

    return sorted(files)


def convert_dataset(data_dir, output_dir):
    """Convert IAM-OnDB to training JSON format."""
    linestrokes_dir = os.path.join(data_dir, 'lineStrokes')
    labels_path = os.path.join(data_dir, 'labels.mlf')

    # Parse labels
    print("Parsing labels...")
    labels = parse_mlf_labels(labels_path)
    print(f"Found {len(labels)} labels")

    # Parse split files
    splits = {}
    for split_name in ['trainset.txt', 'testset_f.txt', 'testset_t.txt', 'testset_v.txt']:
        split_path = os.path.join(data_dir, split_name)
        if os.path.exists(split_path):
            splits[split_name] = parse_split_file(split_path)
            print(f"{split_name}: {len(splits[split_name])} forms")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    all_samples = []
    samples_by_split = defaultdict(list)

    # Find all XML files
    print("\nFinding stroke files...")
    all_xml_files = list(Path(linestrokes_dir).rglob('*.xml'))
    print(f"Found {len(all_xml_files)} stroke files")

    # Build mapping from line ID to XML file
    id_to_file = {}
    for xml_file in all_xml_files:
        line_id = xml_file.stem  # e.g., 'a01-020x-00'
        id_to_file[line_id] = xml_file

    # Process files with labels
    print("\nProcessing files...")
    processed = 0
    skipped = 0

    for line_id, label in tqdm(labels.items()):
        if line_id not in id_to_file:
            skipped += 1
            continue

        xml_file = id_to_file[line_id]
        strokes = parse_stroke_xml(xml_file)

        if not strokes:
            skipped += 1
            continue

        if not label or len(label) == 0:
            skipped += 1
            continue

        sample = {
            'id': line_id,
            'strokes': strokes,
            'label': label
        }

        # Determine which split this belongs to
        form_id = '-'.join(line_id.split('-')[:2])  # e.g., 'a01-020'
        form_id_full = '-'.join(line_id.split('-')[:-1])  # e.g., 'a01-020x'

        assigned = False
        for split_name, form_ids in splits.items():
            for fid in form_ids:
                if form_id_full.startswith(fid) or fid.startswith(form_id):
                    samples_by_split[split_name].append(sample)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            samples_by_split['unassigned'].append(sample)

        all_samples.append(sample)
        processed += 1

    print(f"\nProcessed: {processed}, Skipped: {skipped}")

    # Save splits
    # Combine test sets into one
    train_samples = samples_by_split.get('trainset.txt', [])
    test_samples = (
        samples_by_split.get('testset_f.txt', []) +
        samples_by_split.get('testset_t.txt', []) +
        samples_by_split.get('testset_v.txt', [])
    )
    unassigned = samples_by_split.get('unassigned', [])

    # Add unassigned to train
    train_samples.extend(unassigned)

    # Split train into train/val (90/10)
    import random
    random.seed(42)
    random.shuffle(train_samples)

    val_size = int(len(train_samples) * 0.1)
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]

    # Save files
    splits_to_save = {
        'train.json': train_samples,
        'val.json': val_samples,
        'test.json': test_samples
    }

    for filename, samples in splits_to_save.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(samples, f)
        print(f"Saved {filename}: {len(samples)} samples")

    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)

    all_labels = [s['label'] for s in all_samples]
    all_chars = set(''.join(all_labels))

    print(f"Total samples: {len(all_samples)}")
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")
    print(f"Unique characters: {len(all_chars)}")
    print(f"Characters: {''.join(sorted(all_chars))}")

    label_lengths = [len(l) for l in all_labels]
    print(f"Label length: min={min(label_lengths)}, max={max(label_lengths)}, avg={sum(label_lengths)/len(label_lengths):.1f}")

    # Sample examples
    print("\nSample examples:")
    for i in range(min(5, len(all_samples))):
        s = all_samples[i]
        print(f"  [{s['id']}] '{s['label']}' ({len(s['strokes'])} strokes)")


def main():
    parser = argparse.ArgumentParser(description='Parse IAM-OnDB dataset')
    parser.add_argument('--data-dir', type=str, default='data/db',
                        help='Path to IAM-OnDB data directory')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for JSON files')
    args = parser.parse_args()

    convert_dataset(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
