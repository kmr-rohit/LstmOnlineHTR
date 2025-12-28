"""Stroke preprocessing module for Online HTR."""

import warnings
import numpy as np
from scipy.interpolate import interp1d

# Suppress interpolation warnings for edge cases
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.interpolate')


class StrokePreprocessor:
    """Preprocesses raw stroke data for the HTR model."""

    def __init__(self, target_height=60, resample_density=20):
        """
        Args:
            target_height: Normalize strokes to this height
            resample_density: Points per unit length after resampling
        """
        self.target_height = target_height
        self.resample_density = resample_density

    def normalize(self, strokes):
        """Normalize stroke coordinates to fixed height maintaining aspect ratio."""
        if not strokes:
            return strokes

        # Collect all points
        all_x = []
        all_y = []
        for stroke in strokes:
            if isinstance(stroke, dict) and 'points' in stroke:
                for p in stroke['points']:
                    all_x.append(p['x'])
                    all_y.append(p['y'])
            elif isinstance(stroke, dict):
                all_x.extend(stroke.get('x', []))
                all_y.extend(stroke.get('y', []))

        if not all_x or not all_y:
            return strokes

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        height = y_max - y_min
        scale = self.target_height / height if height > 0 else 1.0

        normalized = []
        for stroke in strokes:
            if isinstance(stroke, dict) and 'points' in stroke:
                norm_stroke = {
                    'x': [(p['x'] - x_min) * scale for p in stroke['points']],
                    'y': [(p['y'] - y_min) * scale for p in stroke['points']],
                    't': [p.get('t', i * 0.01) for i, p in enumerate(stroke['points'])],
                    'pen_down': [p.get('pen_down', True) for p in stroke['points']]
                }
            else:
                norm_stroke = {
                    'x': [(x - x_min) * scale for x in stroke.get('x', [])],
                    'y': [(y - y_min) * scale for y in stroke.get('y', [])],
                    't': stroke.get('t', list(range(len(stroke.get('x', []))))),
                    'pen_down': stroke.get('pen_down', [True] * len(stroke.get('x', [])))
                }
            normalized.append(norm_stroke)

        return normalized

    def resample(self, strokes):
        """Resample strokes to uniform point density."""
        resampled = []

        for stroke in strokes:
            x = np.array(stroke['x'], dtype=np.float64)
            y = np.array(stroke['y'], dtype=np.float64)
            t = np.array(stroke['t'], dtype=np.float64)

            if len(x) < 2:
                resampled.append(stroke)
                continue

            # Remove duplicate consecutive points
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.sqrt(dx**2 + dy**2)

            # Keep points that have non-zero distance from previous
            keep_mask = np.concatenate([[True], distances > 1e-6])
            x = x[keep_mask]
            y = y[keep_mask]
            t = t[keep_mask]

            if len(x) < 2:
                resampled.append(stroke)
                continue

            # Recalculate cumulative arc length
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.sqrt(dx**2 + dy**2)
            cumulative = np.concatenate([[0], np.cumsum(distances)])

            total_length = cumulative[-1]
            if total_length < 1e-6:
                resampled.append(stroke)
                continue

            num_points = max(int(total_length * self.resample_density), 2)
            new_cumulative = np.linspace(0, total_length, num_points)

            # Interpolate
            fx = interp1d(cumulative, x, kind='linear', fill_value='extrapolate')
            fy = interp1d(cumulative, y, kind='linear', fill_value='extrapolate')
            ft = interp1d(cumulative, t, kind='linear', fill_value='extrapolate')

            pen_down = stroke.get('pen_down', [True] * len(stroke['x']))

            resampled.append({
                'x': fx(new_cumulative).tolist(),
                'y': fy(new_cumulative).tolist(),
                't': ft(new_cumulative).tolist(),
                'pen_down': pen_down[:num_points] if len(pen_down) >= num_points else pen_down + [pen_down[-1]] * (num_points - len(pen_down))
            })

        return resampled

    def extract_features(self, strokes):
        """Extract delta features (dx, dy, dt, pen_state) from strokes."""
        features = []

        for i, stroke in enumerate(strokes):
            x = np.array(stroke['x'])
            y = np.array(stroke['y'])
            t = np.array(stroke['t'])

            if len(x) < 1:
                continue

            # Compute deltas
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            dt = np.diff(t, prepend=t[0])

            # Pen state: 1 at stroke start (pen was lifted before), 0 otherwise
            pen_state = np.zeros(len(dx))
            pen_state[0] = 1 if i > 0 else 0  # First point of non-first strokes

            stroke_features = np.stack([dx, dy, dt, pen_state], axis=1)
            features.append(stroke_features)

        if not features:
            return np.zeros((1, 4), dtype=np.float32)

        result = np.vstack(features).astype(np.float32)
        # Replace any NaN or Inf values with 0
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def process(self, strokes):
        """Full preprocessing pipeline."""
        if not strokes:
            return np.zeros((1, 4), dtype=np.float32)

        strokes = self.normalize(strokes)
        strokes = self.resample(strokes)
        features = self.extract_features(strokes)
        return features
