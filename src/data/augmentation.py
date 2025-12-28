"""Data augmentation for stroke sequences."""

import numpy as np


class StrokeAugmentation:
    """Augmentation transforms for stroke feature sequences."""

    def __init__(self,
                 scale_range=(0.8, 1.2),
                 rotation_range=(-15, 15),
                 noise_std=0.5,
                 time_stretch_range=(0.8, 1.2)):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.time_stretch_range = time_stretch_range

    def random_scale(self, features):
        """Random scaling of dx, dy."""
        scale = np.random.uniform(*self.scale_range)
        features = features.copy()
        features[:, :2] *= scale
        return features

    def random_rotation(self, features):
        """Random rotation of dx, dy."""
        angle = np.radians(np.random.uniform(*self.rotation_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        features = features.copy()
        features[:, :2] = features[:, :2] @ rotation_matrix.T
        return features

    def add_noise(self, features):
        """Add Gaussian noise to coordinates."""
        features = features.copy()
        noise = np.random.normal(0, self.noise_std, features[:, :2].shape)
        features[:, :2] += noise
        return features

    def time_stretch(self, features):
        """Time stretching/compression."""
        stretch = np.random.uniform(*self.time_stretch_range)
        features = features.copy()
        features[:, 2] *= stretch
        return features

    def augment(self, features, p=0.5):
        """Apply random augmentations with probability p each."""
        features = features.copy()

        if np.random.random() < p:
            features = self.random_scale(features)
        if np.random.random() < p:
            features = self.random_rotation(features)
        if np.random.random() < p:
            features = self.add_noise(features)
        if np.random.random() < p:
            features = self.time_stretch(features)

        return features
