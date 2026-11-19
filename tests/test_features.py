"""Unit Tests for 138-Dimensional Feature Extractor and Temporal Buffer."""

import numpy as np
import pytest

from pose_control.config import TOTAL_FEATURE_DIM, SEQUENCE_LENGTH
from pose_control.features.extractor import FeatureExtractor
from pose_control.features.buffer import FeatureWindowBuffer


def test_feature_extractor_dimension():
    extractor = FeatureExtractor(fps=30.0)
    coords = np.random.uniform(-1.0, 1.0, size=(33, 3)).astype(np.float32)

    feats1 = extractor.extract(coords)
    assert feats1.shape == (TOTAL_FEATURE_DIM,)
    assert feats1.dtype == np.float32

    # Second frame extracts non-zero finite-difference velocities
    coords2 = coords + 0.05
    feats2 = extractor.extract(coords2)
    assert feats2.shape == (TOTAL_FEATURE_DIM,)

    # Verify velocities are non-zero
    velocities = feats2[99:132]
    assert np.any(velocities > 0)

    # Verify angles are in range [0, pi]
    angles = feats2[132:138]
    assert np.all(angles >= 0.0)
    assert np.all(angles <= np.pi)


def test_feature_window_buffer():
    buffer = FeatureWindowBuffer(window_size=30, feature_dim=138)
    assert buffer.is_filled()

    tensor = buffer.get_tensor()
    assert tensor.shape == (1, 30, 138)
    assert tensor.dtype == np.float32

    # Append new frame
    vec = np.ones(138, dtype=np.float32) * 5.0
    buffer.append(vec)

    new_tensor = buffer.get_tensor()
    assert new_tensor.shape == (1, 30, 138)
    # The last element should match appended vector
    np.testing.assert_allclose(new_tensor[0, -1], vec)
