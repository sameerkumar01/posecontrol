"""Unit Tests for Biomechanical Landmark Normalizer."""

import math
import numpy as np
import pytest

from pose_control.vision.normalizer import PoseNormalizer


def test_normalizer_shape_and_root_centering():
    normalizer = PoseNormalizer()
    raw = np.random.uniform(0.1, 0.9, size=(33, 4)).astype(np.float32)

    # Set known hip coordinates
    raw[23] = [0.4, 0.8, 0.1, 0.9]  # Left hip
    raw[24] = [0.6, 0.8, 0.1, 0.9]  # Right hip

    normalized, scale = normalizer.normalize(raw)

    assert normalized.shape == (33, 3)
    assert scale > 0

    # In normalized space, mid-hip should be at origin (0, 0, 0)
    mid_hip = 0.5 * (normalized[23] + normalized[24])
    np.testing.assert_allclose(mid_hip, [0.0, 0.0, 0.0], atol=1e-5)


def test_normalizer_scale_invariance():
    normalizer1 = PoseNormalizer()
    normalizer2 = PoseNormalizer()

    # Generate canonical pose
    base = np.zeros((33, 4), dtype=np.float32)
    base[:, 3] = 0.9  # High visibility
    base[11] = [-0.2, 0.4, 0.0, 0.9]  # L Shoulder
    base[12] = [0.2, 0.4, 0.0, 0.9]   # R Shoulder
    base[23] = [-0.15, 0.0, 0.0, 0.9] # L Hip
    base[24] = [0.15, 0.0, 0.0, 0.9]  # R Hip
    base[16] = [0.3, 0.5, 0.2, 0.9]   # R Wrist

    # Scaled version (2x distance/size)
    scaled = base.copy()
    scaled[:, :3] *= 2.0

    norm1, scale1 = normalizer1.normalize(base)
    norm2, scale2 = normalizer2.normalize(scaled)

    # Normalized coordinates should be identical regardless of scale factor!
    np.testing.assert_allclose(norm1, norm2, atol=1e-4)
    assert pytest.approx(scale2, rel=1e-3) == 2.0 * scale1


def test_normalizer_imputation():
    normalizer = PoseNormalizer(visibility_threshold=0.5)

    frame1 = np.ones((33, 4), dtype=np.float32)
    frame1[15] = [0.5, 0.5, 0.5, 0.99]  # L Wrist high vis
    normalizer.normalize(frame1)

    # Frame 2: L Wrist drops visibility
    frame2 = np.ones((33, 4), dtype=np.float32)
    frame2[15] = [0.0, 0.0, 0.0, 0.10]  # Lost tracking
    norm2, _ = normalizer.normalize(frame2)

    # Imputed coordinate should not immediately crash or collapse to 0
    assert not np.isnan(norm2).any()
