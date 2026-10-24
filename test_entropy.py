"""Unit tests for Normalized Shannon Entropy, strided pooling, and Jensen-Shannon Divergence."""
import math
import pytest

import emotion_model as em


def test_normalized_entropy_uniform_distribution():
    k = 7
    uniform_scores = {em.EMOTIONS[i]: 1.0 / k for i in range(k)}
    h = em.compute_normalized_entropy(uniform_scores)
    assert h == pytest.approx(1.0, abs=1e-4)


def test_normalized_entropy_one_hot_distribution():
    k = 7
    one_hot = {em.EMOTIONS[i]: 0.0 for i in range(k)}
    one_hot["joy"] = 1.0
    h = em.compute_normalized_entropy(one_hot)
    assert h == pytest.approx(0.0, abs=1e-4)


def test_normalized_entropy_intermediate():
    scores = {"joy": 0.8, "sadness": 0.1, "neutral": 0.05, "fear": 0.05, "anger": 0.0, "disgust": 0.0, "surprise": 0.0}
    h = em.compute_normalized_entropy(scores)
    assert 0.0 < h < 1.0


def test_entropy_weighted_salience_prioritizes_emotional_window():
    # Window A: acute joy (peaked) -> low entropy -> high (1 - H)^1.5 weight
    # Window B: ambiguous/neutral (flat) -> high entropy -> low (1 - H)^1.5 weight
    window_a = {"joy": 0.95, "neutral": 0.05, "anger": 0.0, "disgust": 0.0, "fear": 0.0, "sadness": 0.0, "surprise": 0.0}
    window_b = {label: 1.0 / 7 for label in em.EMOTIONS}

    h_a = em.compute_normalized_entropy(window_a)
    h_b = em.compute_normalized_entropy(window_b)

    assert h_a < 0.2
    assert h_b > 0.99

    omega_a = 50 * ((1.0 - h_a) ** em.SALIENT_GAMMA)
    omega_b = 50 * ((1.0 - h_b) ** em.SALIENT_GAMMA)

    # Acute window should heavily outweigh ambiguous window of same length
    assert omega_a > 10 * omega_b


def test_jensen_shannon_divergence_properties():
    p = {label: 0.0 for label in em.EMOTIONS}
    p["joy"] = 1.0

    # Divergence with itself is zero
    assert em.jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    # Divergence with completely disjoint distribution is 1.0
    q = {label: 0.0 for label in em.EMOTIONS}
    q["sadness"] = 1.0
    assert em.jensen_shannon_divergence(p, q) == pytest.approx(1.0, abs=1e-4)

    # Symmetry: D_JS(P || Q) == D_JS(Q || P)
    r = {"joy": 0.6, "sadness": 0.4}
    s = {"fear": 0.7, "anger": 0.3}
    assert em.jensen_shannon_divergence(r, s) == pytest.approx(em.jensen_shannon_divergence(s, r), abs=1e-6)


def test_single_window_exact_softmax_preservation():
    single_window = {"joy": 0.85, "neutral": 0.10, "sadness": 0.05, "fear": 0.0, "anger": 0.0, "disgust": 0.0, "surprise": 0.0}
    # For a single window, Omega_1 * p_1 / Omega_1 = p_1
    h = em.compute_normalized_entropy(single_window)
    omega = 100 * ((1.0 - h) ** em.SALIENT_GAMMA)
    aggregated = {k: (v * omega) / omega for k, v in single_window.items()}
    assert aggregated == pytest.approx(single_window, abs=1e-6)
