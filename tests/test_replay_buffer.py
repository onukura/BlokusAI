"""Unit tests for ReplayBuffer."""

from __future__ import annotations

import pytest

from blokus_ai.replay_buffer import ReplayBuffer
from blokus_ai.selfplay import Sample


def create_dummy_sample(player: int = 0) -> Sample:
    """Create a dummy Sample for testing."""
    import numpy as np

    return Sample(
        x=np.zeros((5, 14, 14)),
        self_rem=np.zeros(21),
        opp_rem=np.zeros(21),
        moves=[],
        policy=np.array([1.0]),
        player=player,
        chosen_move_idx=0,
    )


class TestReplayBufferInitialization:
    """Test ReplayBuffer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        buffer = ReplayBuffer()
        assert buffer.max_size == 10000
        assert len(buffer) == 0

    def test_init_custom_size(self):
        """Test initialization with custom size."""
        buffer = ReplayBuffer(max_size=100)
        assert buffer.max_size == 100
        assert len(buffer) == 0

    def test_init_invalid_size_zero(self):
        """Test initialization with invalid size (zero)."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=0)

    def test_init_invalid_size_negative(self):
        """Test initialization with invalid size (negative)."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            ReplayBuffer(max_size=-10)


class TestReplayBufferAdd:
    """Test adding samples to ReplayBuffer."""

    def test_add_valid_sample(self):
        """Test adding a valid sample."""
        buffer = ReplayBuffer(max_size=10)
        sample = create_dummy_sample()
        buffer.add(sample, outcome=1)
        assert len(buffer) == 1

    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        buffer = ReplayBuffer(max_size=10)
        for i in range(5):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)
        assert len(buffer) == 5

    def test_add_invalid_outcome_too_high(self):
        """Test adding sample with invalid outcome (> 1)."""
        buffer = ReplayBuffer(max_size=10)
        sample = create_dummy_sample()
        with pytest.raises(ValueError, match="outcome must be -1, 0, or 1"):
            buffer.add(sample, outcome=2)

    def test_add_invalid_outcome_too_low(self):
        """Test adding sample with invalid outcome (< -1)."""
        buffer = ReplayBuffer(max_size=10)
        sample = create_dummy_sample()
        with pytest.raises(ValueError, match="outcome must be -1, 0, or 1"):
            buffer.add(sample, outcome=-2)


class TestReplayBufferFIFO:
    """Test FIFO eviction behavior."""

    def test_fifo_eviction(self):
        """Test that oldest samples are evicted when buffer is full."""
        buffer = ReplayBuffer(max_size=3)

        # Add 3 samples with unique chosen_move_idx for identification
        sample1 = Sample(
            x=create_dummy_sample().x,
            self_rem=create_dummy_sample().self_rem,
            opp_rem=create_dummy_sample().opp_rem,
            moves=[],
            policy=create_dummy_sample().policy,
            player=0,
            chosen_move_idx=100,  # Unique identifier
        )
        sample2 = create_dummy_sample(player=1)
        sample3 = create_dummy_sample(player=0)

        buffer.add(sample1, outcome=1)
        buffer.add(sample2, outcome=-1)
        buffer.add(sample3, outcome=0)

        assert len(buffer) == 3

        # Add one more - should evict sample1
        sample4 = create_dummy_sample(player=1)
        buffer.add(sample4, outcome=1)

        assert len(buffer) == 3

        # Sample should not include sample1 (check by chosen_move_idx)
        samples, outcomes = buffer.sample(batch_size=3)
        assert len(samples) == 3
        chosen_move_indices = [s.chosen_move_idx for s in samples]
        assert 100 not in chosen_move_indices  # sample1's unique ID should be evicted


class TestReplayBufferSample:
    """Test sampling from ReplayBuffer."""

    def test_sample_basic(self):
        """Test basic sampling."""
        buffer = ReplayBuffer(max_size=10)

        # Add samples
        for i in range(5):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        samples, outcomes = buffer.sample(batch_size=3)
        assert len(samples) == 3
        assert len(outcomes) == 3
        assert all(o == 1 for o in outcomes)

    def test_sample_batch_larger_than_buffer(self):
        """Test sampling with batch_size > buffer size."""
        buffer = ReplayBuffer(max_size=10)

        # Add 3 samples
        for i in range(3):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        # Request 5 samples - should return 3
        samples, outcomes = buffer.sample(batch_size=5)
        assert len(samples) == 3
        assert len(outcomes) == 3

    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer raises error."""
        buffer = ReplayBuffer(max_size=10)

        with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
            buffer.sample(batch_size=1)

    def test_sample_invalid_batch_size(self):
        """Test sampling with invalid batch_size."""
        buffer = ReplayBuffer(max_size=10)
        buffer.add(create_dummy_sample(), outcome=1)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            buffer.sample(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            buffer.sample(batch_size=-1)


class TestReplayBufferUtilities:
    """Test utility methods."""

    def test_is_ready_true(self):
        """Test is_ready returns True when buffer has enough samples."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(5):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        assert buffer.is_ready(min_size=3) is True
        assert buffer.is_ready(min_size=5) is True

    def test_is_ready_false(self):
        """Test is_ready returns False when buffer has too few samples."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(3):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        assert buffer.is_ready(min_size=5) is False

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = ReplayBuffer(max_size=10)

        # Add samples
        for i in range(5):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        assert len(buffer) == 5

        # Clear
        buffer.clear()
        assert len(buffer) == 0

    def test_get_utilization(self):
        """Test buffer utilization calculation."""
        buffer = ReplayBuffer(max_size=10)

        assert buffer.get_utilization() == 0.0

        # Add 5 samples
        for i in range(5):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        assert buffer.get_utilization() == 0.5

        # Fill buffer
        for i in range(5):
            sample = create_dummy_sample()
            buffer.add(sample, outcome=1)

        assert buffer.get_utilization() == 1.0


class TestReplayBufferRandomness:
    """Test randomness of sampling."""

    def test_sampling_randomness(self):
        """Test that sampling is random."""
        buffer = ReplayBuffer(max_size=100)

        # Add 10 samples with different players
        for i in range(10):
            sample = create_dummy_sample(player=i)
            buffer.add(sample, outcome=1)

        # Sample twice - should get different results (high probability)
        samples1, _ = buffer.sample(batch_size=5)
        samples2, _ = buffer.sample(batch_size=5)

        # Extract player IDs
        players1 = [s.player for s in samples1]
        players2 = [s.player for s in samples2]

        # With randomness, should be different (could rarely fail)
        # This is a probabilistic test - might need adjustment if flaky
        assert players1 != players2 or len(samples1) == 1  # Allow equal if batch_size=1
