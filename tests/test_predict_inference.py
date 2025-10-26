from unittest.mock import MagicMock

import numpy as np

from opengvl.metrics.voc import VOCMetric
from opengvl.utils.data_types import Episode, FewShotInput
from opengvl.utils.inference import PercentagesNormalizationError, predict_on_fewshot_input


def make_dummy_episode(n_frames=3):
    # Use numpy arrays for images
    dummy_img = np.zeros((1, 1), dtype=np.uint8)
    return Episode(
        instruction="Do something",
        starting_frame=dummy_img,
        episode_index=0,
        original_frames_indices=list(range(n_frames)),
        shuffled_frames_indices=list(range(n_frames)),
        shuffled_frames_approx_completion_rates=[0] * n_frames,
        original_frames_task_completion_rates=[0] * n_frames,
        shuffled_frames=[dummy_img] * n_frames,
    )


def make_dummy_example(n_frames=3):
    return FewShotInput(
        eval_episode=make_dummy_episode(n_frames),
        context_episodes=[],
    )


class DummyVOCMetric(VOCMetric):
    def compute(self, example):
        return MagicMock(name="voc", value=42, details=None)


def test_percentages_normalization_error_count_plus_count_issue(monkeypatch):
    client = MagicMock()
    client.generate_response.return_value = "bad percent"
    ex = make_dummy_example()

    def raise_norm_error(text):
        raise PercentagesNormalizationError()

    monkeypatch.setattr("opengvl.utils.inference.extract_percentages", raise_norm_error)
    record = predict_on_fewshot_input(
        idx=0,
        total=1,
        ex=ex,
        client=client,
        prompt_template="{instruction}",
        save_raw=False,
        voc_metric=DummyVOCMetric(),
        dataset_name="dummy",
    )
    assert record.error_count["PercentagesNormalizationError"] == 1
    assert record.error_count["PercentagesCountMismatch"] == 1
    assert record.predicted_percentages == []


def test_count_mismatch_error_count():
    client = MagicMock()
    # Return only 2 percentages, but expect 3
    client.generate_response.return_value = "10% 90%"
    ex = make_dummy_example(n_frames=3)
    record = predict_on_fewshot_input(
        idx=0,
        total=1,
        ex=ex,
        client=client,
        prompt_template="{instruction}",
        save_raw=False,
        voc_metric=DummyVOCMetric(),
        dataset_name="dummy",
    )
    assert record.error_count["PercentagesCountMismatch"] == 1
    assert record.error_count["PercentagesNormalizationError"] == 0
    assert record.predicted_percentages == [10, 90]
