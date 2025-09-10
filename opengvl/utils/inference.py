import json
import re
from collections.abc import Iterable
from pathlib import Path

from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult
from opengvl.utils.errors import PercentagesCountMismatch, PercentagesNormalizationError

PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def extract_percentages(
    text: str,
    expected: int,
) -> list[int]:
    """Extract percentages in order of appearance and return integers.

    - Accepts both integer and floating-point percentages in the input text.
    - If any extracted value has a fractional part, round the list so that
      the final integers sum to 100 using the largest remainder method.
    - For purely integer inputs, values are returned as-is (cast to int).

    Args:
        text: Source text.
        expected: Expected number of percentages. The function
            will validate that exactly this many values are present. If the
            count does not match, a ValueError is raised. Extraction does not
            truncate; all percentages found are considered.
    Returns:
        List of integer percentages within [0, 100].
    """
    import math

    vals: list[float] = []
    for match in PERCENT_FLOAT_RE.finditer(text):
        try:
            v = float(match.group(1))
        except ValueError:
            continue
        if not (0.0 <= v <= 100.0):
            continue
        vals.append(v)

    # If no values found, return empty list
    if not vals:
        return []

    # Enforce expected length if provided
    if expected is not None and len(vals) != expected:
        raise PercentagesCountMismatch(expected, len(vals))

    has_fractional = any((v % 1) != 0 for v in vals)
    if not has_fractional:
        # All values are already integers; just cast
        return [int(v) for v in vals]

    total = sum(vals)
    if total <= 0:
        # Degenerate case; cannot normalize meaningfully
        raise PercentagesNormalizationError()

    # Normalize to sum to 100, then distribute remainders
    scale = 100.0 / total
    scaled = [v * scale for v in vals]
    floors = [math.floor(x) for x in scaled]
    remainders = [x - f for x, f in zip(scaled, floors)]
    current_sum = sum(floors)
    need = int(100 - current_sum)

    # Indices sorted by largest remainder (stable by original index for ties)
    order = sorted(range(len(vals)), key=lambda i: (-remainders[i], i))
    result = floors[:]
    for i in range(min(max(need, 0), len(result))):
        result[order[i]] += 1

    return list(map(int, result))


def build_inferred_example(
    fewshot: FewShotInput,
    predicted: list[int],
) -> InferredFewShotResult:
    inferred_ep = InferredEpisode.from_predictions(fewshot.eval_episode, predictions=predicted)
    return InferredFewShotResult(eval_episode=inferred_ep, context_episodes=fewshot.context_episodes)


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
