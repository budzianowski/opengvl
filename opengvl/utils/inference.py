import json
import re
from collections.abc import Iterable
from pathlib import Path

from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult

PERCENT_FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def extract_percentages(
    text: str,
    expected: int | None = None,
    *,
    as_int: bool = True,
    round_mode: str = "nearest",
) -> list[int] | list[float]:
    """Extract percentages in order of appearance.

    Args:
        text: Source text.
        expected: If provided, truncates once this many values collected.
        as_int: Return integers (default) or raw floats.
        round_mode: One of {nearest, floor, ceil} when converting to int.
    Returns:
        List of percentages (ints or floats) within [0,100].
    """
    vals: list[float] = []
    for match in PERCENT_FLOAT_RE.finditer(text):
        try:
            v = float(match.group(1))
        except ValueError:
            continue
        if not (0.0 <= v <= 100.0):
            continue
        vals.append(v)
        if expected is not None and len(vals) >= expected:
            break

    if not as_int:
        return vals

    if round_mode == "nearest":
        conv = [round(v) for v in vals]
    elif round_mode == "floor":
        conv = [int(v // 1) for v in vals]
    elif round_mode == "ceil":
        import math

        conv = [math.ceil(v) for v in vals]
    else:  # fallback
        conv = [round(v) for v in vals]
    return [int(c) for c in conv]


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
