import json
import re
from collections.abc import Iterable
from pathlib import Path

from opengvl.utils.data_types import Example as FewShotInput
from opengvl.utils.data_types import InferredEpisode, InferredFewShotResult

PERCENT_RE = re.compile(r"(\d{1,3})%")


def extract_percentages(text: str, expected: int | None = None) -> list[int]:
    """Extract integer percentages (0-100) in order of appearance.

    If expected is set and we collect more than expected, we truncate; if fewer
    we return what we found (validation done later).
    """
    vals: list[int] = []
    for m in PERCENT_RE.finditer(text):
        try:
            v = int(m.group(1))
        except ValueError:
            continue
        if 0 <= v <= 100:
            vals.append(v)
        if expected is not None and len(vals) >= expected:
            break
    return vals


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
