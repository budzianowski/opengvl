from enum import StrEnum
from typing import Final

IMG_SIZE: Final[int] = 244
# MAX_TOKENS_TO_GENERATE: Final[int] = 1024
MAX_TOKENS_TO_GENERATE: Final[int] = 10000
N_DEBUG_PROMPT_CHARS: Final[int] = 400


class PromptPhraseKey(StrEnum):
    INITIAL_SCENE_LABEL = "initial_scene_label"
    INITIAL_SCENE_COMPLETION = "initial_scene_completion"
    CONTEXT_FRAME_LABEL_TEMPLATE = "context_frame_label_template"
    CONTEXT_FRAME_COMPLETION_TEMPLATE = "context_frame_completion_template"
    EVAL_FRAME_LABEL_TEMPLATE = "eval_frame_label_template"
    EVAL_TASK_COMPLETION_INSTRUCTION = "eval_task_completion_instruction"
