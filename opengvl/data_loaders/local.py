from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from PIL import Image

from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.data_types import Example as FewShotInput

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


class LocalDataLoader(BaseDataLoader):
    """Load a single episode from local image files.

    By default, treats an entire directory (or an explicit list of files)
    as one episode ordered by filename. The resulting Example contains the
    eval episode and no context episodes.
    """

    def __init__(
        self,
        *,
        episodes_files: Sequence[Sequence[str]],
        instruction: str = "",
        num_frames: int = 20,
        num_context_episodes: int = 0,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(
            num_frames=num_frames,
            num_context_episodes=num_context_episodes,
            shuffle=shuffle,
            seed=seed,
        )
        if not episodes_files or len(episodes_files) == 0:
            raise ValueError
        # Normalize to absolute Paths at call time to preserve user-specified order
        self.episodes_files: list[list[str]] = [list(ep) for ep in episodes_files]
        self.instruction = instruction or ""

    def _load_images(self, paths: list[Path]):
        images = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning(f"Skipping unreadable image {p}: {exc}")
        return images

    def load_fewshot_input(self, episode_index: int | None = None) -> FewShotInput:
        if episode_index is None:
            episode_index = 0
        if episode_index < 0 or episode_index >= len(self.episodes_files):
            raise IndexError
        # Do not reorder or auto-discover; respect user-provided order strictly.
        paths = [Path(p) for p in self.episodes_files[episode_index]]
        frames = self._load_images(paths)
        if not frames:
            raise ValueError
        ep = self._build_episode(
            frames=frames,
            instruction=self.instruction,
            episode_index=episode_index or 0,
        )
        return FewShotInput(eval_episode=ep, context_episodes=[])
