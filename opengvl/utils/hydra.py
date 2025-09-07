from loguru import logger
from omegaconf import DictConfig, OmegaConf

HYDRA_TARGET_KEY = "_target_"


def ensure_required_keys(cfg: DictConfig, base: str) -> None:
    """Validate that cfg contains all required keys under a given base path.

    Example:
        ensure_required_keys(cfg, "data_loader")

    Raises KeyError with a helpful message when a key is missing.
    """
    node = OmegaConf.select(cfg, base)
    if node is None:
        raise KeyError(base)
    logger.info(f'Validating config: key "{base}" is present.')
