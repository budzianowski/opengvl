from loguru import logger
from omegaconf import DictConfig, OmegaConf

HYDRA_TARGET_KEY = "_target_"


def ensure_required_keys(cfg: DictConfig, *required_keys: str) -> None:
    """Validate that cfg contains all required keys under a given base path.

    FewShotInput:
        ensure_required_keys(cfg, "data_loader")

    Raises KeyError with a helpful message when a key is missing.
    """
    for key in required_keys:
        node = OmegaConf.select(cfg, key)
        if node is None:
            raise KeyError(key)
    logger.info(f'Validating config: keys "{", ".join(required_keys)}" are present.')
