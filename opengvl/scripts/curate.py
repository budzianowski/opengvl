from typing import TypeAlias

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.hydra import ensure_required_keys

CuratedDataset: TypeAlias = dict


def curate_dataset(data_loader: BaseDataLoader) -> CuratedDataset:
    # touch the loader to ensure it's configured
    data_loader.reset()
    return {"status": "ok"}


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/curate")
def main(config: DictConfig) -> None:
    # config validation
    ensure_required_keys(config, "dataset")
    ensure_required_keys(config, "data_loader")
    ensure_required_keys(config, "model")

    # hydra instantiation
    data_loader: BaseDataLoader = instantiate(config.data_loader)
    curated_dataset: CuratedDataset = curate_dataset(data_loader)
    print(curated_dataset)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
