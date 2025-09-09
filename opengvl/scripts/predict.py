from dataclasses import dataclass

import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from opengvl.clients.base import BaseModelClient
from opengvl.data_loaders.base import BaseDataLoader
from opengvl.utils.hydra import ensure_required_keys


@dataclass
class PredictedDataset:
    status: str


def predict_dataset(data_loader: BaseDataLoader):
    # touch the loader to ensure it's configured
    data_loader.reset()
    return {"status": "ok"}


@hydra.main(version_base=None, config_path="../../configs", config_name="experiments/predict")
def main(config: DictConfig) -> None:
    # config validation
    ensure_required_keys(config, "dataset")
    ensure_required_keys(config, "data_loader")
    ensure_required_keys(config, "model")
    ensure_required_keys(config, "prompts")

    # hydra instantiation
    data_loader: BaseDataLoader = instantiate(config.data_loader)
    client: BaseModelClient = instantiate(config.model)
    curated_dataset = predict_dataset(data_loader)
    print(curated_dataset)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    load_dotenv(override=True)
    main()
