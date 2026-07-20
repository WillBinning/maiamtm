from research.mtm.tokenizers.base import TokenizerManager
from research.mtm.datasets.rustothelloinf import ThreadDataset
from torch.utils.data.dataloader import DataLoader
import torch
from research.mtm.train import RunConfig, Dict
from research.logger import WandBLogger, WandBLoggerConfig, logger, stopwatch
from omegaconf import DictConfig, OmegaConf
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.datasets.base import DatasetProtocol
from research.mtm.tokenizers.continuous import ContinuousTokenizer

import hydra

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    logger.info(hydra_data)
    main(hydra_data)

def main(hydra_cfg):

    cfg: RunConfig = hydra.utils.instantiate(hydra_cfg.args)

    train_dataset, val_dataset = hydra.utils.call(
        hydra_cfg.train_dataset,
        seq_steps=cfg.traj_length,
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=4,
    )

    batch = next(iter(train_loader))

    # print(batch.keys())
    # print(batch)

configure_jobs()