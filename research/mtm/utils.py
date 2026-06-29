# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import random
from typing import Optional

import git
import numpy as np
import torch
from omegaconf import OmegaConf

from research.mtm.masks import MaskType

import torch

def create_strat_emb(traj, strats, emb_dim):
    """
    strats shape: [Batch, Time, 5]
    Indices:
      0: black_pieces
      1: white_pieces
      2: black_edge
      3: white_edge
      4: moves
    """
    # Extract features. Shape becomes [Batch, Time]
    black_pieces = strats[:, :, 0]
    white_pieces = strats[:, :, 1]
    black_edge = strats[:, :, 2]
    white_edge = strats[:, :, 3]
    moves = strats[:, :, 4]

    # Calculate strategies across the time dimension (dim=1)
    flipped_amount = calculate_flip_strat(black_pieces)
    edge_amount = calculate_edge_strat(black_edge)
    opp_moves = calculate_move_strat(moves)
    
    # Stack them into a single tensor of shape [Batch, 3]
    # You can now project this in your model using an nn.Linear layer
    strat_features = torch.stack([flipped_amount, edge_amount, opp_moves], dim=-1)
    
    return strat_features

def calculate_flip_strat(black_pieces):
    # Subtract the first timestep from the last timestep for EACH batch item
    flipped_amount = black_pieces[:, -1] - black_pieces[:, 0]
    return flipped_amount

def calculate_edge_strat(black_edge):
    # Subtract the first timestep from the last timestep for EACH batch item
    edge_amount = black_edge[:, -1] - black_edge[:, 0]
    return edge_amount

def calculate_move_strat(moves):
    # Sum every alternating turn (1::2) along the Time dimension (dim=1)
    opp_moves = moves[:, 1::2].sum(dim=1)
    return opp_moves



def load_hydra_path(path):
    hydra_cfg = OmegaConf.load(os.path.join(path, ".hydra/config.yaml"))

    # deal with mask_indicies -> mask_pattern change
    mask_indicies = hydra_cfg.args.mask_indicies
    del hydra_cfg.args.mask_indices
    mask_pattern_names = [member.name for member in MaskType]
    mask_pattern = mask_pattern_names[mask_indicies]
    hydra_cfg.args.mask_patterns = mask_pattern
    return hydra_cfg


def get_ckpt_path_from_folder(folder) -> Optional[str]:
    steps = []
    names = []
    paths_ = os.listdir(folder)
    for name in [os.path.join(folder, n) for n in paths_ if "pt" in n]:
        step = os.path.basename(name).split("_")[-1].split(".")[0]
        steps.append(step)
        names.append(name)

    if len(steps) == 0:
        return None
    else:
        ckpt_path = names[np.argmax(steps)]
        return ckpt_path


def get_cfg_hash(hydra_cfg):
    m = hashlib.md5()
    m.update(OmegaConf.to_yaml(hydra_cfg).encode("utf-8"))
    return m.hexdigest()


def get_git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return str(sha)


def get_git_dirty() -> bool:
    repo = git.Repo(search_parent_directories=True)
    return repo.is_dirty()


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
