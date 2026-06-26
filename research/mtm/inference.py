# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# HYDRA_FULL_ERROR=1 python research/mtm/inference.py +ckpt_path=/home/will/projects/mtm/outputs/mtm_mae/2026-06-08_13-44-10/old/model_1500.p

"""
Inference script to evaluate next-move correctness and print every GT vs Prediction.
"""
import os
import torch
import torch.nn.functional as F
import hydra
import numpy as np
from omegaconf import DictConfig
from typing import Dict, Any
from torch.utils.data.dataloader import DataLoader

from research.mtm.models.mtm_model import MTM
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.logger import logger
import rust_reversi as rr

import random


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def set_seed(seed: int):  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional (strongest determinism, may error for some ops)
    torch.use_deterministic_algorithms(True)

@torch.inference_mode()
def run_inference(
    model: MTM,
    tokenizer_manager: TokenizerManager,
    batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    ratio: int = 1
) -> Dict[str, torch.Tensor]:
    model.eval()
    encoded_batch = tokenizer_manager.encode(batch)
    predictions = model.mask_git_forward(encoded_batch, masks, ratio=ratio)
    return tokenizer_manager.decode(predictions)

def main(hydra_cfg: DictConfig):
    device = hydra_cfg.args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    traj_length = hydra_cfg.args.traj_length
    ckpt_path = hydra_cfg.get("ckpt_path", "model_final.pt") 
    
    # --- CONFIGURATION FOR OVERALL EVALUATION ---
    TOLERANCE = 0.1          # Maximum absolute error per action dimension to count as "correct"
    # --------------------------------------------

    logger.info(f"Loading checkpoint from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ==========================================
    # 1. Setup Dataset & Tokenizers
    # ==========================================
    train_dataset, _ = hydra.utils.call(hydra_cfg.eval_dataset, seq_steps=traj_length, set_type="eval")
    
    logger.info(f"Loaded Training Dataset with {len(train_dataset)} items.")
    
    if "tokenizers" in hydra_cfg:
        tokenizers: Dict[str, Tokenizer] = {
            k: hydra.utils.call(v, key=k, train_dataset=train_dataset)
            for k, v in hydra_cfg.tokenizers.items()
        }
    else:
        tokenizers: Dict[str, Tokenizer] = {
            k: ContinuousTokenizer.create(k, train_dataset)
            for k in train_dataset[0].keys()
        }
    tokenizer_manager = TokenizerManager(tokenizers).to(device)
    print(train_dataset[0])

    dummy_batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in train_dataset[0].items()}
    data_shapes = {k: v.shape[-2:] for k, v in tokenizer_manager.encode(dummy_batch).items()}

    # ==========================================
    # 2. Initialize and Load Model
    # ==========================================
    model_config = hydra.utils.instantiate(hydra_cfg.model_config)
    model = model_config.create(data_shapes, traj_length)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    logger.info(f"Successfully loaded model from step {ckpt.get('step', 'unknown')}")

    # ==========================================
    # 3. Setup Sequential Data Loader
    # ==========================================
    # print(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=10,  # keep deterministic (good choice already)
        worker_init_fn=worker_init_fn
    )
    
    T_states = traj_length
    T_actions = traj_length - 1  # Define the shorter length

    obs_mask = torch.zeros(T_states, device=device)       
    obs_mask[0] = 1  
    actions_mask = torch.zeros(T_actions, device=device) # Explicitly use T-1
        
    masks = {"states": obs_mask, "actions": actions_mask}

    total_mse = 0.0
    correct_predictions = 0
    samples_evaluated = 0

    logger.info("\n" + "="*50)
    logger.info(f"STARTING EVALUATION OF {len(train_loader)} SAMPLES")
    logger.info("="*50)

    # ==========================================
    # 4. Evaluation Loop with Printing
    # ==========================================
    for idx, batch in enumerate(train_loader):
        if idx >= len(train_loader):
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # for k in batch.keys():
        #     if k not in masks:
        #         masks[k] = torch.zeros(T, device=device)

        current_masks = {
            k: masks[k][:batch[k].shape[1]] if k in masks else torch.zeros(batch[k].shape[1], device=device)
            for k in batch.keys()
        }

        predicted_trajectories = run_inference(
            model=model, 
            tokenizer_manager=tokenizer_manager, 
            batch=batch, 
            masks=current_masks,
            ratio=1
        )



        legal = batch["legal"][0].detach().cpu().numpy()

        # Isolate the immediate next move (Action at step 0)
        actual_move = batch["actions"][0, 0].detach().cpu().numpy()
        predicted_move = predicted_trajectories["actions"][0, 0].detach().cpu().numpy()


        # print(predicted_trajectories["actions"][0, 0])
        # print(predicted_move)



        


        # ---> PRINT EVERY GT AND PREDICTION <---
        logger.info(f"[Sample {idx + 1}/{len(train_loader)}]")
        logger.info(f"  GT   : {legal}")
        logger.info(f"  Pred : {np.round(predicted_move, 4)}")

        # Compute metrics
        mse = np.mean((predicted_move - actual_move) ** 2)


        total_mse += mse

        # TODO this changes from legality check 
        if predicted_move in legal:
            is_correct = True
        else:
            is_correct = False

        # is_correct = np.all(np.abs(predicted_move - actual_move) <= TOLERANCE)

        if is_correct:
            correct_predictions += 1
            logger.info("  Status: CORRECT ✓")
        else:
            logger.info("  Status: INCORRECT ✗")
        logger.info("-" * 30)
            
        samples_evaluated += 1

    # ==========================================
    # 5. Output Summary Report
    # ==========================================
    avg_mse = total_mse / samples_evaluated
    accuracy_percentage = (correct_predictions / samples_evaluated) * 100

    logger.info("\n" + "="*50)
    logger.info("          OVERALL PERFORMANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Trajectories Tested : {samples_evaluated}")
    logger.info(f"Evaluation Tolerance Check: ±{TOLERANCE}")
    logger.info(f"Average Next-Move MSE     : {avg_mse:.6f}")
    logger.info(f"OVERALL CORRECTNESS       : {accuracy_percentage:.2f}%")
    logger.info("="*50)

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    set_seed(hydra_data.get("seed", 42))
    main(hydra_data)

if __name__ == "__main__":
    configure_jobs()