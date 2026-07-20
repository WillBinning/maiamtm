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
from research.mtm.datasets.rustothelloinf import thread_to_strategy, evaluate_move_strategy, target_strategy_from_strats
from research.mtm.utils import create_strat_emb
from research.mtm.models.mtm_model import MTM
from research.mtm.tokenizers.base import Tokenizer, TokenizerManager
from research.mtm.tokenizers.continuous import ContinuousTokenizer
from research.logger import logger
import rust_reversi as rr
from termcolor import colored

EDGE_MASK = np.zeros((8, 8), dtype=np.uint8)

# corners
EDGE_MASK[0,0] = EDGE_MASK[0,7] = EDGE_MASK[7,0] = EDGE_MASK[7,7] = 1

# edges
EDGE_MASK[0,:] = 1
EDGE_MASK[7,:] = 1
EDGE_MASK[:,0] = 1
EDGE_MASK[:,7] = 1

import random
def rollout_strat(path,players,opp_moves):
    start_strat = path[0][1]
    end_strat = path[-1][1]

    if players[0] == 0:
        flipped = end_strat[0]-start_strat[0]
        edge = end_strat[2]-start_strat[2]
    else:
        flipped = end_strat[1]-start_strat[1]
        edge = end_strat[3]-start_strat[3]    

    print(flipped,edge,opp_moves)
    return[flipped.item(),edge.item(),opp_moves]

def strat_cal(board):
        edges_black = np.sum(board.get_board_matrix()[1] * EDGE_MASK) #TODO
        edges_white = np.sum(board.get_board_matrix()[0] * EDGE_MASK)
        black_pieces = board.black_piece_num()
        white_pieces = board.white_piece_num()
        return[black_pieces,white_pieces,edges_black,edges_white]

def AI_turn(batch, traj_length, board, cur_colour,masks,tokenizer_manager,model):
        print("AI player making move....")
        inf_batch = batch.copy() 

        if traj_length != len(batch["states"][0]): #Check that this isn't first run
            board_mat = stack_board(board, cur_colour)
            #Stack a dummy game equal to length of thread - Stack to fit model but everything is masked anyway
            states = {
                "states": board_mat.unsqueeze(0).unsqueeze(0).repeat(1, len(batch["states"][0]), 1)
            }
            inf_batch.pop("states", None) 
            inf_batch.update(states)
        
        inf_batch.pop("boards", None)
        inf_batch.pop("players", None)
        predicted_trajectories = thread_inference( #Inference
            model=model, 
            tokenizer_manager=tokenizer_manager, 
            batch=inf_batch, 
            masks=masks,
            ratio=1
        )

        predicted_move = predicted_trajectories["actions"][0, 0].detach().cpu().numpy()
        print(f"{predicted_move}!!!")
        return(predicted_move)

def AB_turn(board):    
    print("AlphaBeta Player making move...")
    evaluater = rr.PieceEvaluator()
    search = rr.AlphaBetaSearch(evaluater, 3, win_score=1 << 10)
    move =search.get_move(board) 
    print(f"{move}!!!")
    board.do_move(move) #find a do move
    next_str = board.get_board_line()
    return(next_str)

def stack_board(board,colour):
    #This will always return active player as [0] which is what we want so always stack 0 then 1
    board_mat = board.get_board_matrix() 
    active = torch.tensor(board_mat[0])
    opp = torch.tensor(board_mat[1])
    return(torch.stack([active,opp])).flatten().float().to("cuda")



def inference_rollout(model, batch, traj_length, masks, tokenizer_manager, cur_play=None, cur_colour=None, path=[], board_str=None, opp_move=0):
    if cur_colour == None: #Triggers only on first run, collects the needed data from the batch
        board_str = batch["strings"][0][0]
        cur_colour = batch["player"][0][0]
        path.append([batch["strings"][0][0], batch["strats"][0][0]])
    print("==========================================")
    print(f"We have {traj_length -1} plays left.")
    # print(len(batch["states"][0]))

    if cur_colour == 0: #Converts colour to rust reversi turn
        turn = rr.Turn.BLACK
        next_colour = 1
    else:
        turn = rr.Turn.WHITE
        next_colour = 0

    print(f"Turn is {turn}")

    #Sets the board for AlphaBeta
    board = rr.Board()
    board.set_board_str(board_str, turn)



    #All legal moves of current board
    legal = board.get_legal_moves_vec()
    print(legal)
    strats = strat_cal(board)
    
    if cur_play ==1:
        opp_move += len(legal)

    if traj_length == 1 or board.is_game_over():#End condition
        players = [batch["player"][0][0],batch["player"][0][1]] #This finds the first and last player
        strats = rollout_strat(path, players,opp_move)
        print(colored("End of Rollout....", 'green'))
        print("==========================================")
        return strats

    if board.is_pass():
        print("No legal moves available. Passing turn.")
        next_player = abs(cur_play -1)
        next_colour = abs(cur_colour - 1)
        board.do_pass()
        next_str = board.get_board_line()
        strats = strat_cal(board)

        path.append([next_str,strats])

        inference_rollout(model,batch,traj_length-1, masks,tokenizer_manager,cur_play=next_player, path=path, cur_colour=next_colour, board_str=next_str, opp_move=opp_move)


    
    if cur_play == 0: #AI player
        predicted_move = AI_turn(batch, traj_length, board, cur_colour,masks,tokenizer_manager,model)
        
        if predicted_move not in legal: #If model predicts illegal move, return a Null trigger
            path.append(None)
            print(colored("illegal move predicted", 'red'))
            return(None)
        else:#Set AlphaBeta as next player, adds board to path
            next_player = 1
            board.do_move(predicted_move)
            next_str = board.get_board_line()
            path.append([next_str,strats])

    else: #AlphaBeta Player
        print(board)

        next_str = AB_turn(board)
        path.append([next_str,strats])
        next_player = 0
    print("==========================================")
    # print(batch)
    return inference_rollout(model,batch,traj_length-1, masks,tokenizer_manager,cur_play=next_player, path=path, cur_colour=next_colour, board_str=next_str, opp_move=opp_move)

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
    strats = batch["strats"]
    encoded_batch = tokenizer_manager.encode(batch)
    predictions = model.mask_git_forward(encoded_batch, masks, strats, ratio=ratio)
    return tokenizer_manager.decode(predictions)

@torch.inference_mode()
def thread_inference(
    model: MTM,
    tokenizer_manager: TokenizerManager,
    batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    ratio: int = 1
) -> Dict[str, torch.Tensor]:
    model.eval()
    strats = batch["strats"]
    encoded_batch = tokenizer_manager.encode(batch)
    predictions = model.mask_git_forward(encoded_batch, masks, strats, ratio=ratio)
    return tokenizer_manager.decode(predictions)

def main(hydra_cfg: DictConfig):
    device = hydra_cfg.args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    traj_length = hydra_cfg.args.traj_length
    ckpt_path = hydra_cfg.get("ckpt_path", "model_final.pt") 
  
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
    # print(train_dataset[0])
    
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
    legal_predictions = 0
    correct_predictions = 0
    samples_evaluated = 0
    total_strategy_mse = 0.0
    valid_strategy_samples = 0

    logger.info("\n" + "="*50)
    logger.info(f"STARTING EVALUATION OF {len(train_loader)} SAMPLES")
    logger.info("="*50)

    # ==========================================
    # 4. Evaluation Loop with Printing
    # ==========================================
    for idx, batch in enumerate(train_loader):
        if idx >= len(train_loader):
            break
        
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        
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
        # print(batch["states"])

        predicted_strategy = evaluate_move_strategy(
        batch["states"][0],
        "black",
        predicted_move,
        traj_length
        )

        target_strategy = target_strategy_from_strats(
            batch["strats"][0].cpu().numpy()
        )

        # CHANGE THIS BLOCK:
        if predicted_strategy is not None:
            strategy_mse = np.mean(
                (predicted_strategy - target_strategy) ** 2
            )
            # --- ADD THESE ---
            total_strategy_mse += strategy_mse
            valid_strategy_samples += 1
        else:
            # Handle illegal moves gracefully (e.g., assign NaN or a penalty value)
            strategy_mse = float('nan')


        


        # ---> PRINT EVERY GT AND PREDICTION <---
        # ---> PRINT EVERY GT AND PREDICTION <---
        logger.info(f"[Sample {idx + 1}/{len(train_loader)}]")
        logger.info(f"  Legal   : {legal}")
        logger.info(f"Real Move : {actual_move}")
        logger.info(f"  Pred : {np.round(predicted_move, 4)}")
        
        # --- ADD THIS ---
        if np.isnan(strategy_mse):
            logger.info("Strat MSE : NaN (Illegal Move)")
        else:
            logger.info(f"Strat MSE : {strategy_mse:.6f}")

        # Compute metrics
        mse = np.mean((predicted_move - actual_move) ** 2)


        total_mse += mse

        # TODO this changes from legality check 
        if predicted_move in legal:
            is_legal = True
        else:
            is_legal = False
        
        if predicted_move == actual_move:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            correct_predictions +=1
            logger.info("Correct: True")
        else:
            logger.info("Correct: False")


        if is_legal:
            legal_predictions += 1
            logger.info("  Legal: CORRECT ✓")
        else:
            logger.info("  Legal: INCORRECT ✗")
        logger.info("-" * 30)
            
        samples_evaluated += 1

    # ==========================================
    # 5. Output Summary Report
    # ==========================================
    avg_mse = total_mse / samples_evaluated
    legal_percentage = (legal_predictions / samples_evaluated) * 100
    correct_percentage = (correct_predictions / samples_evaluated) * 100


    avg_strategy_mse = (
        total_strategy_mse / valid_strategy_samples 
        if valid_strategy_samples > 0 else float('nan')
    )
    logger.info("\n" + "="*50)
    logger.info("          OVERALL PERFORMANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Trajectories Tested : {samples_evaluated}")
    logger.info(f"Average Next-Move MSE     : {avg_mse:.6f}")

    logger.info(f"Average Strategy MSE      : {avg_strategy_mse:.6f} (over {valid_strategy_samples} valid moves)")
    logger.info(f"OVERALL LEGALITY       : {legal_percentage:.2f}%")
    logger.info(f"OVERALL CORRECTNESS       : {correct_percentage:.2f}%")

    logger.info("="*50)



def thread_main(hydra_cfg: DictConfig):
    device = hydra_cfg.args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    traj_length = hydra_cfg.args.traj_length
    ckpt_path = hydra_cfg.get("ckpt_path", "model_final.pt") 
    
  
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
    # print(train_dataset[0])

    dummy_batch = {
        k: (v if k == "strings" else torch.as_tensor(v).unsqueeze(0).to(device))
        for k, v in train_dataset[0].items()
    }    
    
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
        batch_size=1, #Batch size only works with 1 currently
        num_workers=10,  
        worker_init_fn=worker_init_fn
    )
    
    T_states = traj_length
    T_actions = traj_length - 1  # Define the shorter length

    obs_mask = torch.zeros(T_states, device=device)       
    obs_mask[0] = 1  
    actions_mask = torch.zeros(T_actions, device=device) # Explicitly use T-1
        
    masks = {"states": obs_mask, "actions": actions_mask}

    total_mse = 0.0
    legal_predictions = 0
    correct_predictions = 0
    samples_evaluated = 0
    total_strategy_mse = 0.0
    valid_strategy_samples = 0

    logger.info("\n" + "="*50)
    logger.info(f"STARTING EVALUATION OF {len(train_loader)} SAMPLES")
    logger.info("="*50)

    # ==========================================
    # 4. Evaluation Loop with Printing
    # ==========================================
    for idx, batch in enumerate(train_loader):
        if idx >= len(train_loader):
            break
        
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        
        # for k in batch.keys():
        #     if k not in masks:
        #         masks[k] = torch.zeros(T, device=device)

        current_masks = {
            k: (
                masks[k][:batch[k].shape[1]]
                if k in masks
                else torch.zeros(batch[k].shape[1], device=device)
            )
            for k in batch
            if isinstance(batch[k], torch.Tensor)
        }
        strats = inference_rollout(model, batch, traj_length,masks,tokenizer_manager, cur_play=0)
        target_strategy = target_strategy_from_strats(
            batch["strats"][0].cpu().numpy()
        )
        print(strats)
        print(target_strategy)
        samples_evaluated += 1

        if strats == None:
            continue
        else:
            legal_predictions += 1

        if (strats==target_strategy).all():
            correct_predictions +=1


                # ==========================================
    # 5. Output Summary Report
    # ==========================================
    avg_mse = total_mse / samples_evaluated
    legal_percentage = (legal_predictions / samples_evaluated) * 100
    correct_percentage = (correct_predictions / samples_evaluated) * 100


    avg_strategy_mse = (
        total_strategy_mse / valid_strategy_samples 
        if valid_strategy_samples > 0 else float('nan')
    )
    logger.info("\n" + "="*50)
    logger.info("          OVERALL PERFORMANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total Trajectories Tested : {samples_evaluated}")
    logger.info(f"Average Next-Move MSE     : {avg_mse:.6f}")

    logger.info(f"Average Strategy MSE      : {avg_strategy_mse:.6f} (over {valid_strategy_samples} valid moves)")
    logger.info(f"OVERALL LEGALITY       : {legal_percentage:.2f}%")
    logger.info(f"OVERALL CORRECTNESS       : {correct_percentage:.2f}%")

    logger.info("="*50)


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def configure_jobs(hydra_data: DictConfig) -> None:
    set_seed(hydra_data.get("seed", 42))
    thread_main(hydra_data)

if __name__ == "__main__":
    configure_jobs()