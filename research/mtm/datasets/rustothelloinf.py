import torch
import numpy as np
import ast
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
from torch.utils.data import Dataset
import rust_reversi as rr
import random

EDGE_MASK = np.zeros((8, 8), dtype=np.uint8)

# corners
EDGE_MASK[0,0] = EDGE_MASK[0,7] = EDGE_MASK[7,0] = EDGE_MASK[7,7] = 1

# edges
EDGE_MASK[0,:] = 1
EDGE_MASK[7,:] = 1
EDGE_MASK[:,0] = 1
EDGE_MASK[:,7] = 1

def thread_to_strategy(thread):

    black_pieces = []
    black_edges = []
    moves = []

    for state in thread:

        black_pieces.append(
            state["black_pieces"].item()
        )

        black_edges.append(
            state["black_edge"].item()
        )

        moves.append(
            state["moves"].item()
        )

    flip_strat = (
        black_pieces[-1]
        - black_pieces[0]
    )

    edge_strat = (
        black_edges[-1]
        - black_edges[0]
    )

    move_strat = sum(
        moves[1::2]
    )

    return np.array([
        flip_strat,
        edge_strat,
        move_strat
    ], dtype=np.float32)

def evaluate_move_strategy(
    states_tensor,
    player,
    move,
    depth
):
    board = rr.Board()
    print(board.get_board_matrix())

    # Catch both "b" and "black" properly
    turn = (
        rr.Turn.BLACK
        if player in ["b", "black"]
        else rr.Turn.WHITE
    )

    # 1. Grab the current state (first step in the trajectory)
    current_state = states_tensor[0] 
    
    # 2. Unpack the 128-element tensor into white and black halves
    white_mask = current_state[:64]
    black_mask = current_state[64:]

    # 3. Reconstruct the 64-character board string
    board_chars = []
    for i in range(64):
        if white_mask[i] > 0.5:
            board_chars.append('O')
        elif black_mask[i] > 0.5:
            board_chars.append('X')
        else:
            board_chars.append('-')
            
    reconstructed_board_str = "".join(board_chars)
    # print(reconstructed_board_str)

    board.set_board_str(
        reconstructed_board_str,
        turn
    )

    legal_moves = board.get_legal_moves_vec()

    if move not in legal_moves:
        return None

    move_idx = legal_moves.index(move)

    child_board = board.get_child_boards()[move_idx]

    threads = shuffle_moves(
        child_board,
        depth - 1,
        child_board.get_board()[2],
        max_threads=1000000
    )

    strategies = []

    for thread in threads:
        strategies.append(
            thread_to_strategy(thread)
        )

    return np.mean(
        np.stack(strategies),
        axis=0
    )

def target_strategy_from_strats(strats):

    black_pieces = strats[:, 0]
    black_edges = strats[:, 2]
    moves = strats[:, 4]

    return np.array([
        black_pieces[-1] - black_pieces[0],
        black_edges[-1] - black_edges[0],
        moves[1::2].sum()
    ], dtype=np.float32)

def shuffle_moves(parent_board, k, start_play, path=None, results=None, max_threads=1):
    # print("Current k: ", k)

    # shared results list across recursion
    if results is None:
        results = []

    # hard cap
    if len(results) >= max_threads:
        return results

    if path is None:
        edges_black = np.sum(parent_board.get_board_matrix()[1] * EDGE_MASK)
        edges_white = np.sum(parent_board.get_board_matrix()[0] * EDGE_MASK)
        player = 0 if start_play == rr.Turn.BLACK else 1

        path = [{
            "white": torch.tensor(parent_board.get_board_matrix()[abs(player-1)]),
            "black": torch.tensor(parent_board.get_board_matrix()[player]),
            "player": torch.tensor(player),
            "move": torch.tensor(65),
            "black_pieces": torch.tensor(parent_board.black_piece_num()),
            "white_pieces": torch.tensor(parent_board.white_piece_num()),
            "black_edge": torch.tensor(edges_black),
            "white_edge": torch.tensor(edges_white),
            "moves": torch.tensor(len(parent_board.get_legal_moves_vec()))
        }]

    # terminal depth
    if k == 0:
        if len(results) < max_threads:
            results.append(path)
        return results

    parent_moves = parent_board.get_legal_moves_vec()
    child_boards = parent_board.get_child_boards()

    # no legal children
    if not child_boards:
        if len(results) < max_threads:
            results.append(path)
        return results
    zipped = list(zip(child_boards,parent_moves))
    random.shuffle(zipped) #Shuffles the list to prevent bias towards lower index moves
    child_boards, parent_moves = zip(*zipped)
    child_num = 0

    for child_board in child_boards:

        # stop immediately if capped
        if len(results) >= max_threads:
            break

        edges_black = np.sum(child_board.get_board_matrix()[1] * EDGE_MASK)
        edges_white = np.sum(child_board.get_board_matrix()[0] * EDGE_MASK)
        player = 0 if child_board.get_board()[2] == rr.Turn.BLACK else 1

        entry = {
            "white": torch.tensor(child_board.get_board_matrix()[abs(player-1)]),
            "black": torch.tensor(child_board.get_board_matrix()[player]),
            "player": torch.tensor(player),
            "move": torch.tensor(parent_moves[child_num]),
            "black_pieces": torch.tensor(child_board.black_piece_num()),
            "white_pieces": torch.tensor(child_board.white_piece_num()),
            "black_edge": torch.tensor(edges_black),
            "white_edge": torch.tensor(edges_white),
            "moves": torch.tensor(len(child_board.get_legal_moves_vec()))
        }

        new_path = path + [entry]
        child_num += 1

        shuffle_moves(
            child_board,
            k - 1,
            start_play=start_play,
            path=new_path,
            results=results,
            max_threads=max_threads
        )

    return results

def read_boards(board_path):
    board_ret = []
    player_ret = []
    with open(board_path,'r') as file:
        for each in file:
            # print(each)
            board = each.strip("\n").rstrip("w").rstrip("b")
            player =''
            if "b" in each:
                player = "b"
            else:
                 player = "w"        
            board_ret.append(board)
            player_ret.append(player)

    return board_ret, player_ret

import torch
from torch.utils.data import Dataset

class ThreadDataset(Dataset):
    # Change depth to seq_steps to align with train.py call signature
    def __init__(self, data_path, seq_steps, paths_list=None, type="train"):
        self.data_path = data_path
        self.depth = seq_steps
        self.paths = []
        self.type = type
        # If we are cloning for a split, skip parsing raw files again
        if paths_list is not None:
            self.paths = paths_list
            return

        data, players = read_boards(data_path)
        self.legal_moves = []
        for state, value in zip(data, players):
            board = rr.Board()
            player = rr.Turn.BLACK if value == 'b' else rr.Turn.WHITE
            board.set_board_str(state, player)
            current_legal = board.get_legal_moves_vec()
            
            threads = shuffle_moves(board, self.depth, player, max_threads=2)
            
            for path in threads:
                self.paths.append(
                    self.process_path(
                        path,
                        current_legal,
                        board_str=state,
                        player=value
                    )
                )
    def padding_check(self, boards, moves, strats):
        if len(boards) >= self.depth:
            return boards, moves, strats

        boards.append(torch.zeros(128))
        moves.append(torch.tensor(65))
        strats.append(torch.zeros(5)) 

        return self.padding_check(boards, moves, strats)
    def process_path(self, path, legal, board_str=None, player=None):
        boards = []
        moves = []
        strats = []

        for state in path[:self.depth]:

            white = state["white"]
            black = state["black"]

            board = torch.stack([white, black]).flatten().float()
            boards.append(board)

            if state["move"] != 65:
                moves.append(state["move"].long())

            current_strat = torch.stack([
                state["black_pieces"].float(),
                state["white_pieces"].float(),
                state["black_edge"].float(),
                state["white_edge"].float(),
                state["moves"].float()
            ])

            strats.append(current_strat)

        boards, moves, strats = self.padding_check(
            boards,
            moves,
            strats
        )

        if self.type == "train":
            return {
                "states": torch.stack(boards),
                "actions": torch.stack(moves),
                "strats" : torch.stack(strats)
            }

        if self.type == "eval":
            return {
                "states": torch.stack(boards),
                "actions": torch.stack(moves),
                "legal": torch.tensor(legal),
                "strats": torch.stack(strats),

                # Needed for inference rollout
                "board_str": board_str,
                "player": player
            }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]

    @classmethod
    def create_splits(cls, data_path, seq_steps, split_ratio=0.5, set_type="train"):
        """
        Factory method to load data once and split it into Train and Validation datasets,
        matching the tuple return expectation of train.py
        """
        # print(set_type)
        full_dataset = cls(data_path=data_path, seq_steps=seq_steps,type=set_type)
        
        train_size = int(len(full_dataset) * split_ratio)
        train_paths = full_dataset.paths[:train_size]
        val_paths = full_dataset.paths[train_size:]
        
        train_dataset = cls(data_path=data_path, seq_steps=seq_steps, paths_list=train_paths, type=set_type)
        val_dataset = cls(data_path=data_path, seq_steps=seq_steps, paths_list=val_paths, type=set_type)
        
        return train_dataset, val_dataset
    def eval_logs(self, model, tokenizer_manager):
        # MTM expects this method to return a dictionary of metrics
        return {}
    def trajectory_statistics(self):
            """
            Provides mean and std as tensor attributes for the ContinuousTokenizer.
            Using tensors of 0s and 1s ensures binary board states remain untouched
            and prevents array-indexing crashes in the tokenizer.
            """
            class DummyStats:
                def __init__(self):
                    # 128 matches your flattened board state size
                    self.mean = torch.zeros(128)
                    self.std = torch.ones(128)
                    
            return {
                "states": DummyStats()
            }