import os
import json
import math
import shutil
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from .MCTS_Gomoku_NeuralNet import PolicyValueNet

# ------------------------------------------------------------
# 1. 游戏引擎
# ------------------------------------------------------------
class Game:
    def __init__(self, size=8):
        self.last_player = None
        self.current_player = 1
        self.map_size = (size, size)
        self.board = np.zeros((size, size), dtype=np.int32)
        self.last_move = None

    def check_win(self, player):
        if self.last_move is None: return False
        r, c = self.last_move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                nr, nc = r + dr * direction, c + dc * direction
                while 0 <= nr < self.map_size[0] and 0 <= nc < self.map_size[1] and self.board[nr][nc] == player:
                    count += 1
                    nr += dr * direction
                    nc += dc * direction
            if count >= 5: return True
        return False

    def get_possible_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def apply_move(self, move):
        self.board[move[0]][move[1]] = self.current_player
        self.last_player = self.current_player
        self.last_move = move
        self.current_player = 3 - self.current_player
        return self.check_win(self.last_player)

    def clone(self):
        new_game = Game(size=self.map_size[0])
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_player = self.last_player
        new_game.last_move = self.last_move
        return new_game

# ------------------------------------------------------------
# 2. 神经网络引导的 MCTS
# ------------------------------------------------------------
class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior_p=0.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {} 
        self.visits = 0
        self.value = 0.0 
        self.prior_p = prior_p 

    def get_score(self, c_puct):
        # AlphaZero PUCT公式
        q_value = self.value / self.visits if self.visits > 0 else 0
        u_value = c_puct * self.prior_p * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

    def select_best(self, c_puct):
        return max(self.children.items(), key=lambda x: x[1].get_score(c_puct))

def build_model_input_np(game):
    board_size = game.map_size[0]
    state = np.zeros((4, board_size, board_size), dtype=np.float32)
    state[0] = (game.board == game.current_player).astype(np.float32)
    state[1] = (game.board == (3 - game.current_player)).astype(np.float32)
    if game.last_move is not None:
        r, c = game.last_move
        state[2][r][c] = 1.0
    state[3] = (game.current_player - 1.0)
    return state

def mcts_predict(game, model, device):
    state = build_model_input_np(game)    
    input_ts = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        p_logits, v = model(input_ts)
        probs = F.softmax(p_logits, dim=1).cpu().numpy().flatten()
        value = v.item()
    return probs, value

def mcts_search(game, model, device, iters=400, c_puct=1.5, top_k=10, temperature=10):
    root = MCTSNode(game)

    for _ in range(iters):
        # Selection
        node = root
        while node.children:
            move, node = node.select_best(c_puct)
            
        # Expansion
        win = node.game.check_win(node.game.last_player)
        if not win and len(node.game.get_possible_moves()) > 0:
            p_batch, v = mcts_predict(node.game, model, device)
            for move in node.game.get_possible_moves():
                p = p_batch[move[0] * game.map_size[0] + move[1]]
                new_game = node.game.clone()
                new_game.apply_move(move)
                node.children[move] = MCTSNode(new_game, parent=node, move=move, prior_p=p)
        else:
            v = 1.0 if win else 0.5

        # Backpropagation
        v_back = v
        curr_node = node
        while curr_node is not None:
            curr_node.visits += 1
            curr_node.value += v_back
            v_back = 1.0 - v_back
            curr_node = curr_node.parent
    
    # Selection with top-k & temperature sampling
    items = list(root.children.items()) # [(move, node), ...]
    moves = [item[0] for item in items]
    visits = np.array([item[1].visits for item in items], dtype=np.float32)

    # Top-K：只保留访问量前 K 大的动作
    k = min(top_k, len(moves))
    top_k_indices = np.argpartition(visits, -k)[-k:]
    filtered_visits = np.zeros_like(visits)
    filtered_visits[top_k_indices] = visits[top_k_indices]

    # 温度采样处理
    if temperature < 1e-2:
        # 极低温度等同于 Argmax (选访问量绝对最大的)
        best_move = moves[np.argmax(visits)]
    else:
        # 计算概率：P = (visits^(1/temp)) / sum(visits^(1/temp))
        ps = np.power(filtered_visits, 1.0 / temperature)
        ps /= np.sum(ps)
        # 随机采样
        best_idx = np.random.choice(len(moves), p=ps)
        best_move = moves[best_idx]

    # 构造训练数据
    inputs = build_model_input_np(game)
    visit_counts = {f"{m[0]},{m[1]}": int(child.visits) for m, child in root.children.items()}
    value_estimate = float(root.value / root.visits) if root.visits > 0 else 0.5
    state = {
        "player": game.current_player,
        "inputs": inputs.tolist(),
        "visit_counts": visit_counts,
        "value_estimate": value_estimate
    }

    return best_move, state

# ------------------------------------------------------------
# 3. 模型评估器 (Model Evaluator)
# ------------------------------------------------------------
class ModelEvaluator:
    def __init__(self, model_1, model_2, device, board_size=8, iters=400):
        self.models = {1: model_1, 2: model_2}  # 映射玩家编号到模型
        self.device = device
        self.board_size = board_size
        self.iters = iters

    def play_game(self, m1_starts=True):
        """
        进行一场比赛
        m1_starts: True 则 model_1 执黑(1), model_2 执白(2)
                  False 则 model_2 执黑(1), model_1 执白(2)
        """
        game = Game(size=self.board_size)
        # 确定当前对局中，1号玩家用哪个模型，2号玩家用哪个模型
        current_models = {
            1: self.models[1] if m1_starts else self.models[2],
            2: self.models[2] if m1_starts else self.models[1]
        }

        while True:
            active_model = current_models[game.current_player]
            
            # 评估时温度设为 0 (极低)，只选访问量最高的动作，取消 top_k 限制或设大
            best_move, _ = mcts_search(
                game, active_model, self.device, 
                iters=self.iters, temperature=5e-1
            )

            if game.apply_move(best_move):
                winner = game.last_player
                break
            elif not game.get_possible_moves():
                winner = 0 # 平局
                break
        
        # 转换为逻辑上的胜者：返回 1 代表 model_1 赢，2 代表 model_2 赢，0 平局
        if winner == 0: return 0
        if m1_starts:
            return 1 if winner == 1 else 2
        else:
            return 1 if winner == 2 else 2

import time
import multiprocessing as mp

# 单场对战
def _run_single_game(game_idx, model_idx_1, model_idx_2, board_size, iters, m1_starts, device_str):
    start_t = time.time()
    device = torch.device(device_str)
    
    # 加载模型
    path_1 = f"checkpoint/gomoku_policy_value_net_{model_idx_1}.pth"
    model_1 = PolicyValueNet(input_channels=4, num_res_blocks=3, board_size=board_size).to(device)
    model_1.load_state_dict(torch.load(path_1, map_location=device))
    model_1.eval()

    path_2 = f"checkpoint/gomoku_policy_value_net_{model_idx_2}.pth"
    model_2 = PolicyValueNet(input_channels=4, num_res_blocks=3, board_size=board_size).to(device)
    model_2.load_state_dict(torch.load(path_2, map_location=device))
    model_2.eval()

    evaluator = ModelEvaluator(model_1, model_2, device, board_size=board_size, iters=iters)
    
    # 执行对战
    res = evaluator.play_game(m1_starts=m1_starts)
    
    # 准备打印信息
    duration = time.time() - start_t
    m1_color = "Black" if m1_starts else "White"
    
    if res == 1:
        winner_tag = f"M{model_idx_1}"
    elif res == 2:
        winner_tag = f"M{model_idx_2}"
    else:
        winner_tag = "Draw"

    # 打印：[对局号] 胜者 | M1颜色 | 用时
    print(f"[Game {game_idx:3d}] Winner: {winner_tag:7s} | Model_{model_idx_1} was {m1_color:5s} | Time: {duration:5.1f}s", flush=True)
    
    return res

def evaluate(model_idx_1, model_idx_2, num_games=20, iters=400, num_processes=8):
    BOARD_SIZE = 8
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print(f"EVALUATION: Model {model_idx_1} vs Model {model_idx_2}")
    print(f"Settings: {num_games} games, MCTS iters={iters}, Workers={num_processes}")
    print("="*60)

    # 准备参数列表，增加 game_idx (i+1)
    args = []
    for i in range(num_games):
        m1_starts = (i % 2 == 0) # 轮流先手
        args.append((i + 1, model_idx_1, model_idx_2, BOARD_SIZE, iters, m1_starts, device_str))

    # 使用进程池执行
    # 沿用之前自我博弈生成数据阶段的 mp.set_start_method('spawn')
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(_run_single_game, args)

    # 统计
    m1_wins = results.count(1)
    m2_wins = results.count(2)
    draws = results.count(0)

    print("="*60)
    print(f"Evaluation Finished!")
    print(f"Model {model_idx_1} Wins: {m1_wins}")
    print(f"Model {model_idx_2} Wins: {m2_wins}")
    print(f"Draws: {draws}")
    
    win_rate = m1_wins / num_games
    print(f"Model {model_idx_1} Win Rate: {win_rate:.2%}")
    print("="*60)
    
    return win_rate

if __name__ == "__main__":
    # 确保在 Windows 环境下正确运行
    evaluate(model_idx_1=10, model_idx_2=9, num_games=10, iters=800, num_processes=5)