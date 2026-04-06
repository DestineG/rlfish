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
    if temperature < 1e-3:
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
# 3. 自我对弈采集器
# ------------------------------------------------------------
class SelfPlayCollector:
    def __init__(self, model, device, board_size=8, iters=400):
        self.model = model
        self.device = device
        self.board_size = board_size
        self.iters = iters

    def run_episode(self):
        game = Game(size=self.board_size)
        game_history = []
        
        while True:
            best_move, state = mcts_search(
                game, self.model,
                self.device, iters=self.iters
            )
            game_history.append(state)

            if game.apply_move(best_move):
                winner = game.last_player
                break
            elif not game.get_possible_moves():
                winner = 0
                break
        
        return {
            "board_size": self.board_size,
            "winner": winner,
            "total_moves": len(game_history),
            "moves": game_history
        }

# ------------------------------------------------------------
# 4. 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    BOARD_SIZE = 8
    MCTS_ITERS = 400
    EPISODES = 100
    DATA_PATH = "train_data/self_play_data.jsonl"
    MODEL_PATH = 'checkpoint/gomoku_policy_value_net.pth'
    
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PolicyValueNet(input_channels=4, num_res_blocks=3, board_size=BOARD_SIZE).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded existing model.")
    model.eval()

    collector = SelfPlayCollector(model, device, board_size=BOARD_SIZE, iters=MCTS_ITERS)

    for ep in range(EPISODES):
        print(f"Episode {ep+1}/{EPISODES} starting...")
        game_result = collector.run_episode()
        
        with open(DATA_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(game_result) + "\n")
        
        print(f"Finished. Winner: {game_result['winner']}, Moves: {game_result['total_moves']}")