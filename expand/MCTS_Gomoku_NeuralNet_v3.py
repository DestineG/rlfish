import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import sys
import os

# ------------------------------------------------------------
# 1. 模型结构定义 (必须与训练时一致)
# ------------------------------------------------------------
from .MCTS_Gomoku_NeuralNet_train import PolicyValueNet


# ------------------------------------------------------------
# 2. 游戏引擎
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
# 3. 神经网络引导的 MCTS
# ------------------------------------------------------------
class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior_p=0.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {} # {move: MCTSNode}
        self.visits = 0
        self.value = 0.0 # 累计价值 Q
        self.prior_p = prior_p # 网络给出的先验概率 P

    def get_score(self, c_puct):
        # AlphaZero PUCT 公式: Q + U
        q_value = self.value / self.visits if self.visits > 0 else 0
        u_value = c_puct * self.prior_p * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

    def select_best(self, c_puct):
        return max(self.children.items(), key=lambda x: x[1].get_score(c_puct))

def mcts_predict(game, model, device):
    """ 使用模型评估当前局面 """
    board_size = game.map_size[0]
    state = np.zeros((3, board_size, board_size), dtype=np.float32)
    state[0] = (game.board == game.current_player).astype(np.float32)
    state[1] = (game.board == (3 - game.current_player)).astype(np.float32)
    state[2] = game.current_player
    
    input_ts = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        p_logits, v = model(input_ts)
        probs = F.softmax(p_logits, dim=1).cpu().numpy().flatten()
        value = v.item()
    return probs, value

def mcts_search(game, model, device, iters=800, c_puct=1.5):
    root = MCTSNode(game)
    
    for _ in range(iters):
        node = root
        # 1. Selection
        while node.children:
            move, node = node.select_best(c_puct)
            
        # 2. Expansion & Evaluation
        win = node.game.check_win(node.game.last_player)
        if not win and len(node.game.get_possible_moves()) > 0:
            probs, v = mcts_predict(node.game, model, device)
            # 展开所有合法子节点
            for move in node.game.get_possible_moves():
                p = probs[move[0] * game.map_size[0] + move[1]]
                new_game = node.game.clone()
                new_game.apply_move(move)
                node.children[move] = MCTSNode(new_game, parent=node, move=move, prior_p=p)
        else:
            # 终止状态评估
            if win: v = 1.0 # 上一手走棋的人赢了
            else: v = 0.5   # 平局

        # 3. Backpropagation
        # 注意：每一层的胜率对于父节点来说是相反的
        # 由于我们存储的是相对当前玩家的 v (0~1)，需要映射到回溯逻辑
        v_back = v
        while node is not None:
            node.visits += 1
            node.value += v_back
            v_back = 1.0 - v_back # 对手视角的胜率
            node = node.parent
            
    # 返回访问量最高的动作
    return max(root.children.items(), key=lambda x: x[1].visits)[0]

# ------------------------------------------------------------
# 4. GUI 与 主程序
# ------------------------------------------------------------
CELL_SIZE = 60
BOARD_SIZE = 8
SCREEN_SIZE = CELL_SIZE * (BOARD_SIZE + 1)
COLORS = {
    "BACKGROUND": (230, 190, 140),
    "LINE": (0, 0, 0),
    "PLAYER1": (20, 20, 20),
    "PLAYER2": (240, 240, 240),
    "HIGHLIGHT": (200, 50, 50)
}

class GomokuGUI:
    def __init__(self, game, ai_player=2):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("AlphaZero Gomoku")
        self.game = game
        self.ai_player = ai_player
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet(num_res_blocks=3, board_size=BOARD_SIZE).to(self.device)
        model_path = 'checkpoint/gomoku_policy_value_net.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded trained model successfully.")
        else:
            print("Warning: Model path not found. AI will be untrained.")
        self.model.eval()
        self.iters = 1000

    def draw(self):
        self.screen.fill(COLORS["BACKGROUND"])
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, COLORS["LINE"], (CELL_SIZE, (i+1)*CELL_SIZE), (BOARD_SIZE*CELL_SIZE, (i+1)*CELL_SIZE), 2)
            pygame.draw.line(self.screen, COLORS["LINE"], ((i+1)*CELL_SIZE, CELL_SIZE), ((i+1)*CELL_SIZE, BOARD_SIZE*CELL_SIZE), 2)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.game.board[r, c] > 0:
                    color = COLORS["PLAYER1"] if self.game.board[r, c] == 1 else COLORS["PLAYER2"]
                    pygame.draw.circle(self.screen, color, ((c+1)*CELL_SIZE, (r+1)*CELL_SIZE), CELL_SIZE//2 - 5)
        if self.game.last_move:
            r, c = self.game.last_move
            pygame.draw.circle(self.screen, COLORS["HIGHLIGHT"], ((c+1)*CELL_SIZE, (r+1)*CELL_SIZE), 6)

    def run(self):
        game_over = False
        while True:
            self.draw()
            pygame.display.flip()

            if not game_over and self.game.current_player == self.ai_player:
                print("AI is thinking...")
                move = mcts_search(self.game, self.model, self.device, iters=self.iters)
                if self.game.apply_move(move):
                    print("AI Wins!")
                    game_over = True
                elif not self.game.get_possible_moves():
                    print("Draw!")
                    game_over = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if not game_over and event.type == pygame.MOUSEBUTTONDOWN and self.game.current_player != self.ai_player:
                    c, r = round(event.pos[0]/CELL_SIZE)-1, round(event.pos[1]/CELL_SIZE)-1
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.game.board[r, c] == 0:
                        if self.game.apply_move((r, c)):
                            print("Player Wins!")
                            game_over = True

if __name__ == "__main__":
    gui = GomokuGUI(Game(BOARD_SIZE))
    gui.run()