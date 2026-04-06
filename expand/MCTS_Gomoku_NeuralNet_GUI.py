import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import pygame
import sys

# ------------------------------------------------------------
# 1. 模型结构导入
# ------------------------------------------------------------
from .MCTS_Gomoku_NeuralNet import PolicyValueNet

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
        self.children = {} 
        self.visits = 0
        self.value = 0.0 
        self.prior_p = prior_p 

    def get_score(self, c_puct):
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

def mcts_search(game, model, device, iters=400, c_puct=1.5, top_k=10, temperature=0):
    root = MCTSNode(game)

    for _ in range(iters):
        node = root
        while node.children:
            move, node = node.select_best(c_puct)
            
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

        v_back = v
        curr_node = node
        while curr_node is not None:
            curr_node.visits += 1
            curr_node.value += v_back
            v_back = 1.0 - v_back
            curr_node = curr_node.parent
    
    items = list(root.children.items())
    moves = [item[0] for item in items]
    visits = np.array([item[1].visits for item in items], dtype=np.float32)

    # 人机对战模式：通常使用 temperature=0，即直接选访问量最大的
    if temperature < 1e-3:
        best_move = moves[np.argmax(visits)]
    else:
        # 训练/探索模式
        ps = np.power(visits, 1.0 / temperature)
        ps /= np.sum(ps)
        best_idx = np.random.choice(len(moves), p=ps)
        best_move = moves[best_idx]

    return best_move

# ------------------------------------------------------------
# 4. GUI 适配
# ------------------------------------------------------------
CELL_SIZE = 60
BOARD_SIZE = 8
SCREEN_SIZE = CELL_SIZE * (BOARD_SIZE + 1)
COLORS = {
    "BACKGROUND": (230, 190, 140),
    "LINE": (0, 0, 0),
    "PLAYER1": (20, 20, 20),
    "PLAYER2": (240, 240, 240),
    "HIGHLIGHT": (200, 50, 50),
    "TEXT": (50, 50, 50)
}

class GomokuGUI:
    def __init__(self, game, ai_player=2):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("AlphaZero Gomoku 人机对战")
        self.font = pygame.font.SysFont("arial", 30)
        self.game = game
        self.ai_player = ai_player
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # 注意：此处 input_channels 需与你的 4 通道输入一致
        self.model = PolicyValueNet(input_channels=4, num_res_blocks=3, board_size=BOARD_SIZE).to(self.device)
        
        MODEL_PATH = 'checkpoint/gomoku_policy_value_net.pth'
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("Loaded trained model successfully.")
        else:
            print("Warning: No checkpoint found. AI will play randomly!")
        
        self.model.eval()
        self.iters = 800  # AI思考深度

    def draw(self, status_text=""):
        self.screen.fill(COLORS["BACKGROUND"])
        # 画棋盘线
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, COLORS["LINE"], 
                             (CELL_SIZE, (i+1)*CELL_SIZE), 
                             (BOARD_SIZE*CELL_SIZE, (i+1)*CELL_SIZE), 2)
            pygame.draw.line(self.screen, COLORS["LINE"], 
                             ((i+1)*CELL_SIZE, CELL_SIZE), 
                             ((i+1)*CELL_SIZE, BOARD_SIZE*CELL_SIZE), 2)
        
        # 画棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.game.board[r, c] > 0:
                    color = COLORS["PLAYER1"] if self.game.board[r, c] == 1 else COLORS["PLAYER2"]
                    pygame.draw.circle(self.screen, color, 
                                       ((c+1)*CELL_SIZE, (r+1)*CELL_SIZE), 
                                       CELL_SIZE//2 - 5)
        
        # 标记最后一手
        if self.game.last_move:
            r, c = self.game.last_move
            pygame.draw.circle(self.screen, COLORS["HIGHLIGHT"], 
                               ((c+1)*CELL_SIZE, (r+1)*CELL_SIZE), 6)
        
        # 显示状态文字
        if status_text:
            text_surf = self.font.render(status_text, True, COLORS["TEXT"])
            self.screen.blit(text_surf, (20, 10))

    def run(self):
        game_over = False
        winner_msg = ""
        
        while True:
            status = winner_msg if game_over else ("AI Thinking..." if self.game.current_player == self.ai_player else "Your Turn")
            self.draw(status)
            pygame.display.flip()

            # AI 回合
            if not game_over and self.game.current_player == self.ai_player:
                # 增加延时感，避免秒下看不清
                pygame.time.delay(500)
                move = mcts_search(self.game, self.model, self.device, iters=self.iters, temperature=0)
                if self.game.apply_move(move):
                    winner_msg = "AI (White) Wins!"
                    game_over = True
                elif not self.game.get_possible_moves():
                    winner_msg = "Draw Game!"
                    game_over = True

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # 玩家点击
                if not game_over and event.type == pygame.MOUSEBUTTONDOWN and self.game.current_player != self.ai_player:
                    # 坐标转换：pygame(x, y) -> 棋盘(col, row)
                    mx, my = event.pos
                    c = round(mx / CELL_SIZE) - 1
                    r = round(my / CELL_SIZE) - 1
                    
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.game.board[r, c] == 0:
                        if self.game.apply_move((r, c)):
                            winner_msg = "You (Black) Win!"
                            game_over = True
                        elif not self.game.get_possible_moves():
                            winner_msg = "Draw Game!"
                            game_over = True

if __name__ == "__main__":
    # 初始化 8x8 棋盘，玩家先手(1)，AI后手(2)
    gui = GomokuGUI(Game(BOARD_SIZE), ai_player=2)
    gui.run()