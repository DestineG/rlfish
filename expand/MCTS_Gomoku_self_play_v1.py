import os
import random
import math
import pygame
import os
import shutil
import json

# --- Game Interface: Gomoku ---
class Game:
    def __init__(self, size=8):
        self.last_player = None
        self.current_player = 1
        self.map_size = (size, size)
        self.board = [[0] * self.map_size[1] for _ in range(self.map_size[0])]
        self.last_move = None 

    def check_win(self, player):
        if not self.last_move: return False
        r, c = self.last_move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.map_size[0] and 0 <= nc < self.map_size[1] and self.board[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.map_size[0] and 0 <= nc < self.map_size[1] and self.board[nr][nc] == player:
                count += 1
                nr -= dr
                nc -= dc
            if count >= 5: return True
        return False

    def get_possible_moves(self):
        return [(i, j) for i in range(self.map_size[0]) for j in range(self.map_size[1]) if self.board[i][j] == 0]

    def apply_move(self, move):
        i, j = move
        self.board[i][j] = self.current_player
        self.last_player = self.current_player
        self.last_move = move
        self.current_player = 3 - self.current_player
        return self.check_win(self.last_player)

    def clone(self):
        new_game = Game(size=self.map_size[0])
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.last_player = self.last_player
        new_game.last_move = self.last_move
        return new_game

    def simulate(self):
        temp_game = self.clone()
        if temp_game.last_move and temp_game.check_win(temp_game.last_player):
            return temp_game.last_player
        while True:
            moves = temp_game.get_possible_moves()
            if not moves: return 0
            move = random.choice(moves)
            if temp_game.apply_move(move): 
                return temp_game.last_player

# --- MCTS Data Structure ---
class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.untried_moves = game.get_possible_moves()
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            score = child.value / child.visits + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            choices.append((score, child))
        return max(choices, key=lambda x: x[0])[1]

    def expand(self):
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        new_game = self.game.clone()
        new_game.apply_move(move)
        child = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, result):
        self.visits += 1
        if result == 0: self.value += 0.5
        elif result == self.game.last_player: self.value += 1.0
        if self.parent: self.parent.update(result)

# --- MCTS 算法：返回访问量和胜率值 ---
def mcts_search(game, iters):
    root = MCTSNode(game)
    for _ in range(iters):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        if not node.is_fully_expanded() and not (node.game.last_move and node.game.check_win(node.game.last_player)):
            node = node.expand()
        result = node.game.simulate()
        node.update(result)
    
    # 提取访问量统计
    raw_visits = {child.move: child.visits for child in root.children}
    # 提取平均价值（胜率）统计: Q = Value / Visits
    raw_values = {child.move: (child.value / child.visits) for child in root.children}
    
    best_move = max(root.children, key=lambda c: c.visits).move
    return best_move, raw_visits, raw_values

# --- 可视化 ---
CELL_SIZE = 60
LINE_WIDTH = 2
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
    def __init__(self, game, ai_player=2, data_log_path="gomoku_raw_data.jsonl"):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("MCTS Data Collector (Visits + Values)")
        self.game = game
        self.ai_player = ai_player
        self.iters = 50000
        self.data_log_path = data_log_path
        self.current_game_moves = []

    def draw_board(self):
        self.screen.fill(COLORS["BACKGROUND"])
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, COLORS["LINE"], (CELL_SIZE, (i + 1) * CELL_SIZE), (BOARD_SIZE * CELL_SIZE, (i + 1) * CELL_SIZE), LINE_WIDTH)
            pygame.draw.line(self.screen, COLORS["LINE"], ((i + 1) * CELL_SIZE, CELL_SIZE), ((i + 1) * CELL_SIZE, BOARD_SIZE * CELL_SIZE), LINE_WIDTH)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                center = ((c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE)
                if self.game.board[r][c] == 1:
                    pygame.draw.circle(self.screen, COLORS["PLAYER1"], center, CELL_SIZE // 2 - 5)
                elif self.game.board[r][c] == 2:
                    pygame.draw.circle(self.screen, COLORS["PLAYER2"], center, CELL_SIZE // 2 - 5)
        if self.game.last_move:
            r, c = self.game.last_move
            center = ((c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE)
            pygame.draw.circle(self.screen, COLORS["HIGHLIGHT"], center, 5)

    def record_step(self, board, player, mcts_visits, mcts_values):
        """记录每一步的棋盘、玩家、访问量及价值"""
        serializable_visits = {f"{k[0]},{k[1]}": v for k, v in mcts_visits.items()}
        serializable_values = {f"{k[0]},{k[1]}": round(v, 4) for k, v in mcts_values.items()}
        
        step_snapshot = {
            "board": [row[:] for row in board],
            "player": player,
            "mcts_raw_visits": serializable_visits,
            "mcts_raw_values": serializable_values  # 新增字段
        }
        self.current_game_moves.append(step_snapshot)

    def save_full_game(self, winner):
        game_data = {
            "board_size": BOARD_SIZE,
            "winner": winner,
            "total_moves": len(self.current_game_moves),
            "moves": self.current_game_moves
        }
        with open(self.data_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(game_data) + "\n")
        print(f"Game Saved! Winner: {winner}, Moves: {len(self.current_game_moves)}")
        self.current_game_moves = []

    def run(self):
        running = True
        game_over = False
        while running:
            self.draw_board()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            if not game_over:
                curr_p = self.game.current_player
                print(f"Player-{curr_p} AI searching {self.iters} steps...")
                
                # 获取三个返回值
                move, raw_visits, raw_values = mcts_search(self.game, iters=self.iters)
                
                # 传入 record_step
                self.record_step(self.game.board, curr_p, raw_visits, raw_values)
                
                if self.game.apply_move(move):
                    self.draw_board()
                    pygame.display.flip()
                    self.save_full_game(winner=curr_p)
                    game_over = True
                elif not self.game.get_possible_moves():
                    self.save_full_game(winner=0)
                    game_over = True
            if game_over: running = False


if __name__ == "__main__":
    base_dir = "train_data"
    filename = "gomoku_game.jsonl"
    filepath = os.path.join(base_dir, filename)

    if os.path.exists(base_dir):
        print(f"Directory exists, removing: {base_dir}")
        shutil.rmtree(base_dir)
    
    os.makedirs(base_dir, exist_ok=True)

    BOARD_SIZE = 8
    episodes = 1000
    
    for ep in range(episodes):
        my_game = Game(size=BOARD_SIZE)
        gui = GomokuGUI(my_game, data_log_path=filepath)
        gui.run()