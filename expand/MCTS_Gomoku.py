import random
import math

# --- Game Interface: Gomoku ---
class Game:
    def __init__(self, size=8):
        self.last_player = None
        self.current_player = 1
        self.map_size = (size, size)
        self.board = [[0] * self.map_size[1] for _ in range(self.map_size[0])]
        self.last_move = None # 记录最后一步，优化 check_win 性能

    def check_win(self, player):
        if not self.last_move:
            return False
        
        r, c = self.last_move
        # 四个扫描方向：水平, 垂直, 左下到右上, 左上到右下
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # 向正方向探测
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.map_size[0] and 0 <= nc < self.map_size[1] and self.board[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc
            # 向反方向探测
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.map_size[0] and 0 <= nc < self.map_size[1] and self.board[nr][nc] == player:
                count += 1
                nr -= dr
                nc -= dc
                
            if count >= 5:
                return True
        return False

    def get_possible_moves(self):
        return [(i, j) for i in range(self.map_size[0]) for j in range(self.map_size[1]) if self.board[i][j] == 0]

    def apply_move(self, move):
        i, j = move
        self.board[i][j] = self.current_player
        self.last_player = self.current_player
        self.last_move = move # 更新最后一步坐标
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

        # 如果 expand 阶段已经导致游戏结束，直接返回结果
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
        self.gamma = 0.99 # 折扣因子: 越小越倾向于快速胜利

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            # UCB formula
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
        if self.parent: self.parent.update(result * self.gamma) # 递归更新父节点，应用折扣因子

# --- MCTS Algorithm ---
def mcts_search(game, iters):
    root = MCTSNode(game)
    for _ in range(iters):
        node = root

        # 选择阶段：从根节点开始，选择一个未完全展开的节点
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        
        # 扩展阶段：如果游戏未结束且节点未完全展开，随机选择一个未尝试的动作进行扩展
        if not node.is_fully_expanded() and not node.game.check_win(node.game.last_player):
            node = node.expand()

        # 模拟阶段：从当前节点开始，随机模拟游戏直到结束，记录结果
        result = node.game.simulate()

        # 回传阶段：将模拟结果回传给所有经过的节点
        node.update(result)
    return max(root.children, key=lambda c: c.visits).move

import pygame
import sys

# --- 可视化常量配置 ---
CELL_SIZE = 60
LINE_WIDTH = 2
BOARD_SIZE = 8
SCREEN_SIZE = CELL_SIZE * (BOARD_SIZE + 1)
COLORS = {
    "BACKGROUND": (230, 190, 140), # 经典的木质棋盘色
    "LINE": (0, 0, 0),
    "PLAYER1": (20, 20, 20),      # 黑棋
    "PLAYER2": (240, 240, 240),    # 白棋
    "HIGHLIGHT": (200, 50, 50)     # 最后落子标记
}

class GomokuGUI:
    def __init__(self, game, ai_player=2):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("MCTS Gomoku AI")
        self.game = game
        self.ai_player = ai_player
        self.iters = 40000 # MCTS 迭代次数

    def draw_board(self):
        self.screen.fill(COLORS["BACKGROUND"])
        # 画网格线
        for i in range(BOARD_SIZE):
            # 横线
            pygame.draw.line(self.screen, COLORS["LINE"], 
                             (CELL_SIZE, (i + 1) * CELL_SIZE), 
                             (BOARD_SIZE * CELL_SIZE, (i + 1) * CELL_SIZE), LINE_WIDTH)
            # 纵线
            pygame.draw.line(self.screen, COLORS["LINE"], 
                             ((i + 1) * CELL_SIZE, CELL_SIZE), 
                             ((i + 1) * CELL_SIZE, BOARD_SIZE * CELL_SIZE), LINE_WIDTH)

        # 画棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                center = ((c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE)
                if self.game.board[r][c] == 1:
                    pygame.draw.circle(self.screen, COLORS["PLAYER1"], center, CELL_SIZE // 2 - 5)
                elif self.game.board[r][c] == 2:
                    pygame.draw.circle(self.screen, COLORS["PLAYER2"], center, CELL_SIZE // 2 - 5)
        
        # 标记最后一步
        if self.game.last_move:
            r, c = self.game.last_move
            center = ((c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE)
            pygame.draw.circle(self.screen, COLORS["HIGHLIGHT"], center, 5)

    def run(self):
        running = True
        game_over = False

        while running:
            self.draw_board()
            pygame.display.flip()

            # AI 回合逻辑
            if not game_over and self.game.current_player == self.ai_player:
                print(f"AI is thinking...")
                move = mcts_search(self.game, iters=self.iters)
                if self.game.apply_move(move):
                    print(f"AI Wins!")
                    game_over = True
                elif not self.game.get_possible_moves():
                    print("Draw!")
                    game_over = True

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if not game_over and event.type == pygame.MOUSEBUTTONDOWN and self.game.current_player != self.ai_player:
                    x, y = event.pos
                    # 将点击坐标映射为行列
                    c = round(x / CELL_SIZE) - 1
                    r = round(y / CELL_SIZE) - 1
                    
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.game.board[r][c] == 0:
                        if self.game.apply_move((r, c)):
                            print("Player Wins!")
                            game_over = True
                        elif not self.game.get_possible_moves():
                            print("Draw!")
                            game_over = True

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    my_game = Game(size=BOARD_SIZE)
    gui = GomokuGUI(my_game, ai_player=2)
    gui.run()