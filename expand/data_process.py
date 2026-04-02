import json
import numpy as np

def load_gomoku_data(file_path):
    data_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            game = json.loads(line)
            print(f"Processing Game: Winner={game['winner']}, Total Moves={game['total_moves']}")
            for move in game['moves']:
                print(sum(move["mcts_raw_visits"].values()))
                break
            break
            
    return data_pairs

# 使用示例
samples = load_gomoku_data('train_data\gomoku_game.jsonl')
print(f"加载了 {len(samples)} 个状态样本")