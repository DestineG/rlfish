# pipeline.py

from .MCTS_Gomoku_NeuralNet_CLI_self_play import main as self_play_main
from .MCTS_Gomoku_NeuralNet_train import train as train_main
from .MCTS_Gomoku_NeuralNet_CLI_self_play_test import evaluate

total_iterations = 10
# EPISODES = 10
# MCTS_ITERS = 20
EPISODES = 50
MCTS_ITERS = 1000
num_epochs = 200

def run_pipeline():
    """
    total_iterations: 整体大循环次数
    每次循环：
    1. 基于当前模型(model_i)进行自我对弈，生成数据(data_i)
    2. 基于数据(data_i)训练，产生新模型(model_{i+1})
    """

    for i in range(total_iterations):
        print(f"\n{'='*20} Starting Pipeline Iteration {i}/{total_iterations} {'='*20}")
        
        # 自我对弈阶段
        # 产生的数据索引和当前模型索引一致
        print(f"--- Self-Play Stage (Model {i} -> Data {i}) ---")
        self_play_main(
            model_index=i, data_index=i,
            EPISODES=EPISODES - int((0.9*EPISODES)*(i/total_iterations)),
            MCTS_ITERS=MCTS_ITERS - int((0.5*MCTS_ITERS)*(i/total_iterations))
        )
        
        # 训练阶段
        # 读取刚产生的数据 data_i，训练出下一个模型 model_{i+1}
        print(f"--- Training Stage (Data {i} -> Model {i+1}) ---")
        train_main(model_index=i+1, data_index=i, num_epochs=num_epochs)

        # 测试阶段
        # 评估新模型 model_{i+1} 的性能
        print(f"--- Evaluation Stage (Model {i+1}) ---")
        if i > 0:  # 从第二轮开始评估新模型与上一轮模型的对战表现
            win_rate = evaluate(model_idx_1=i+1, model_idx_2=i, num_games=50)
            print(f"Model {i+1} vs Model {i} Win Rate: {win_rate:.2f}")

if __name__ == "__main__":
    run_pipeline()