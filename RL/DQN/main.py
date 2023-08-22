from time import sleep
from environments.flappy_boi import FlappyBoïEnv
from model import Agent, ModelConfig

MAX_H = 10000

config = ModelConfig(INPUT_SHAPE=FlappyBoïEnv.INPUT_SHAPE, ACTION_SPACE=FlappyBoïEnv.ACTION_SPACE, TRAIN_EACH=4, LR=5e-4, TAU=5e-4, BATCH_SIZE=64)

agent = Agent(config)

env = FlappyBoïEnv()

if __name__ == "__main__":
    TRAINING_STEPS = 30000
    score_history = []
    mean_score_history = []
    play_episode = False
    for i in range(TRAINING_STEPS):
        ratio = i / TRAINING_STEPS
        if ratio < 0.05:
            greed = 0.95
        else:
            greed = 1.0
        
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, greed)
            _, _, reward, next_state, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if play_episode or reward > 5:
                env.render()
        
        play_episode = False

        score_history.append(env.score)

        if i % 200 == 0:
            mean_score = sum(score_history) / len(score_history)
            mean_score_history.append(mean_score)
            print(f"Epoch {int(i / 200)} / {int(TRAINING_STEPS/200)} ; Greed = {greed} ; Mean score = {mean_score}")
            score_history = []
            play_episode = True

    print("Training done")
    # Plot
    import matplotlib.pyplot as plt
    plt.plot(mean_score_history)
    plt.show()

    # Save model
    import torch
    torch.save(agent.Q_o.state_dict(), "model.pt")

    while True:
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, 0.0)
            _, _, reward, next_state, done = env.step(action)
            state = next_state
            env.render()
        print(env.score)
        sleep(1)
