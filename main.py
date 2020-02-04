from wrappers import make_env
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    env = make_env("PongNoFrameskip-v4")
    num_games = 20000
    load_checkpoint = False
    best_score = -21
    agent = Agent(alpha=1e-4, gamma=0.99, n_actions=3, action_map={0: 0, 1: 4, 2: 5}, mem_size=25000,
                  batch_size=32, replace=1000, input_dims=(4, 80, 80),
                  epsilon=1.0, epsilon_dec=1e-5, epsilon_min=0.02, load_from_checkpoint=load_checkpoint)

    scores, eps_history = [], []
    n_steps = 0
    render_every = 25
    plot_every = 25
    save_every = 100

    for i in range(num_games):
        score = 0
        observation = env.reset()
        done = False
        render = (i % render_every == 0)
        plot = (i % plot_every == 0) and i > 0
        save = (i % save_every == 0)

        while not done:
            if render:
                env.render()
            action = agent.get_action(observation)
            move = agent.action_map[action]
            new_observation, reward, done, info = env.step(move)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action, reward, new_observation, int(done))
            agent.learn()
            observation = new_observation
        if render:
            env.close()

        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print(f"Games: {i}, Average Score: {avg_score}")

        if avg_score > best_score:
            agent.save_models()
            best_score = avg_score

        if save:
            agent.save_models(extension=i)

        if plot:
            xs = [i for i in range(len(scores[1:]))]
            ys = [np.mean(scores[:i + 1][-100:]) for i in range(len(scores[1:]))]
            plt.plot(xs, ys)
            plt.xlabel("Num Games")
            plt.ylabel("Running average")
            plt.savefig("scores.png")
            plt.clf()


def make_gif(extension):
    agent = Agent(alpha=1e-4, gamma=0.99, n_actions=3, action_map={0: 0, 1: 4, 2: 5}, mem_size=25000,
                  batch_size=32, replace=0, input_dims=(4, 80, 80),
                  epsilon=0.02, epsilon_dec=0, epsilon_min=0, load_from_checkpoint=False)
    agent.load_models(extension=extension)

    frames = []
    done = False
    env = make_env("PongNoFrameskip-v4")
    observation = env.reset()
    i = 0
    while not done:
        if i % 3 == 0:
            frames.append(Image.fromarray(env.render(mode='rgb_array')))
        action = agent.get_action(observation)
        move = agent.action_map[action]
        new_observation, reward, done, info = env.step(move)
        observation = new_observation
        i += 1

    with open(f'{extension}.gif', 'wb') as f:  # change the path if necessary
        im = Image.new('RGB', frames[0].size)
        im.save(f, save_all=True, append_images=frames)


if __name__ == '__main__':
    make_gif(0)
    make_gif(100)
    make_gif(200)
    make_gif(300)
    make_gif(400)
    make_gif(500)
    #main()
