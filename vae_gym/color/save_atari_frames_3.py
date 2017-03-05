import gym
import pickle
import Image
import numpy as np

# Extract Y channel from RGB, resize to 84x84 and return
def vae84_3_preprocess(frame) :
	return np.array(Image.fromarray(frame).resize((84,84),Image.BILINEAR))
env = gym.make('Breakout-v0')

num_episodes = 100
num_steps = 1000

frames = []

for episode in range(num_episodes):
    print episode
    obs = env.reset()
    for time_step in range(num_steps):
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        frames.append(vae84_3_preprocess(next_obs))
        if done:
            break

pickle.dump(frames, open('data/frames_3.pkl', 'w'))
#Image.fromarray(toNatureDQNFormat(frames[0])).show()


# If you want to save as images, use Image module or cv2"


