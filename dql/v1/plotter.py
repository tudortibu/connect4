import matplotlib.pyplot as plt
import numpy as np

ID = "stats_smart3"

stats = np.load(ID+".npy")

initial_size = len(stats)

stats = np.array_split(stats, int(len(stats)/50))  # group every 50 items
stats = np.average(stats, axis=1)  # compute average of each group

fig, (loss, win_rate, reward, moves) = plt.subplots(4, 1, figsize=(10,15))

x_axis = np.arange(initial_size*10, step=50*10)  # *10 because each entry is an average of 10 episodes

loss.set_title("Training Loss")
loss.plot(x_axis, stats[:, 0], "magenta")
loss.set_xlabel("Episode")
loss.set_ylabel("Loss")
loss.set_ylim(0, 1000)

# win_rate.set_title("Win vs Opponent Win vs Invalid Move")
win_rate.set_title("Episode Ending Distribution")
win_rate.plot(x_axis, stats[:, 1]*100, "g", label="Agent Win")
win_rate.plot(x_axis, (np.ones(stats.shape[0])-stats[:, 1]-stats[:, 4]-stats[:, 5])*100, "orange", label="Opponent Win")
win_rate.plot(x_axis, stats[:, 5]*100, "b", label="Draw")
win_rate.plot(x_axis, stats[:, 4]*100, "red", label="Invalid Move")
win_rate.set_xlabel("Episode")
win_rate.set_ylabel("% Percent")
win_rate.set_ylim(0, 100)
win_rate.legend()

reward.set_title("Average Reward")
reward.plot(x_axis, stats[:, 2], "b")
reward.set_xlabel("Episode")
reward.set_ylabel("Reward")
reward.set_ylim(-200, 100)

moves.set_title("Average Moves")
moves.plot(x_axis, stats[:, 3], "k")
moves.set_xlabel("Episode")
moves.set_ylabel("Moves")
moves.set_ylim(4, 15)

plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05)
fig.savefig(ID+".png")
