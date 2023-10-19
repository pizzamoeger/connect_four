import matplotlib.pyplot as plt

dir = "MCTS/simulations/"

f = open(dir+"RANKING.txt", "r")
lines = f.readlines()
f.close()

rand_rollout = []
not_rand_rollout = []

for i in range(len(lines)):
    lines[i] = lines[i].split(" ")
    elo = int(lines[i][2].strip("\n"))
    params = lines[i][1].replace(dir, "", 1).strip(".txt:").split("_")

    if params[0] == "r":
        num = int(params[1])+int(params[2])+int(params[3]) - 40
        rand_rollout.append([num, elo])
    else:
        num = int(params[0])+int(params[1])+int(params[2]) - 40
        not_rand_rollout.append([num, elo])

rand_rollout.sort()
not_rand_rollout.sort()

x = [rand_rollout[i][0] for i in range(len(rand_rollout))]
y = [rand_rollout[i][1] for i in range(len(rand_rollout))]

plt.plot(x, y, label="Random rollout")

x = [not_rand_rollout[i][0] for i in range(len(not_rand_rollout))]
y = [not_rand_rollout[i][1] for i in range(len(not_rand_rollout))]

plt.plot(x, y, label="Not random rollout")

plt.legend()

plt.xlabel("Number of simulations")
plt.ylabel("Ranking")
plt.show()