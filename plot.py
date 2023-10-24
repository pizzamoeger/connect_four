import matplotlib.pyplot as plt

dir = "MCTS_plot/it*sim/"

f = open(dir+"RANKING.txt", "r")
lines = f.readlines()
f.close()

rand_rollout = []
not_rand_rollout = []

for i in range(len(lines)):
    lines[i] = lines[i].split(" ")
    elo = int(lines[i][2].strip("\n"))
    params = lines[i][1].replace(dir, "", 1).strip(":").split(".txt")
    params = params[0].split("_")

    if params[0] == "r":
        num = int(params[1])
        #+int(params[2])+int(params[3]) - 40
        rand_rollout.append([num, elo])
    else:
        num = int(params[0])
        #+int(params[1])+int(params[2]) - 40
        not_rand_rollout.append([num, elo])

rand_rollout.sort()
not_rand_rollout.sort()

new_rr = []
new_nrr = []
count = 1
# if there are multiple elos with same num, take the average
for i in range(len(rand_rollout)-1):
    if rand_rollout[i][0] == rand_rollout[i+1][0]:
        rand_rollout[i+1][1] += rand_rollout[i][1]
        rand_rollout[i][1] = 0

        not_rand_rollout[i+1][1] += not_rand_rollout[i][1]
        not_rand_rollout[i][1] = 0

        count += 1
    else:
        new_rr.append([rand_rollout[i][0], rand_rollout[i][1]/count])
        new_nrr.append([not_rand_rollout[i][0], not_rand_rollout[i][1]/count])
        count = 1
new_rr.append([rand_rollout[-1][0], rand_rollout[-1][1]/count])
new_nrr.append([not_rand_rollout[-1][0], not_rand_rollout[-1][1]/count])

rand_rollout = new_rr
not_rand_rollout = new_nrr

# add all the elos with same num together
for i in range(len(rand_rollout)-1):
    if rand_rollout[i][0] == rand_rollout[i+1][0]:
        rand_rollout[i+1][1] += rand_rollout[i][1]
        rand_rollout[i][1] = 0

x = [rand_rollout[i][0] for i in range(len(rand_rollout))]
y = [rand_rollout[i][1] for i in range(len(rand_rollout))]

plt.plot(x, y, label="Random rollout")

x = [not_rand_rollout[i][0] for i in range(len(not_rand_rollout))]
y = [not_rand_rollout[i][1] for i in range(len(not_rand_rollout))]

plt.plot(x, y, label="Not random rollout")

plt.legend()

plt.xlabel("Number of rollouts")
plt.ylabel("Ranking")

# save as vector image
plt.savefig(dir+"/RANKING.svg", format="svg")