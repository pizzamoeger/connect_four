import matplotlib.pyplot as plt

dir = "data/DQN/c/"

f = open(dir+"RANKING.txt", "r")
lines = f.readlines()
f.close()

rand_rollout = []
not_rand_rollout = []
#for i in range(10):
#    rand_rollout.append([])

for i in range(len(lines)):
    lines[i] = lines[i].split(" ")
    elo = int(lines[i][2].strip("\n"))
    params = lines[i][1].replace(dir, "", 1).strip(":").split(".txt")
    params = params[0].split("_")

    if params[0] == "fc":
        num = int(params[1])#+int(params[2])+int(params[3]) - 8-8
        #num = int(params[1])
        rand_rollout.append([num, elo]) #rand_roll_out[int(params[4])-1]
    else:
        num = int(params[0])#+int(params[1])+int(params[2]) - 8-8
        #num = int(params[1])
        not_rand_rollout.append([num, elo])

"""for hypar in range(10):
    if (len(rand_rollout[hypar]) == 0):
        continue
    rand_rollout[hypar].sort()
    #not_rand_rollout.sort()

    new_rr = []
    new_nrr = []
    count = 1
    # if there are multiple elos with same num, take the average
    for i in range(len(rand_rollout[hypar])-1):
        if rand_rollout[hypar][i][0] == rand_rollout[hypar][i+1][0]:
            rand_rollout[hypar][i+1][1] += rand_rollout[hypar][i][1]
            rand_rollout[hypar][i][1] = 0

            #not_rand_rollout[i+1][1] += not_rand_rollout[i][1]
            #not_rand_rollout[i][1] = 0

            count += 1
        else:
            new_rr.append([rand_rollout[i][0], rand_rollout[i][1]/count])
            new_nrr.append([not_rand_rollout[i][0], not_rand_rollout[i][1]/count])
            count = 1
    new_rr.append([rand_rollout[-1][0], rand_rollout[-1][1]/count])
    new_nrr.append([not_rand_rollout[-1][0], not_rand_rollout[-1][1]/count])

    rand_rollout[hypar] = new_rr
    #not_rand_rollout = new_nrr

    # add all the elos with same num together
    for i in range(len(rand_rollout[hypar])-1):
        if rand_rollout[hypar][i][0] == rand_rollout[hypar][i+1][0]:
            rand_rollout[hypar][i+1][1] += rand_rollout[hypar][i][1]
            rand_rollout[hypar][i][1] = 0

    x = [rand_rollout[hypar][i][0] for i in range(len(rand_rollout[hypar]))]
    y = [rand_rollout[hypar][i][1] for i in range(len(rand_rollout[hypar]))]

    plt.plot(x, y, label=str(hypar+1))"""

rand_rollout.sort()
not_rand_rollout.sort()

new_nrr = []
new_rr = []

for i in range(len(not_rand_rollout)):
    if (i%3 == 0):
        if (i > 0):
            new_nrr[-1][1] /= 3
            #new_rr[-1][1] /= 3
        new_nrr.append(not_rand_rollout[i])
        #new_rr.append(rand_rollout[i])
    else:
        new_nrr[-1][1] += not_rand_rollout[i][1]
        #new_rr[-1][1] += rand_rollout[i][1]

new_nrr[-1][1] /= 3
#new_rr[-1][1] /= 3
not_rand_rollout = new_nrr
#rand_rollout = new_rr

f = open("nrr_data.txt", "w")
for i in not_rand_rollout:
    f.write(str(i[0])+": "+str(i[1])+"\n")
f.close()
f = open("rr_data.txt", "w")
for i in rand_rollout:
    f.write(str(i[0])+": "+str(i[1])+"\n")
f.close()
"""
x = [not_rand_rollout[i][0] for i in range(len(not_rand_rollout))]
y = [not_rand_rollout[i][1] for i in range(len(not_rand_rollout))]

plt.plot(x, y, label="Not random rollout")

x = [rand_rollout[i][0] for i in range(len(rand_rollout))]
y = [rand_rollout[i][1] for i in range(len(rand_rollout))]

plt.plot(x, y, label="Random rollout")

plt.legend(ncol=4)

plt.xlabel("Num of Simulations")
plt.ylabel("ELO Ranking")

# save as vector image
plt.savefig(dir+"/RANKING.svg", format="svg")
"""
