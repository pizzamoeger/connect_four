import subprocess
import math
import os

print(os.getcwd())

base = 1.5

#fcw, fcb, cnw, cnb, l2, mom
params = [-20, -7, -20, -7, -100000, -100000]
names = ["fully connected weight learning rate", "fully connected bias learning rate", "convolutional weight learning rate", "convolutional bias learning rate", "l2", "mom"]

s = 35
st = [10, 10, 3, 3]

testfile = 'train_DQL_b1024'
name = 'DQL/fc_1024_15_100.txt'

def evaluate():
    process = subprocess.Popen(['./bash_scrips/eval_dql.sh'] + [str(base**a) for a in params] + [testfile] + [name], stdout=subprocess.PIPE)

    output, _ = process.communicate()

    output = output.decode()
    output = output.split("\n")
    output = output[-2]
    print("output: {}".format(output), flush=True)
    return float(output)

dp = {}

def goodness(x):
    return -15*x**2 + 9/4*x + 7

def score():
    dp_name = "".join(["{:6.4f}".format(el) for el in params])
    if dp_name in dp:
        return dp[dp_name]
    runs = 5
    average = 0
    for _ in range(runs):
        average += evaluate()/runs
    dp[dp_name] = average
    return dp[dp_name]

def betterrange(l, r, s):
    li = []
    while l < r:
        li.append(l)
        l += s
    return li

def searchopt(i):
    l = params[i] - s
    r = params[i] + s
    for ste in st:
        ttest = betterrange(l, r + 10**-6, (r-l)/ste)
        vals = []
        for a in ttest:
            params[i] = a
            print("testing {}={}\n============================================".format(names[i], base**params[i]), flush=True)
            vals.append(score())
            print("============================================", flush=True)
        start = -1
        best = 0
        for test in range(len(vals) - 1):
            if start == -1 or vals[test] + vals[test+1] > best:
                start = test
                best = vals[test] + vals[test+1]
        l = ttest[start]
        r = ttest[start+1]
    params[i] = (l + r) / 2
    print("Done looking for best {}, found {} to be best".format(names[i], base**params[i]), flush=True)

searchopt(0)
searchopt(1)

print("============================================", flush=True)
print("I am done: the best parameters are: {}".format([base**el for el in params]), flush=True)

with open(input(), "w") as file:
    file.write(str([base**el for el in params]))
