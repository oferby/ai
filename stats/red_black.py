import random


win = 0
lose = 0
for j in range(10000):
    coin = []
    for i in range(6):
        coin.append(random.randrange(0, 2))
    print (coin)
    if coin == [0,0,0,0,0,0]:
        r = random.randrange(0, 2)
        if r == 1:
            win += 1
        else:
            lose += 1
print("win:", win, "lose:", lose)