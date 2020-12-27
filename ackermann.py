from collections import defaultdict

#25734

MOD = 32768

ack = [[0] * MOD for i in range(5)]

for Z in range(1, MOD):
    for a in range(0, 5):
        for b in range(0, 32768):
            if a == 0:
                v = (b+1)%MOD
            elif b == 0:
                v = ack[a-1][Z]
            else:
                v = ack[a-1][ack[a][b-1]]
            ack[a][b] = v

    if ack[4][1] == 6:
        print("found! ", Z)
