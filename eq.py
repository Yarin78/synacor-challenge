from itertools import permutations

for (a, b, c, d, e) in permutations([2, 3, 5, 7, 9], 5):
    #print(a,b,c,d,e)
    if a + b*c*c + d*d*d - e == 399:
        print(a,b,c,d,e)
