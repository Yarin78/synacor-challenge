matrix = [
    ['*', 8, '-', 1],
    [4, '*', 11, '*'],
    ['+', 4, '-', 18],
    [22, '-', 9, '*']
]

directions = [('E', 1,0), ('W', -1,0), ('N', 0,-1), ('S', 0,1)]

def search(x, y, cur, s, left):
    if x == 3 and y == 0:
        if cur==30:
            print(s)
            return True
        return False
    if left == 0:
        return False

    for d, dx, dy in directions:
        if y+dy < 0 or y+dy >= 4:
            continue
        if x+dx < 0 or x+dx >= 4:
            continue        
        for e, ex, ey in directions:
            if y+dy+ey < 0 or y+dy+ey >= 4:
                continue
            if x+dx+ex < 0 or x+dx+ex >= 4:
                continue        
            if x+dx+ex == 0 and y+dy+ey == 3:
                continue
            
            op = matrix[y+dy][x+dx]
            val = matrix[y+dy+ey][x+dx+ex]
            
            if op == '+':
                new_val = (cur + val) % 32768
            elif op == '-':
                new_val = (cur - val) % 32768
            elif op == '*':
                new_val = (cur * val) % 32768
            
            if search(x+dx+ex, y+dy+ey, new_val, f'{s}{d}{e} ', left-1):
                return True
                
    return False

for i in range(1, 10):
    search(0, 3, 22, '', i)
