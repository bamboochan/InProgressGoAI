import numpy as np
import tarfile
import re
import random


def getdata(tar, filename):
    member = tar.getmember("pro/" + filename)
    f = tar.extractfile(member)
    data = str(f.read())
    starts = [m.start() for m in re.finditer(';', data)]
    length = len(starts) - 1
    black = np.zeros((19, 19), dtype=np.int)
    white = np.zeros((19, 19), dtype=np.int)
    ko = np.zeros((19, 19), dtype=np.int)
    pos = np.zeros((19, 19, 3), dtype=np.int)
    answer = np.zeros((19, 19), dtype=np.int)
    positions, moves = np.zeros((length, 19, 19, 3), dtype=np.int), np.zeros((length, 19, 19), dtype=np.int)
    for j1, i in enumerate(starts[1:]):
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        if x < 0 or x >= 19 or y < 0 or y >= 19:
            continue
        answer[x][y] = 1
        if data[i + 1] == 'B':
            for i in range(19):
                for j in range(19):
                    pos[i][j][0] = black[i][j]
                    pos[i][j][1] = white[i][j]
                    pos[i][j][2] = ko[i][j]
            positions[j1] = np.copy(pos)
            black[x][y] = 1
        else:
            for i in range(19):
                for j in range(19):
                    pos[i][j][0] = white[i][j]
                    pos[i][j][1] = black[i][j]
                    pos[i][j][2] = ko[i][j]
            positions[j] = np.copy(pos)
            white[x][y] = 1
        moves[j1] = np.copy(answer)
        answer[x][y] = 0
    return positions[10:], moves[10:]

# res = getdata(tar, "00010.sgf")
# arr = res[11][3]
# np.savetxt('test.txt', arr, '%d')
