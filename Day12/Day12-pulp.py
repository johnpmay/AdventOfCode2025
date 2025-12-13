#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
from time import time
import math
import numpy as np

input_file = Path('../../AdventOfCode_inputs/AoC-2025-12-input.txt')

rinput = input_file.read_text()


# In[ ]:


sinput = '''0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
'''


# In[ ]:


ainput = rinput


# In[ ]:


[*shapes, trees] = ainput.split('\n\n')
shapes = [ s.splitlines()[1:]  for s in shapes]
trees = [ [list(map(int,a[0].split("x"))), list(map(int, a[1].split())) ] for a in [t.split(': ') for t in trees.splitlines()]]


# In[ ]:


svols = [sum( shapes[j][i].count('#') for i in range(3)) for j in range(6)]


# In[ ]:


def shape2Num(shape):
    return eval("0b"+"".join([ '1' if c=="#" else '0' for c in "".join(shape) ]))
def flipshape(shape):
    return list(s[::-1] for s in shape)
def rotateshape(shape):
    transp = np.array([list(r) for r in shape]).transpose()
    return [ "".join(transp[i])[::-1] for i in range(3) ]
def allshapeNums(shape):
    r = [0]*8
    r[0] = shape
    r[1] = rotateshape(r[0])
    r[2] = rotateshape(r[1])
    r[3] = rotateshape(r[2])
    r[4] = flipshape(r[0])
    r[5] = flipshape(r[1])
    r[6] = flipshape(r[2])
    r[7] = flipshape(r[3])
    return set(map(shape2Num, r))
def num2Shape(num):
     return [("..."+s[2:])[-3:].replace('1','#').replace('0','.') for s in [bin(num//2**6%2**3), bin(num//2**3%2**3), bin(num%2**3)]]

Slist = [ allshapeNums(s) for s in shapes ]
S = sorted(list(set.union(*Slist)))

masks = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]


# In[ ]:


S


# In[ ]:


num2Shape( 256 ^ 502 )


# **Part One**

# In[ ]:


from pulp import *

startTime = time()
count = 0

for t in range(len(trees)):

    h, w = trees[t][0]
    solve = LpProblem("TreeProblem-%s"%t)
    
    # variables: T[x,y,s] = 1 <==> there is a tile s at pos x,y
    #T = {(i,j,s):Int("T__%s,%s,%s" % (i,j,s)) for s in S for j in range(w) for i in range(h) }
    T = LpVariable.dicts("T", (range(-1,h+1),range(-1,w+1),S), cat="Binary")
    
    # Tiles cannot be positioned on the edges
    for i in range(-1,h+1):
        for s in S:
            T[i][-1][s] = 0
            T[i][0][s] = 0
            T[i][w-1][s] = 0
            T[i][w][s] = 0
    for j in range(-1,w+1):
        for s in S:
            T[-1][j][s] = 0
            T[0][j][s] = 0
            T[h-1][j][s] = 0
            T[h][j][s] = 0
            
    # the T's are 0 or 1
    for i in range(1,h-1):
        for j in range(1,w-1):
            for s in S:
                solve+=(T[i][j][s]>=0)
                solve+=(T[i][j][s]<=1)

    # only one s at each x,y
    for i in range(h):
        for j in range(w):
            solve+=(sum(T[i][j][s] for s in S)<=1)

    # exactly the correct number of each s
    for k in range(len(shapes)):
        solve+=(sum(sum(T[i][j][s] for i in range(h) for j in range(w)) for s in Slist[k]) == trees[t][1][k])

    # no overlaps
    for i in range(h):
        for j in range(w):
            solve+=( sum( sum(T[i+masks[k][0]][j+masks[k][1]][s]*((s//2**k)%2) for s in S) for k in range(9)) <= 1 )
    
    solve.solve()
    if solve.status > 0:
        count+=1
                                               
print("%s s"%(time()-startTime))  
ans1 = count


# In[ ]:
print("\n\n\n\n\n\n\n")

print("ans1 = %s"%ans1)


# In[ ]:





# In[ ]:




