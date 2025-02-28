import sys
from rehualPrim import randomSample
if(len(sys.argv) != 5):
    print("Incorrect number of command lines. randmst terminal usage is: \n")
    print("python randmst.py [flag] [numpoints] [numtrials] [dimension]")
    sys.exit()
try:
    global flag
    flag = int(sys.argv[1])
except ValueError:
    print("flag must be an integer.")
    sys.exit()
try:
    global nump
    nump = int(sys.argv[2])
except ValueError:
    print("numpoints must be an integer.")
    sys.exit()
try:
    global numt
    numt = int(sys.argv[3])
except ValueError:
    print("numtrials must be an integer.")
    sys.exit()
try:
    global dim
    dim = int(sys.argv[4])
except ValueError:
    print("dimension must be an integer.")
    sys.exit()
randomSample(nump,numt,dim)