import init
from genGraphs.py import genGraph0, genGraph1, genGraph2, genGraph3, genGraph4
init.initialize()
if(init.flag == 0):
    graph = genGraph0(init.nump)
if(init.flag == 1):
    graph = genGraph1(init.nump)
if(init.flag == 2):
    graph = genGraph2(init.nump)
if(init.flag == 3):
    graph = genGraph3(init.nump)
if(init.flag == 4):
    graph = genGraph4(init.nump)
