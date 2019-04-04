#coding:__utf-8__
import trees
import treePlotter
fr=open('西瓜.txt')
xigua = [inst.strip().split(' ') for inst in fr.readlines()]
xigua = [[float(i) if '.' in i else i for i in row] for row in xigua]  # change decimal from string to float
Labels = ['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
xiguaTree = trees.createTree(xigua,Labels,xigua,Labels)
treePlotter.createPlot(xiguaTree)
