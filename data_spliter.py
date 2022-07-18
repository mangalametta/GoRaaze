import glob
import os
import random

def getFiles(path):
	# includes the full path
	return glob.glob(os.path.join(path,'*'))

def genFileListSplit(percents,path):
    # sum of percents should be 1
    # precents contains at least 2 and at most 3
    # get files
    files = getFiles(path)
    # filtering
    files = [f for f in files if f.split('.')[-1] == 'sgf']
    # random shuffle
    random.shuffle(files)
    # split
    splited = []
    start = 0
    for i in range(len(percents)-1):
        end = start+int(percents[i]*len(files))
        splited.append(files[start:end])
        start = end + 1
    splited.append(files[start:])
    # write to file
    with open('train.txt','w') as fp:
        fp.write('\n'.join(splited[0]))
    with open('test.txt','w') as fp:
        fp.write('\n'.join(splited[-1]))
    if len(percents) == 3:
        with open('val.txt','w') as fp:
            fp.write('\n'.join(splited[-1]))

if __name__ == '__main__':
    genFileListSplit((0.7,0.2,0.1),'data')