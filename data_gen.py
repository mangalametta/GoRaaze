from sgfmill import sgf
from sgfmill import sgf_moves
import torch
import random

def board2Data(board, colour):
    tensor = torch.zeros(362, dtype=torch.float32)
    for i in range(19):
        for j in range(19):
            if not board[i][j] is None:
                if board[i][j] == 'b':
                    tensor[i*19+j] = 1.0
                else:
                    tensor[i*19+j] = -1.0
    tensor[361] = int(colour == 'b')
    return tensor.reshape((1,362))

def move2Target(move,result):
    tensor = torch.zeros(362, dtype=torch.float32)
    tensor[move[0]*19 +move[1]] = 1 #
    tensor[361] = int(result == 'b')
    return tensor

def data_fetcher(set_name, sharedQueue):
    # path: path of file list
    with open(set_name+'.txt', 'r') as fp:
        fileList = fp.read().split('\n')
    # number of moves that sampled on one game
    moves = 10
    # main loop
    data = []
    while True:
        idx = random.randint(0,len(fileList)-1)
        with open(fileList[idx], 'rb') as fp:
            sgf_src = fp.read()
        try:
            sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
            board, plays = sgf_moves.get_setup_and_moves(sgf_game)
        except:
            continue
        # skip bad games
        # the last 5 is under consideration of passing move
        if len(plays) <= moves + 5:
            continue
        # sampling
        moveSample = random.sample(range(1,len(plays)-5), moves)
        moveSample.sort()

        # extract board and next move
        moveCount = 0
        for colour, move in plays:
            if moveCount > moveSample[-1]+1:
                break
            if moveCount+1 in moveSample:
                data.append([
                    board2Data(board.board, colour), # with color, 362 dim
                    move2Target(move,sgf_game.get_winner()), # with "black win or not", 362 dim
                ])
            
            moveCount += 1
            if move is None:
                continue
            row, col = move
            try:
                board.play(row, col, colour)
            except:
                break
        # commit to sharedQueue
        if len(data) >= 1000:
            # avoid overfitting(?
            random.shuffle(data)
            for i in range(len(data)):
                sharedQueue.put(data[i])
            data = []
            sharedQueue.task_done()

def getSetSize(set_name):
    with open(set_name+'.txt', 'r') as fp:
        fileList = fp.read().split('\n')
    return len(fileList) * 10

if __name__ == "__main__":
    
    import multiprocessing
    m = multiprocessing.Manager()
    q = m.Queue(2048)
    p1 = multiprocessing.Process(target=data_fetcher, args=('train', q,))
    p1.start()
    p2 = multiprocessing.Process(target=data_fetcher, args=('train', q,))
    p2.start()
    consum_count = 0
    while True:
        q.get()
        consum_count += 1
        print(consum_count)
        if consum_count > 1000:
            break
    p1.terminate()
    p2.terminate()