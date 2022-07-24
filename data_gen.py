from signal import valid_signals
from sgfmill import sgf
from sgfmill import sgf_moves
import torch
import random

def valid(x,y):
    return 0 <= x < 19 and 0<= y < 19

def getConnectedComponet(board,location,visited):
    r, c = location
    visited[r][c] = 1
    connected = []
    for x,y in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
        if valid(x,y) and board[x][y] == board[r][c] and not visited[x][y]:
            connected = connected + getConnectedComponet(board,(x,y),visited)
    connected.append((r,c))
    return connected

def markKi(board,ki,connected):
    visited = torch.zeros((19,19), dtype=torch.int32)
    kiCount = 0
    for r,c in connected:
        for x,y in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
            if valid(x,y) and board[x][y] is None and not visited[x][y]:
                kiCount += 1
                visited[x][y] = 1
    for r,c in connected:
        ki[r][c] = kiCount


def getKi(board):
    ki = torch.zeros((19,19), dtype=torch.float32)
    visited = torch.zeros((19,19), dtype=torch.int32)
    for r in range(19):
        for c in range(19):
            if board[r][c] is None or visited[r][c]:
                continue
            connected = getConnectedComponet(board,(r,c),visited)
            markKi(board, ki, connected)
    ki = torch.reshape(ki,(1,19,19))
    return ki
    
    

def board2Data(board, colour):
    # we do three channel? or 4( include handcraft feature?
    tensor = torch.zeros((3,19,19), dtype=torch.float32)
    for i in range(19):
        for j in range(19):
            if not board[i][j] is None:
                if board[i][j] == 'b':
                    tensor[0][i][j] = 1.0
                elif board[i][j] == 'w':
                    tensor[1][i][j] = 1.0
                else:
                    tensor[2][i][j] = 1.0
    ki = getKi(board)
    # concat feature
    tensor = torch.concat((tensor,ki),0)
    # concat colour
    color = torch.zeros((1), dtype=torch.float32)
    if colour == 'b':
        color[0] = 1
    return (tensor, color)

def move2Target(move,result):
    tensor = torch.zeros((1,19,19), dtype=torch.float32)
    win = torch.zeros((1), dtype=torch.float32)
    tensor[0][move[0]][move[1]] = 1 #
    win[0] = 1 if result == 'b' else 0
    return (tensor,win)

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
            
        except Exception as e:
            print(e)
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
        # test
        '''
        print(fileList[idx], moveCount)
        print(board.board)
        print(getKi(board.board))
        exit()
        '''
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
    b = None
    while True:
        q.get()
        consum_count += 1
        print(consum_count)
        if consum_count > 1000:
            break
    p1.terminate()
    p2.terminate()

