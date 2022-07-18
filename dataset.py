from torch.utils.data import Dataset
from data_gen import *
import multiprocessing
#from net import Network

class GoData(Dataset):
    def __init__(self,set_name='train'):
        self.size = getSetSize(set_name)
        m = multiprocessing.Manager()
        self.q = m.Queue(1024)
        self.process_count = 2
        self.processes = []
        for i in range(self.process_count):
            self.processes.append(multiprocessing.Process(target=data_fetcher, args=(set_name, self.q,)))
            self.processes[-1].start()
        

    def __getitem__(self,index):
        # don't care index, we are random sampling
        data,label = self.q.get()
        return data, label

    def __len__(self):
        return self.size

    def __del__(self):
        for p in self.processes:
            p.terminate()

if __name__ == '__main__':
    pass
    '''
    data = GoData()
    net = Network()
    print(data[0][1].shape)
    
    for i in range(512):
        x = torch.reshape(data[i][0],(1,1,362))
        print(net.forward(x))
    del(data)
    '''