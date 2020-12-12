import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
class Smote:
    #初始化
    def __init__(self, samples, N=10, k=5):
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0
    def over_sampling(self):
        N = int(self.N/100)
        self.synthetic = np.zeros((self.n_samples *N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        #print('neighbors', neighbors)
        for i in range(len(self.samples)):
            #print('samples', self.samples[i])
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)), return_distance=False)[0]#找到一个点的k近邻点
            #print('nna', nnarray)
            self._populate(N, i, nnarray)
        return self.synthetic

    #对于每一个数量少的样本i，选择k个邻近点中的N个，生成N个人工合成的样本
    def _populate(self, N, i, nnarray):
        for j in range(N):
            #print('j', j)
            nn = random.randint(0, self.k-1)
            dif = self.samples[nnarray[nn]]-self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i]+gap*dif
            self.newindex+=1