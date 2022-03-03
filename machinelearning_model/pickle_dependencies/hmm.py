import numpy as np
from hmmlearn.hmm import MultinomialHMM


class MyHMM(MultinomialHMM):
    """
    only applies to W:5, R:4, N1:3, N2:2, N3:1
    """
    def fit(self, ys, yps):
        """
        ys is a array/list of sleep stages (array/list)
        yps is a array/list of predicted sleep stages (array/list)
        """
        assert len(ys)==len(yps)
        assert self.n_components==5
        
        startmat = np.zeros(5)
        transmat = np.zeros((5,5))
        emissionmat = np.zeros((5,5))
        
        for i in range(len(ys)):
            y =  np.array(ys[i])
            yp = np.array(yps[i])
            startmat[int(y[0])-1] += 1
            transmat += np.array([[np.sum((y[:-1]==ii)&(y[1:]==jj)) for jj in range(1,6)] for ii in range(1,6)])
            emissionmat += np.array([[np.sum([(y==ii)&(yp==jj)]) for jj in range(1,6)] for ii in range(1,6)])
            
        self.n_features = self.n_components
        self.transmat_ = transmat*1./transmat.sum(axis=1,keepdims=True)
        self.emissionprob_ = emissionmat*1./emissionmat.sum(axis=1,keepdims=True)
        self.startprob_ = startmat*1./startmat.sum()
        
    def predict(self, yps):
        """
        yps is a array/list of predicted sleep stages (array/list)
        """
        ys = []
        for yp in yps:
            y = (super().predict(np.array(yp).reshape(-1,1)-1)+1).flatten().astype(int) #-1 and +1 to switch between 0-based and 1-based encoding
            ys.append(y)
        return ys
        
        
if __name__=='__main__':
    hmm = MyHMM(n_components=5, algorithm='viterbi')
    hmm.fit([[5,4,3,2,1,1,1,2,3,4,5]],[[5,4,4,2,1,3,1,3,3,4,5]])
    ys = hmm.predict([[5,4,3,3,3,3,1,2,3,4,5,3,2,2,3,4,4,2]])
    print(ys)
