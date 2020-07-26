import numpy as np
from collections import defaultdict

class customTSNE:
    def __init__(self,learning_rate=0.5,perplexity=2,no_dims=2,max_iter=200,momentum=0.8):
        self.learning_rate=learning_rate
        self.perplexity=perplexity
        self.no_dims=no_dims
        self.max_iter=max_iter
        self.momentum=momentum
        
    
    def pairwise_squared_d(self,X):
        sum_X = np.sum(np.square(X), 1, keepdims=True)
        suqared_d = sum_X + sum_X.T - 2*np.dot(X, X.T)
        sd = suqared_d.clip(min=0)
        return sd
    
    
    def cal_p(self,suqared_d, sigma, i):
       
        a = 2*sigma**2
        prob = np.exp(-suqared_d/a)
        prob[i] = 0
        prob = prob/np.sum(prob)    
        H = -np.sum([p*np.log2(p) for p in prob if p!=0])
        perp = 2**H
        return prob,perp
    
    def search_sigma(self,x, i, perplexity, tol): 
        
        # binary search
        sigma_min, sigma_max = 0, np.inf       
        prob,perp = self.cal_p(x, sigma[i], i)
        perp_diff = perplexity - perp                 
        times = 0
        hit_upper_limit = False
        while (abs(perp_diff) > tol) and (times<50):
            if perp_diff > 0:
                if hit_upper_limit:
                    sigma_min = sigma[i]
                    sigma[i] = (sigma_min + sigma_max)/2
                else:
                    sigma_min, sigma_max = sigma[i], sigma[i]*2
                    sigma[i] = sigma_max
            else:
                sigma_max = sigma[i]
                sigma[i] = (sigma_min + sigma_max) / 2
                hit_upper_limit = True
            prob,perp = self.cal_p(x, sigma[i], i)
            perp_diff = perplexity - perp  
            times = times + 1
        return prob
    
    
    def get_prob(self,X, perplexity, verbose,tol=1e-4):
        
        n = X.shape[0]
        squared_d = self.pairwise_squared_d(X)
        squared_d = squared_d/np.std(squared_d, axis=-1)*10
        # init
        pairwise_prob = np.zeros((n,n))
        global sigma
        sigma = np.ones(n)

        for i in range(n):
            x = squared_d[i]
            prob = self.search_sigma(x, i,perplexity, tol)
            pairwise_prob[i] = prob
            if i%100 == 0:
                if verbose is not None:
                    print("processed %s of total %s points"%(i,n))
        return pairwise_prob
    
    
    def pca(self,x, n_components=None):
        
        vec, val = np.linalg.eig(np.dot(x.T, x))
        assert np.alltrue(np.imag(val)) == False
        
        if n_components:
            return np.real(np.dot(x, val[:,0:n_components]))
        else:
            v_p = vec/sum(vec)
            v_s, i = 0, 0
            while v_s < 0.8:
                v_s += v_p[i]
                i += 1
            return np.real(np.dot(x, val[:,0:i]))
    
    
    
    
    def run(self,X,verbose=None):
        n=X.shape[0]
        scale = lambda x: np.nan_to_num((x-np.mean(x,axis=0))/np.std(x,axis=0), 0)
        P =self.get_prob(self.pca(scale(X)),self.perplexity, tol=1e-2,verbose=verbose)
        assert not np.any(np.isnan(P))
        P = P + np.transpose(P)
        P = P / (2)
        tsne_Di = defaultdict(list)
        key_ = str(self.learning_rate)+'__'+str(self.momentum)

        y_ = np.random.normal(loc=0,scale=0.01,size=(n,self.no_dims))
        li = tsne_Di[key_]

        v = 0
        if verbose is not None:
            print("Cross entropy:")
        for iter in range(self.max_iter):
            y_s_dist = self.pairwise_squared_d(y_)
            q = 1/(1+y_s_dist)
            np.fill_diagonal(q,0)
            Q = q/np.sum(q, axis=1, keepdims=True)
            y_f = y_.flatten()
            d = y_f.reshape(self.no_dims, n, 1, order='F') - y_f.reshape(self.no_dims, 1, n, order='F')
            CE = -P* np.log2(Q)
            np.fill_diagonal(CE, 0)
            #if iter%2==0:
            li.append(y_.copy())
            if iter%10==0:
                if verbose is not None:
                    print(CE.sum())
            gd = 4*(P-Q)*q*d
            gradient = np.sum(gd, axis=2).T

            v = self.learning_rate*gradient + self.momentum*v    
            y_ = y_ - v
        return li
    
    

