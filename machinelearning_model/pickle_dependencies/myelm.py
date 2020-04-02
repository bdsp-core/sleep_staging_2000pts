#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import combinations
from collections import Counter
import os.path
import numpy as np
from scipy.stats import mode
from scipy.linalg import orth
from numpy.linalg import svd, lstsq, inv, pinv, multi_dot
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import as_float_array, check_array, check_X_y, check_random_state
#from sklearn.utils.fixes import expit as sigmoid
from scipy.special import expit as sigmoid
#from sklearn.utils.estimator_checks import check_estimator
#from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.linear_model import Ridge, RidgeClassifier, Lasso
from sklearn import metrics
#import matlab.engine
#from cvxpy import *
#from utils import *
#from mysoftclassifier import *

dot = np.dot  # alias for np.dot


#def sigmoid(x):
#    return 0.5*np.tanh(0.5*x)+0.5


class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=100, C=1.0, batch_size=None, fit_intercept=False, ovo=False, classes=None, activation_func='sigmoid', return_y=False, random_projection=True, random_state=None):
        self.n_hidden = n_hidden
        self.C = C
        self.W = None
        self.b = None
        self.beta = None
        self.P = None  # P = (H'*H+C*I)^-1
        self.activation_func = activation_func.lower()
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_projection = random_projection
        self.random_state = random_state
        self.random_state_ = None
        self.ovo = ovo
        self.classes = classes#np.array([1,2,3,4,5])
        self.return_y = return_y
        self.label_binarizer = None
        self.fitted_ = False

    def _validate_X(self, X):
        if len(X.shape)==1:
            raise ValueError('X should be a 2-dimensional array.')
        #    if one feature:
        #        X = X.reshape(1,-1)
        #    else:  # one sample
        #        X = X.reshape(-1,1)
        if X.shape[0]==0:
            raise ValueError('Empty samples.')
        if X.shape[1]==0:
            raise ValueError('0 feature(s) (shape=(3, 0)) while a minimum of %d is required.'%(1,))
        return as_float_array(check_array(X))

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        X, y = check_X_y(X, y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        if np.allclose(np.array(y,dtype=int),y):
            self.label_binarizer = LabelBinarizer(neg_label=-2,pos_label=2,sparse_output=False)# y \in {-2,2} according to extreme logistic regression
            if self.classes is not None:
                self.label_binarizer.fit(self.classes)
                y = self.label_binarizer.transform(y)
            else:
                y = self.label_binarizer.fit_transform(y)
                self.classes = self.label_binarizer.classes_
            if self.label_binarizer.classes_.shape[0]<2:
                raise ValueError('Label contains less than 2 classes.')
        else:
            self.label_binarizer = None
            self.fit_intercept = True
        return X, y

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = False
        self.random_state_ = check_random_state(self.random_state)

        if np.any(np.isnan(y)):
            nonnan_ids = np.logical_not(np.isnan(y))
            X = X[nonnan_ids,:]
            y = y[nonnan_ids]
        X, y = self._validate_X_y(X, y)
        N, dx = X.shape
        N_ = N-self.n_hidden
        self.classes_ = self.classes
        #self.n_classes_ = len(self.classes)
        if self.random_projection and (self.batch_size is None or self.P is None):
            self.b = self.random_state_.uniform(size=self.n_hidden)*2-1
            self.W = self.random_state_.uniform(size=(dx,self.n_hidden))*2-1

        if self.batch_size is None or N_<=0:
            # fit all
            if self.random_projection:
                if self.activation_func == 'sigmoid':
                    H = sigmoid(dot(X,self.W)+self.b)
                else:
                    raise NotImplementedError('activation_func="%s" is not implemented.')
            else:
                self.n_hidden = X.shape[1]
                H = X
            if self.label_binarizer is None:
                if self.ovo:
                    raise NotImplementedError('OVO for probabilistic label is not implemented yet.')
                if sample_weight is not None:
                    raise NotImplementedError('sampled_weight for probabilistic label is not implemented yet.')
                if not hasattr(self,'fit_intercept') or not self.fit_intercept:
                    raise TypeError('For probabilistic labels, self.fit_intercept must be True.')
                output_layer=SoftLogisticRegression(C=self.C, learning_rate=0.01, momentum=0.9, max_iter=200,
                                random_state=self.random_state, tol=1e-4, verbose=False).fit(H,y)
                self.beta = np.r_[output_layer.coefs_[-1].ravel(),output_layer.intercepts_[-1]]
            else:
                if hasattr(self,'fit_intercept') and self.fit_intercept:
                    H = np.c_[X,np.ones((N,1))]
                    nh = self.n_hidden+1
                else:
                    nh = self.n_hidden
                if N>self.n_hidden:
                    if self.ovo:
                        if sample_weight is not None:
                            raise NotImplementedError('OVO and sampled_weight at the same time is not implemented yet.')
                        self.beta = np.empty((nh,self.label_binarizer.classes_.shape[0]*(self.label_binarizer.classes_.shape[0]-1)//2))
                        cc = 0
                        for ii in combinations(range(self.label_binarizer.classes_.shape[0]),2):
                            id_ = np.where(np.logical_or(y[:,ii[0]]==2,y[:,ii[1]]==2))[0]
                            #if self.C==0:
                            #    self.beta[:,cc] = dot(pinv(H[id_,:]),y[id_,ii[0]])
                            #else:
                            Ht_ = H[id_,:].T
                            self.beta[:,cc] = multi_dot((inv(dot(Ht_,Ht_.T)+self.C*N*1.0/nh*np.eye(nh)),Ht_,y[id_,ii[0]]))
                            cc += 1
                    else:
                        if sample_weight is None:
                            #if self.C==0:
                            #    self.beta = dot(pinv(H),y)
                            #else:
                            self.beta = multi_dot((inv(dot(H.T,H)+self.C*N*1.0/nh*np.eye(nh)),H.T,y))
                        else:
                            Ht =sample_weight*H.T
                            #if self.C==0:
                            #    self.beta = dot(pinv(Ht.T),y)
                            #else:
                            self.beta = multi_dot((inv(dot(Ht,H)+self.C*1.0/nh*np.eye(nh)),Ht,y))
                else:
                    if self.ovo:
                        if sample_weight is not None:
                            raise NotImplementedError('OVO and sampled_weight at the same time is not implemented yet.')
                        n_beta = self.label_binarizer.classes_.shape[0]*(self.label_binarizer.classes_.shape[0]-1)//2
                        self.beta = np.empty((nh,n_beta))
                        cc = 0
                        for ii in combinations(range(self.label_binarizer.classes_.shape[0]),2):
                            id_ = np.where(np.logical_or(y[:,ii[0]]==2,y[:,ii[1]]==2))[0]
                            H_ = H[id_,:]
                            #if self.C==0:
                            #    self.beta[:,cc] = dot(pinv(H_),y[id_,ii[0]])
                            #else:
                            self.beta[:,cc] = multi_dot((H_.T,inv(dot(H_,H_.T)+self.C*N*1.0/nh*np.eye(N)),y[id_,ii[0]]))
                            cc += 1
                    else:
                        if sample_weight is None:
                            #if self.C==0:
                            #    self.beta = dot(pinv(H),y)
                            #else:
                            self.beta = multi_dot((H.T,inv(dot(H,H.T)+self.C*N*1.0/nh*np.eye(N)),y))
                        else:
                            self.beta = multi_dot((H.T,inv((sample_weight*dot(H,H.T)).T+self.C*1.0/nh*np.eye(N)),(sample_weight*y.T).T))
        else:
            # OS-ELM
            raise NotImplementedError('OS-ELM is not implemented yet.')
            if self.ovo:
                raise NotImplementedError('OVO in batch mode is not implemented yet.')
            if sample_weight is not None:
                raise NotImplementedError('sampled_weight in batch mode is not implemented yet.')
            if N_%self.batch_size==0:
                batches = [self.n_hidden]+[self.batch_size]*(N_//self.batch_size)
            else:
                batches = [self.n_hidden]+[self.batch_size]*(N_//self.batch_size)+[N_%self.batch_size]
            #shuffled_id = list(range(N))
            #self.random_state_.shuffle(shuffled_id)
            #X = X[shuffled_id,:]
            #y = y[shuffled_id]

            for i in range(len(batches)):
                start_n = sum(batches[:i])
                end_n = sum(batches[:i+1])
                y_part = y[start_n:end_n]
                if self.random_projection:
                    if self.activation_func == 'sigmoid':
                        H = sigmoid(dot(X[start_n:end_n,:],self.W)+self.b)
                        if hasattr(self,'fit_intercept') and self.fit_intercept:
                            H = np.c_[H,np.ones((batches[i],1))]
                    else:
                        raise NotImplementedError('activation_func="%s" is not implemented.')
                else:
                    self.n_hidden = X.shape[1]
                    if hasattr(self,'fit_intercept') and self.fit_intercept:
                        H = np.c_[X[start_n:end_n,:],np.ones((batches[i],1))]
                    else:
                        H = X[start_n:end_n,:]

                if i==0 or self.P is None:
                    if hasattr(self,'fit_intercept') and self.fit_intercept:
                        nh = self.n_hidden+1
                    else:
                        nh = self.n_hidden
                    self.P = inv(dot(H.T,H)+self.C*N*1.0/nh*np.eye(nh))
                    self.beta = multi_dot((self.P,H.T,y_part))
                else:
                    if N==1:
                        h = H.ravel()
                        hht = np.outer(h,h)
                        self.P = self.P - multi_dot((self.P,hht,self.P))/(1.+(self.P*hht).sum())
                    else:
                        PHt = dot(self.P,H.T)
                        self.P = self.P - multi_dot((PHt,inv(dot(H,PHt)+np.eye(batches[i])),H,self.P))

                    self.beta = self.beta + dot(dot(self.P,H.T),y_part-dot(H,self.beta))

        self.fitted_ = True
        return self

    def fit_transform(self, X, y):
        return self.fit(X,y).transform(X)

    def transform(self, X):
        return self.decision_function(X)

    def decision_function(self, X):
        if not self.fitted_:
            raise ValueError('This ELMClassifier instance is not fitted yet.')
        X = self._validate_X(X)
        if self.random_projection:
            H = sigmoid(dot(X,self.W)+self.b)
        else:
            H = X
        if hasattr(self,'fit_intercept') and self.fit_intercept:
            H = np.hstack((H,np.ones((X.shape[0],1))))
        return dot(H,self.beta)

    def predict(self, X):
        if self.ovo:
            yy = self.decision_function(X)
            cc = 0
            for ii in combinations(range(self.label_binarizer.classes_.shape[0]),2):
                id_ = yy[:,cc]>=0
                yy[:,cc][id_] = ii[0]
                yy[:,cc][np.logical_not(id_)] = ii[1]
                cc += 1
            yy = mode(yy,axis=1)[0].ravel()
            return self.label_binarizer.inverse_transform(label_binarize(yy, range(self.label_binarizer.classes_.shape[0])))
        else:
            proba, y = self.predict_proba(X,return_y=True)
            if y is None:
                return proba
            else:
                return y

    def predict_proba(self, X):
        # [1] Ngufor, C., & Wojtusiak, J. (2013).
        #     Learning from large-scale distributed health data: An approximate logistic regression approach.
        #     In Proceedings of the 30th International Conference on Machine Learning, Atlanta, Georgia, USA, JMLR: W&CP (pp. 1-8).
        # [2] Ngufor, C., Wojtusiak, J., Hooker, A., Oz, T., & Hadley, J. (2014, May).
        #     Extreme Logistic Regression: A Large Scale Learning Algorithm with Application to Prostate Cancer Mortality Prediction.
        #     In FLAIRS Conference.
        #if self.label_binarizer.classes_.shape[0]!=2:
        #    print('Warning: This is one-vs-all probability for each class.')

        if self.ovo:
            proba = label_binarize(self.predict(X),self.label_binarizer.classes_)
            """
            K = self.label_binarizer.classes_.shape[0]
            proba = np.zeros((X.shape[0],K))
            for i in range(K):
                cc = 0
                for ii in combinations(range(self.label_binarizer.classes_.shape[0]),2):
                    if ii[0]==i:
                        proba[:,i] = np.maximum(proba[:,i],proba_[:,cc])
                    elif ii[1]==i:
                        proba[:,i] = np.maximum(proba[:,i],1-proba_[:,cc])
                    cc += 1
            """
        else:
            hb = self.decision_function(X)
            proba = sigmoid(hb)

        if proba.ndim>1:
            proba = (proba.T/proba.sum(axis=1)).T

        if self.return_y:
            if self.label_binarizer is None:
                return proba, None
            else:
                return proba, self.label_binarizer.inverse_transform(hb)
        else:
            return proba



class WeightedELMClassifier(ELMClassifier):
    def __init__(self, n_hidden=100, C=1.0, batch_size=None, fit_intercept=False, ovo=False, classes=None, activation_func='sigmoid', random_projection=True, return_y=False, random_state=None):
        super(WeightedELMClassifier, self).__init__(n_hidden=n_hidden, C=C, batch_size=batch_size, fit_intercept=fit_intercept, ovo=ovo, classes=classes, activation_func=activation_func, random_projection=random_projection, random_state=random_state, return_y=return_y)

    def fit(self, X, y):
        yc = Counter(y)
        sample_weight = np.empty(X.shape[0])
        #average_yc = np.mean(yc.values())
        for yy in yc:
            #if yc[yy]>average_yc:
            #    sample_weight[y==yy] = 1./np.sqrt(yc[yy])
            #else:
            #    sample_weight[y==yy] = (np.sqrt(5)-1)/2/np.sqrt(yc[yy])
            sample_weight[y==yy] = 1./np.sqrt(yc[yy])
        return super(WeightedELMClassifier, self).fit(X, y, sample_weight=sample_weight/sample_weight.sum())


class SSELMClassifier(ELMClassifier):
    def __init__(self, n_hidden=100, C=1.0, lambda_=1.0, activation_func='sigmoid', matlab_code_path=None, classes=None, random_projection=True, random_state=None):
        super(SSELMClassifier, self).__init__(n_hidden=n_hidden, C=C, batch_size=None, fit_intercept=False, activation_func=activation_func, classes=classes, random_projection=random_projection, random_state=random_state)
        self.lambda_ = self.lambda_
        self.eng = None
        self.L = None
        #self.model_matlab = None
        if matlab_code_path is None:
            self.matlab_code_path = None
        else:
            self.matlab_code_path = os.path.normpath(matlab_code_path)

    def start_matlab_connection(self):
        if self.matlab_code_path is not None:
            if self.eng is None:
                self.eng = matlab.engine.start_matlab()
                self.eng.addpath(self.matlab_code_path, nargout=0)
        else:
            self.eng = None

    def close_matlab_connection(self):
        if self.eng is not None:
            self.eng.exit()
        self.eng = None

    def compute_graph_laplacian(self, X, params):
        self.start_matlab_connection()
        self.L = self.eng.laplacian(params, matlab.double(X.tolist()), nargout=1)

    def fit(self, X, y):
        self.fitted_ = False
        self.random_state_ = check_random_state(self.random_state)
        X, y = self._validate_X_y(X, y)

        if self.matlab_code_path is None:
            raise NotImplementedError('No Python implementation for SSELM yet.')
            """
            N, dx = X.shape
            Nu = np.sum(np.isnan(y))
            Nl = N-Nu
            self.b = self.random_state_.uniform(size=self.n_hidden)*2-1
            self.W = self.random_state_.uniform(size=(dx,self.n_hidden))*2-1

            if self.activation_func == 'sigmoid':
                H = sigmoid(dot(X,self.W)+self.b)
            else:
                raise NotImplementedError('activation_func="%s" is not implemented.')
            C = np.eye(N,dtype=float)*self.C
            C[range(Nl,N),range(Nl,N)] = 0.
            L = ???
            if Nl>self.n_hidden:
                self.beta = multi_dot((inv(np.eye(self.n_hidden,dtype=float)+multi_dot((H.T,C+self.lambda_*L,H))),H.T,C,y))
            else:
                self.beta = multi_dot(H.T,inv(np.eye(N,dtype=float)+multi_dot((C+self.lambda_*L,H,H.T))),C,y)
            """
        else:
            unlabeled_id = np.isnan(y)
            labeled_id = np.logical_not(unlabeled_id)
            self.start_matlab_connection()
            params = {'NN':50,'GraphWeights':'binary','GraphDistanceFunction':'euclidean',
                        'LaplacianNormalize':1,'LaplacianDegree':5,
                        'NoDisplay':1,'Kernel':'sigmoid','random_state':self.random_state,'random_projection':self.random_projection,
                        'NumHiddenNeuron':self.n_hidden,'C':self.C,'lambda':self.lambda_}
            if self.L is None:
                L = self.compute_graph_laplacian(X, params)
            else:
                L = self.L
            import scipy.io as sio
            sio.savemat('bb.mat',{'paras':params,'X':X,'Xl':X[labeled_id,:],'Yl':y[labeld_id],'Xu':X[unlabeled_id,:],'L':L})
            model_matlab = self.eng.sselm(matlab.double(X[labeled_id,:].tolist()),matlab.double(y[labeled_id].tolist()),
                    matlab.double(X[unlabeled_id,:].tolist()), L, params, nargout=1)
            self.W = self.model_matlab._data['InputWeight']
            self.b = self.model_matlab._data['InputBias']
            self.beta = self.model_matlab._data['OutputWeight']

        self.fitted_ = True
        return self


class ELMAutoEncoderClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hiddens, Cs, reg_type='l2',output_layer=None, SSELM_lambda_=1., sigparas=1., sigparas1=1., matlab_code_path=None, random_state=None):
        self.n_hiddens = n_hiddens
        self.Cs = Cs
        self.output_layer = output_layer
        self.SSELM_lambda_ = SSELM_lambda_
        if type(sigparas)==list:
            self.sigparas = sigparas
        else:
            self.sigparas = [sigparas]*len(self.n_hiddens)
        if type(sigparas1)==list:
            self.sigparas1 = sigparas1
        else:
            self.sigparas1 = [sigparas1]*len(self.n_hiddens)
        if matlab_code_path is None:
            self.matlab_code_path = None
            self.layers = None
        else:
            self.matlab_code_path = os.path.normpath(matlab_code_path)
            self.layers_matlab = None
            self.eng = None
        #self.batch_size = batch_size
        #self.fit_intercept = fit_intercept
        #self.activation_func = activation_func
        self.reg_type = reg_type
        self.L = None
        self.random_state = random_state
        self.random_state_ = None
        self.fitted_ = False

    def start_matlab_connection(self):
        if self.matlab_code_path is not None:
            if self.eng is None:
                self.eng = matlab.engine.start_matlab()
                self.eng.addpath(self.matlab_code_path, nargout=0)
        else:
            self.eng = None

    def close_matlab_connection(self):
        if self.eng is not None:
            self.eng.exit()
        self.eng = None

    def _validate_X(self, X):
        if len(X.shape)==1:
            raise ValueError('X should be a 2-dimensional array.')
        #    if one feature:
        #        X = X.reshape(1,-1)
        #    else:  # one sample
        #        X = X.reshape(-1,1)
        if X.shape[0]==0:
            raise ValueError('Empty samples.')
        if X.shape[1]==0:
            raise ValueError('0 feature(s) (shape=(3, 0)) while a minimum of %d is required.'%(1,))
        return as_float_array(check_array(X))

    def _validate_X_y(self, X, y):
        return self.output_layer._validate_X_y(X,y)

    def compute_graph_laplacian(self, X, params):
        self.start_matlab_connection()
        import scipy.io as sio
        sio.savemat('aa.mat',{'paras':params,'X':X})
        self.L = self.eng.laplacian(params, matlab.double(X.tolist()), nargout=1)

    def fit(self, X, y=None):
        self.reg_type = self.reg_type.lower()
        self.fitted_ = False
        self.random_state_ = check_random_state(self.random_state)
        if self.output_layer is None or y is None:
            X = self._validate_X(X)
        else:
            X, y = self._validate_X_y(X, y)

        if self.matlab_code_path is None:
            # our python translation of the original ELM-Autoencoder in Matlab
            hidden_layer_num = len(self.n_hiddens)
            self.layers = []
            X = X.T
            dx, N = X.shape
            n_layers = np.r_[dx,self.n_hiddens]

            for i in range(hidden_layer_num):
                W = self.random_state_.rand(n_layers[i+1],n_layers[i])*2.-1.
                if n_layers[i+1] > n_layers[i]:
                    W = orth(W)
                else:
                    W = orth(W.T).T
                b = orth(self.random_state_.rand(n_layers[i+1],1)*2-1).ravel()

                H = (dot(W,X).T+b).T
                #print('AutoEncorder Max Val %f Min Val %f',H.max(),H.min())
                H = sigmoid(self.sigparas1[i]*H)

                self.layers.append({})
                self.layers[-1]['W'] = W
                self.layers[-1]['b'] = b
                self.layers[-1]['n_hidden'] = n_layers[i+1]
                self.layers[-1]['sigpara'] = self.sigparas[i]
                self.layers[-1]['sigpara1'] = self.sigparas1[i]
                if n_layers[i+1]==n_layers[i]:
                    C = dot(H,X.T)
                    _,_,v1 = svd(dot(C.T,C))
                    u2,_,_ = svd(dot(C,C.T))
                    self.layers[-1]['beta'] = dot(u2,v1)
                else:
                    if self.Cs[i] == 0:
                        self.layers[-1]['beta'], _, _, _ = lstsq(H.T,X.T)
                    elif self.reg_type=='l2':
                        rho = 0.05
                        rhohats = np.mean(H,axis=1)
                        KLsum = np.sum(rho*np.log(rho/rhohats)+(1.-rho)*np.log((1.-rho)/(1.-rhohats)))
                        Hsquare =  dot(H,H.T)
                        HsquareL = np.diag(np.max(Hsquare,axis=1))
                        self.layers[-1]['beta'] = multi_dot((inv((np.eye(H.shape[0])*KLsum+HsquareL)*self.Cs[i]+Hsquare),H,X.T))
                    elif self.reg_type=='l1':
                        tol = 1e-3
                        """
                        beta_ = Variable(X.shape[0],H.shape[0])
                        prob = Problem(Minimize(norm(beta_*H-X,'fro')+norm(beta_,1)*self.Cs[i]))
                        prob.solve(solver=SCS,use_indirect=False,eps=tol)#,verbose=True)
                        self.layers[-1]['beta'] = beta_.value.getA().T
                        """
                        lasso = Lasso(alpha=self.Cs[i]/H.shape[1], fit_intercept=False, precompute='auto', max_iter=3000,
                                tol=tol, warm_start=False, random_state=self.random_state*2, selection='random')
                        lasso.fit(H.T,X.T)
                        self.layers[-1]['beta'] = lasso.coef_.T
                    else:
                        raise NotImplementedError('Regularization type "%s" is not implemented.'%self.reg_type)
                H = dot(self.layers[-1]['beta'],X)

                if n_layers[i+1]==n_layers[i]:
                    X = H
                else:
                    #print('Layered Max Val %f Min Val %f',H.max(),H.min())
                    X = sigmoid(self.sigparas[i]*H)

            if self.output_layer is not None and y is not None:
                self.output_layer.fit(X.T,y)
                """
                self.layers.append({})
                if np.any(np.isnan(y)):  # semi-supervised ELM
                    nonnan_ids = np.logical_not(np.isnan(y))
                    Xl = X[:,nonnan_ids].T
                    yl = y[nonnan_ids]
                    Xu = X[:,np.logical_not(nonnan_ids)].T
                    if self.L is None:
                        L = self.compute_graph_laplacian(X.T, {'NN':50,'GraphWeights':'binary','GraphDistanceFunction':'euclidean',
                            'LaplacianNormalize':1,'LaplacianDegree':5})
                    else:
                        L = self.L
                    if Nl>self.n_hidden:
                        self.beta = multi_dot((inv(np.eye(self.n_hidden,dtype=float)+multi_dot((H.T,C+self.SSELM_lambda_*L,H))),H.T,C,y))
                    else:
                        self.beta = multi_dot(H.T,inv(np.eye(N,dtype=float)+multi_dot((C+self.SSELM_lambda_*L,H,H.T))),C,y)

                else:  # normal ELM
                    if self.Cs[hidden_layer_num] == 0:
                        self.layers[-1]['beta'], _, _, _ = lstsq(X.T,y.T)
                    else:
                        self.layers[-1]['beta'] = multi_dot((inv(np.eye(X.shape[0])/self.Cs[hidden_layer_num]+dot(X,X.T)),X,y.T))
                """

        else: # call the original ELM-Autoencoder in Matlab
            self.start_matlab_connection()
            matlab_X = matlab.double(X.tolist())
            matlab_y = matlab.double(y.tolist())
            matlab_n_hiddens = matlab.double(list(self.n_hiddens))
            matlab_Cs = matlab.double(list(self.Cs))
            matlab_sigparas = matlab.double(list(self.sigparas))
            matlab_sigparas1 = matlab.double(list(self.sigparas1))
            #import scipy.io as sio
            #sio.savemat('aa.mat',{'X':X,'y':y,'n_hiddens':self.n_hiddens,'Cs':self.Cs,'sigparas':self.sigparas,'sigparas1':self.sigparas1})
            self.layers_matlab = self.eng.elm_autoencoder_train(self.random_state,matlab_X,matlab_y,
                matlab_n_hiddens,matlab_Cs,matlab_sigparas,matlab_sigparas1,self.output_layer is not None,nargout=1)

        self.fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)

    def transform(self, X):
        return self.decision_function(X)

    def _transform_X(self, X):
        X = X.T
        hidden_layer_num = len(self.n_hiddens)

        for i in range(hidden_layer_num):
            H = dot(self.layers[i]['beta'],X)
            if i==0:
                n_hidden = X.shape[0]
            else:
                n_hidden = self.layers[i-1]['n_hidden']
            if n_hidden == self.layers[i]['n_hidden']:
                X = H
            else:
                X = sigmoid(dot(self.layers[i]['sigpara'],H))
        return X.T

    def decision_function(self, X):
        if not self.fitted_:
            raise ValueError('This ELMAutoEncoderClassifier instance is not fitted yet.')
        X = self._validate_X(X)
        if self.matlab_code_path is None:
            X = self._transform_X(X)
            if self.output_layer is None:
                yy = X
            else:
                yy = self.output_layer.decision_function(X)
        else:
            matlab_X = matlab.double(X.tolist())
            yy = self.eng.elm_autoencoder_test(self.layers_matlab, matlab_X, nargout=1)
            yy = np.array(list(yy._data)).T
        return yy

    def predict(self, X):
        if not self.fitted_:
            raise ValueError('This ELMAutoEncoderClassifier instance is not fitted yet.')
        X = self._transform_X(self._validate_X(X))
        if self.output_layer is None:
            raise TypeError('self.output_layer is None.')
        y = self.output_layer.predict(X)
        if y.ndim>1:
            y = y[:,1]
        return y

    def predict_proba(self, X, return_y=False):
        if not self.fitted_:
            raise ValueError('This ELMAutoEncoderClassifier instance is not fitted yet.')
        X = self._transform_X(self._validate_X(X))
        if self.output_layer is None:
            raise TypeError('self.output_layer is None.')
        yp = self.output_layer.predict_proba(X)
        return yp

    #def score(self, X, y):
    #    nonan_ids = np.logical_not(np.isnan(y))
    #    if self.problem_type == 'classification':
    #        return metrics.accuracy_score(y[nonan_ids], self.predict(X[nonan_ids,:]))
    #    else:
    #        return -metrics.mean_squared_error(y[nonan_ids], self.predict(X[nonan_ids,:]))


class VigilanceELMAutoEncoder(BaseEstimator, ClassifierMixin):
    def __init__(self, channel_num, sig_length, spec_length, sig_n_hidden=50, spec_n_hidden=50, n_hiddens=[], sig_C=0.1, spec_C=0.1, Cs=[0.1], sigparas=1., sigparas1=1.,
            lr=0.01, mc=0.9, max_epoch_num=50, matlab_code_path=None, verbose=False, random_state=None,to_transpose=False):#, classes_=None
        self.channel_num = channel_num
        self.sig_length = sig_length
        self.spec_length = spec_length
        self.sig_n_hidden = sig_n_hidden
        self.spec_n_hidden = spec_n_hidden
        self.n_hiddens = n_hiddens
        self.sig_C = sig_C
        self.spec_C = spec_C
        self.Cs = Cs
        self.sigparas = sigparas
        self.sigparas1 = sigparas1
        self.lr = lr
        self.mc = mc
        self.max_epoch_num = max_epoch_num
        self.matlab_code_path = matlab_code_path
        self.verbose = verbose
        self.random_state = random_state
        self.to_transpose = to_transpose
        self.fitted_ = False

    def fit(self, sigs_specs, vigs):
        if self.to_transpose:
            sigs_specs = np.transpose(sigs_specs,(1,0,2))
        # sigs_specs: channel_num x seg_num x (sig_length+spec_length)
        if sigs_specs.shape[0]!=self.channel_num:
            raise ValueError('sigs_specs.shape[0](%d) != channel_num(%d)'%(sigs_specs.shape[0],self.channel_num))
        if sigs_specs.shape[2]!=self.sig_length+self.spec_length:
            raise ValueError('sigs_specs.shape[2](%d) != sig_length(%d) + spec_length(%d)'%(sigs_specs.shape[2],self.sig_length,self.spec_length))

        self.fitted_ = False
        #self.random_state_ = check_random_state(self.random_state)  # done in the component classifiers
        self.sig_elmaes = [ELMAutoEncoderClassifier([self.sig_n_hidden], [self.sig_C], sigparas=1, sigparas1=1, reg_type='l2',
                matlab_code_path=self.matlab_code_path, random_state=self.random_state+i) for i in range(self.channel_num)]
        self.spec_elmaes = [ELMAutoEncoderClassifier([self.spec_n_hidden], [self.spec_C], sigparas=1, sigparas1=1, reg_type='l2',
                matlab_code_path=self.matlab_code_path, random_state=self.random_state+self.channel_num+i) for i in range(self.channel_num)]
        self.later_elmae = ELMAutoEncoderClassifier(self.n_hiddens, self.Cs[:-1], reg_type='l2',
                output_layer=SoftLogisticRegression(C=self.Cs[-1], learning_rate=0.01, momentum=0.9, max_iter=200,
                random_state=self.random_state, tol=1e-4, verbose=False),
                sigparas=self.sigparas, sigparas1=self.sigparas1, matlab_code_path=self.matlab_code_path,# classes_=classes_,
                random_state=self.random_state+2*self.channel_num)

        ## first fit_transform sig_elmaes and spec_elmaes
        seg_num = sigs_specs.shape[1]
        #X = np.empty((seg_num, (self.sig_n_hidden+self.spec_n_hidden)*self.channel_num))
        X = np.empty((seg_num, self.spec_n_hidden*self.channel_num))
        for i in range(self.channel_num):
            if self.verbose:
                print('channel %d/%d'%(i+1,self.channel_num))
            #X[:,self.sig_n_hidden*i:self.sig_n_hidden*(i+1)] =\
            #        self.sig_elmaes[i].fit_transform(sigs_specs[i,:,:self.sig_length], None)
            X[:,self.spec_n_hidden*i:self.spec_n_hidden*(i+1)] =\
                    self.spec_elmaes[i].fit_transform(sigs_specs[i,:,self.sig_length:])

        ## then fit later_elmae
        self.later_elmae.fit(X, vigs)

        self.fitted_ = True
        return self

    def predict(self, sigs_specs):
        return self.predict_proba(sigs_specs)

    def predict_proba(self, sigs_specs):
        if self.to_transpose:
            sigs_specs = np.transpose(sigs_specs,(1,0,2))
        # sigs_specs: channel_num x seg_num x (sig_length+spec_length)
        if sigs_specs.shape[0]!=self.channel_num:
            raise ValueError('sigs_specs.shape[0](%d) != channel_num(%d)'%(sigs_specs.shape[0],self.channel_num))
        if sigs_specs.shape[2]!=self.sig_length+self.spec_length:
            raise ValueError('sigs_specs.shape[2](%d) != sig_length(%d) + spec_length(%d)'%(sigs_specs.shape[2],self.sig_length,self.spec_length))

        ## first transform using sig_elmaes and spec_elmaes
        seg_num = sigs_specs.shape[1]
        #X = np.empty((seg_num, (self.sig_n_hidden+self.spec_n_hidden)*self.channel_num))
        X = np.empty((seg_num, self.spec_n_hidden*self.channel_num))
        for i in range(self.channel_num):
            #X[:,self.sig_n_hidden*i:self.sig_n_hidden*(i+1)] =\
            #        self.sig_elmaes[i].transform(sigs_specs[i,:,:self.sig_length])
            X[:,self.spec_n_hidden*i:self.spec_n_hidden*(i+1)] =\
                    self.spec_elmaes[i].transform(sigs_specs[i,:,self.sig_length:])

        yp = self.later_elmae.predict_proba(X)
        if type(self.later_elmae.output_layer)==SoftLogisticRegression:
            return yp[:,1]
        else:
            return yp[:,0]

    #def score(self, sigs_specs, vigs):
    #    nonan_ids = np.logical_not(np.isnan(vigs))
    #    return -metrics.mean_squared_error(vigs[nonan_ids], self.predict(sigs_specs[:,nonan_ids,:]))


## deprecated!
"""
class OSELMClassifier(ELMClassifier):
    def __init__(self, n_hidden=100, C=1.0, batch_size=1, activation_func='sigmoid', classes=None, random_state=None):
        super(OSELMClassifier, self).__init__(n_hidden=n_hidden,C=C,activation_func=activation_func,classes=classes,random_state=random_state)
        self.P = None  # P = (H'*H+C*I)^-1
        self.batch_size = batch_size
        self.random_state_ = check_random_state(self.random_state)

    def fit(self, X, y):
        self.fitted_ = False
        if self.batch_size <= 0:
            raise ValueError('batch_size must be larger than 0.')
        N = X.shape[0]
        N_ = N-self.n_hidden
        if N_>0:
            if N_%self.batch_size==0:
                batches = [self.n_hidden]+[self.batch_size]*(N_//self.batch_size)
            else:
                batches = [self.n_hidden]+[self.batch_size]*(N_//self.batch_size)+[N_%self.batch_size]
        else:
            batches = [N]
        #shuffled_id = list(range(N))
        #self.random_state_.shuffle(shuffled_id)
        #X = X[shuffled_id,:]
        #y = y[shuffled_id]

        for i in range(len(batches)):
            start_n = sum(batches[:i])
            end_n = sum(batches[:i+1])
            self.fit_part(X[start_n:end_n,:],y[start_n:end_n],continue_fit=i!=0)

        self.fitted_ = True

        return self

    def fit_part(self, X, y, continue_fit=True):
        #recursive least square
        self.fitted_ = False
        X, y = self._validate_X_y(X, y)
        N, dx = X.shape

        if self.activation_func == 'sigmoid':
            if not continue_fit or self.P is None:
                self.b = self.random_state_.uniform(size=self.n_hidden)*2-1
                self.W = self.random_state_.uniform(size=(dx,self.n_hidden))*2-1
            H = sigmoid(dot(X,self.W)+self.b)
        else:
            raise NotImplementedError('activation_func="%s" is not implemented.')

        if not continue_fit or self.P is None:
            #if N<self.n_hidden:
            #    raise ValueError('Number of samples (N=%d) cannot be smaller than hidden neuron number (n_hidden=%d) in the initial fit.'%(N,self.n_hidden))
            self.P = inv(dot(H.T,H)+self.C*np.eye(self.n_hidden))
            self.beta = multi_dot((self.P,H.T,y))
        else:
            if N==1:
                h = H.ravel()
                hht = np.outer(h,h)
                self.P = self.P - multi_dot((self.P,hht,self.P))/(1.+(self.P*hht).sum())
            else:
                PHt = dot(self.P,H.T)
                self.P = self.P - multi_dot((PHt,inv(dot(H,PHt)+np.eye(N)),H,self.P))

            self.beta = self.beta + dot(dot(self.P,H.T),y-dot(H,self.beta))

        return self

    def predict(self, X):
        return super(OSELMClassifier, self).predict(X,allow_not_fitted=True)
"""


if __name__=='__main__':
    import copy
    import pdb
    import timeit
    from sklearn import datasets, preprocessing, cross_validation

    #check_estimator(ELMClassifier)
    #check_estimator(SequentialELMClassifier)
    random_state = 1
    np.random.seed(random_state)
    X, y = datasets.make_classification(n_samples=2000, n_features=20, n_informative=3, n_redundant=2, n_repeated=0, n_classes=3, n_clusters_per_class=2,random_state=random_state)
    train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, train_size=0.8,random_state=random_state)
    scaler = preprocessing.StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    hnn = 50
    C = 1
    elmclf = ELMClassifier(C=C, n_hidden=hnn, fit_intercept=False,random_state=random_state)
    oselmclf = ELMClassifier(C=C, n_hidden=hnn, fit_intercept=False,batch_size=300, random_state=random_state)
    selmclf = SequentialELMClassifier([elmclf,copy.deepcopy(elmclf)])
    st_elm = timeit.default_timer()
    elmclf.fit(train_X,train_y)
    et_elm = timeit.default_timer()
    st_oselm = timeit.default_timer()
    oselmclf.fit(train_X,train_y)
    et_oselm = timeit.default_timer()
    st_selm = timeit.default_timer()
    selmclf.fit(train_X,train_y)
    et_selm = timeit.default_timer()
    print('ELM and OS-ELM are consistent: %s.'%np.allclose(elmclf.beta,oselmclf.beta))
    print('ELM time: %gs'%(et_elm-st_elm,))
    print('OS-ELM time: %g'%(et_oselm-st_oselm,))
    print('S-ELM time: %g'%(et_selm-st_selm,))
    print('ELM acc: %g'%elmclf.score(test_X,test_y))
    print('OS-ELM acc: %g'%oselmclf.score(test_X,test_y))
    print('S-ELM acc: %g'%selmclf.score(test_X,test_y))

