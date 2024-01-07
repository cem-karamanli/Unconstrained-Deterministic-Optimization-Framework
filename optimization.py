#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 14:24:19 2024

@author: Cem Karamanli
"""

import numpy as np
import scipy.io as sio
from scipy.special import expit
from scipy.linalg import qr
from pylab import *

# Optimization framework for solving unconstrained deterministic optimization problems.
# The problems considered are logistic regression, quadratic and rosenbrock functions.
# The methods implemented are "SD", "Newton", "BFGS_Bk", "BFGS_Hk", "LBFGS".


### USER: CHANGE BELOW 

config = {
    #'path'           : 'Write your path here/',  # specify the path to the folder 
    # where the data for logistic regression problems are stored.
    'prob'           : "logReg",                # "logReg" or "rosenbrock" or "quad"
    'method'         : "SD",                    # "SD", "Newton", "BFGS_Bk", "BFGS_Hk", "LBFGS"
    'alpha'          : 1,                       # stepsize
    'num_of_iter'    : 1000,                    # number of iterations
    'is_linesearch'  : False,                   # if linesearch is going to be used. 
    #If yes, alpha becomes the starting stepsize for the linesearch algorithm
    
    # if logreg:
    'dataset_name'  : "mushroom",               # choose from: 
    #"a9a", "australian", "ijcnn", "ionosphere", "mushroom", "mnist"
    
    # if rosenbrock or quad, specify the problem dimension:
    'd'             : 10,
    
    # if quad, specify the condition number:
    'kappa'         : 10000,
    
    # if quad, specify a random seed which is used to create a random quadratic problem
    'seed'          : 7
}
    
### USER: CHANGE ABOVE

class Optimization():
    def __init__(self,config):
        self.config = config
        return
    
    def initialize(self):
        self.path = self.config['path']
        self.prob = self.config['prob']
        self.method = self.config['method']
        
        self.alpha = self.config['alpha']
        self.num_of_iter = self.config['num_of_iter']
        self.is_linesearch = self.config['is_linesearch']
        
        if self.prob == "logReg":
            self.dataset_name = self.config['dataset_name']
            y,z = self.get_data()
            # get the info from data:
            self.y = y.copy()
            self.z = z.copy()
            self.z = self.z.flatten()
            self.d = y.shape[1]
            self.N = y.shape[0]
            self.reg_param = 1/self.N
        elif self.prob == "rosenbrock":
            self.d = self.config['d']
        elif self.prob == "quad":
            self.d = self.config['d']
            self.kappa = self.config['kappa']
            self.seed = self.config['seed']
            np.random.seed(self.seed)
            # the quadratic problem is f(x) = 0.5x^TAx + x^Tb
            eigval = np.random.rand(self.d)   # create random eigenvalues
            rang = max(eigval) - min(eigval)   # transform the eigenvalues such that condition number is met
            factor = (self.kappa-1)/rang
            eigval *= factor
            eigval += 1-min(eigval)
            # create the A matrix by multiplying it with orthonormal matrices
            H = np.random.randn(self.d, self.d)
            Q, R = qr(H)
            self.A = np.matmul(Q.T,np.matmul(np.diag(eigval),Q))
            # alternative for creating the A matrix:
            #self.A = np.tril(np.random.rand(self.d,self.d))
            #self.A += np.identity(self.d) # to make sure it is sufficiently positive definite
            self.b = np.random.randn(self.d)
            # normalize self.A and self.b:
            self.A = self.A/np.linalg.norm(self.A, ord="fro")
            self.b = self.b/np.linalg.norm(self.b)
        else:
            print("Problem not implemented yet.")
            return
        
        if self.method == "LBFGS":
            self.m_lbfgs = 10
            self.yk_matrix = np.zeros((self.m_lbfgs,self.d))
            self.sk_matrix = np.zeros((self.m_lbfgs,self.d))
        elif self.method == "Newton" or self.method == "BFGS_Bk":
            self.Bk = np.identity(self.d)
        elif self.method == "BFGS_Hk":
            self.Hk = np.identity(self.d)
            
        # special initialization for rosenbrock function:
        if self.prob == "rosenbrock":
            if self.method == "BFGS_Bk":
                self.Bk = 100*np.identity(self.d)
            else:
                self.Hk = 0.01*np.identity(self.d)
        
        # other common parameters to initialize:
        self.k = 0
        self.x = np.zeros(self.d)
        self.grad = self.gradf(self.x)
        self.p = -self.alpha*self.grad
        self.grad_norm_hist = [];
        
    # getting y,z data of a given dataset for logReg:
    def get_data(self):
        filename = self.config['path']
        if self.dataset_name == "australian":
            filename += 'australian_scale.mat'
        elif self.dataset_name == "ionosphere":
            filename += 'ionosphere.mat'
        elif self.dataset_name == "mushroom":
            filename += 'mushroom.mat'
        elif self.dataset_name == "mnist":
            filename += 'MNIST.mat'
        elif self.dataset_name == "ijcnn":
            filename += 'ijcnn1.mat'
        elif self.dataset_name == "a9a":
            filename += 'a9a.mat'
            
        data = sio.loadmat(filename)
        y = data["X"]
        z = data["y"]
        if self.dataset_name == "mnist" or self.dataset_name == "ijcnn" or self.dataset_name == "a9a":
            # stored as sparse matrix, make it dense matrix:
            y = y.toarray()
        # include vector of ones to y matrix:
        y = np.c_[y, np.ones([y.shape[0],1])]
        return y, z
    
    def orthogonalityTest(self): # Test to see if p_k and the gradient make an acute angle, and an error between the two:
        res = np.dot(self.p_k,self.gradf(self.x))
        if res < 0:
            print("descent")
        else:
            print("not-descent")
        print(res/(np.linalg.norm(self.gradf(self.x))**2))
    
    def linesearch(self): # Armijo line search:
        gamma = 1
        k = 0
        condition = True
        fx = self.f(self.x)
        gradx = self.gradf(self.x)
        dp = np.dot(gradx,self.p_k)
        while condition:
            x_prop = self.x + gamma*self.alpha*self.p_k
            f_prop = self.f(x_prop)
            if f_prop < fx + 0.0001*gamma*self.alpha*dp:
                return gamma
            else:
                gamma /= 2
            k +=1
            if k >20: # quick fix
                # line search failed
                self.p_k = -gradx
                gamma = 0.001
                return gamma
            
            
    def chooseH0(self): # for LBFGS algorithm: choosing H0
        s_k = self.sk_matrix[-1,:]
        y_k = self.yk_matrix[-1,:]
        dp = np.dot(y_k,s_k)
        snorm = np.linalg.norm(s_k)
        ynorm = np.linalg.norm(y_k)
        if dp > (10**(-12))*snorm*ynorm:
            gamma_k = dp/(ynorm**2)
        else:
            gamma_k = 1
        H0 = gamma_k*np.identity(self.d)
        return H0
        
    def twoLoopRecursionFull(self,q,H0):  # for LBFGS algorithm: returns matrix-vector product of H_k and q
        alpha_vec = np.zeros(self.m_lbfgs)
        for i in range(self.m_lbfgs):
            i2 = i+1
            s_i = self.sk_matrix[-i2,:]
            y_i = self.yk_matrix[-i2,:]
            rho_i = 1/(np.dot(s_i,y_i))
            alpha_vec[-i2] = rho_i*np.dot(s_i,q)
            q -= alpha_vec[-i2]*y_i
        r = np.dot(H0,q)
        for i in range(self.m_lbfgs):
            s_i = self.sk_matrix[i,:]
            y_i = self.yk_matrix[i,:]
            rho_i = 1/(np.dot(s_i,y_i))
            beta = rho_i*np.dot(y_i,r)
            r += s_i*(alpha_vec[i] - beta)
        return r
    
    def twoLoopRecursion(self,q,H0): # for LBFGS algorithm: returns matrix-vector product of H_k and q
        m_gen = self.k
        lag = self.m_lbfgs - self.k
        if lag > 0:
            alpha_vec = np.zeros(m_gen)
            for i in range(m_gen):
                i2 = i+1
                s_i = self.sk_matrix[-i2,:]
                y_i = self.yk_matrix[-i2,:]
                rho_i = 1/(np.dot(s_i,y_i))
                alpha_vec[-i2] = rho_i*np.dot(s_i,q)
                q -= alpha_vec[-i2]*y_i
            r = np.dot(H0,q)
            for i in range(m_gen):
                i2 = i + lag
                s_i = self.sk_matrix[i2,:]
                y_i = self.yk_matrix[i2,:]
                rho_i = 1/(np.dot(s_i,y_i))
                beta = rho_i*np.dot(y_i,r)
                r += s_i*(alpha_vec[i] - beta)
            return r
        else:
            return self.twoLoopRecursionFull(q,H0)
        
    
    def updateSkYk(self,s_k,y_k): # for LBFGS algorithm: update sk_matrix and yk_matrix given new inputs
        yk_matrix = np.zeros((self.m_lbfgs,self.d))
        sk_matrix = np.zeros((self.m_lbfgs,self.d))
        for i in range(self.m_lbfgs-1):
            yk_matrix[i,:] = self.yk_matrix[i+1,:]
            sk_matrix[i,:] = self.sk_matrix[i+1,:]
        yk_matrix[-1,:] = y_k
        sk_matrix[-1,:] = s_k
        self.yk_matrix = yk_matrix
        self.sk_matrix = sk_matrix
            
    def updateBkBFGS(self,s_k,y_k): # for BFGS_Bk algorithm: update BFGS matrix Bk
        # use Sherman Morison Woodbury formula:
        s = np.reshape(s_k,(self.d,1))
        y = np.reshape(y_k,(self.d,1))
        denom1 = np.dot(s_k,np.dot(self.Bk,s_k))
        rho = 1/(np.dot(y_k,s_k))
        self.Bk += - (1/denom1)*np.matmul(np.matmul(self.Bk,s),np.matmul(s.T,self.Bk)) \
            + rho*np.matmul(y,y.T)
            
    def updateHkBFGS(self,s_k,y_k): # for BFGS_Hk algorithm: update BFGS matrix Hk
        # use usual formula for Hk update in BFGS:
        s = np.reshape(s_k,(self.d,1))
        y = np.reshape(y_k,(self.d,1))
        I = np.identity(self.d)
        rho = 1/np.dot(y_k,s_k)
        self.Hk = np.matmul(np.matmul(I - rho*np.matmul(s,y.T),self.Hk),I - rho*np.matmul(y,s.T)) \
            + rho*np.matmul(s,s.T)
        
        
    def LBFGSiter(self):
        self.grad_norm_hist.append(np.linalg.norm(self.grad))
        if self.k >= 1:
            H0 = self.chooseH0()
            q = self.grad.copy()
            self.p_k = -self.twoLoopRecursion(q,H0)
        else:
            self.p_k = -self.grad
        if self.is_linesearch == True:
            gamma = self.linesearch()
        else:
            gamma = 1
        s_k = self.alpha*gamma*self.p_k
        self.x += s_k
        new_grad = self.gradf(self.x)
        y_k = new_grad - self.grad
        self.grad = new_grad
        self.updateSkYk(s_k,y_k)
        self.k += 1
            
    def SDiter(self):
        self.grad_norm_hist.append(np.linalg.norm(self.grad))
        self.p_k = -self.grad
        if self.is_linesearch == True:
            gamma = self.linesearch()
        else:
            gamma = 1
        self.x += self.alpha*gamma*self.p_k
        self.grad = self.gradf(self.x)
        self.k += 1
        
    def BFGS_Bkiter(self):
        self.grad_norm_hist.append(np.linalg.norm(self.grad))
        self.p_k = -np.linalg.solve(self.Bk, self.grad)
        if self.is_linesearch == True:
            gamma = self.linesearch()
        else:
            gamma = 1
        s_k = self.alpha*gamma*self.p_k
        self.x += s_k
        new_grad = self.gradf(self.x)
        y_k = new_grad - self.grad
        self.grad = new_grad
        self.updateBkBFGS(s_k,y_k)
        self.k += 1
        
            
    def BFGS_Hkiter(self):
        self.grad_norm_hist.append(np.linalg.norm(self.grad))
        self.p_k = -np.dot(self.Hk, self.grad)
        if self.is_linesearch == True:
            gamma = self.linesearch()
        else:
            gamma = 1
        s_k = self.alpha*gamma*self.p_k
        self.x += s_k
        new_grad = self.gradf(self.x)
        y_k = new_grad - self.grad
        self.grad = new_grad
        self.updateHkBFGS(s_k,y_k)
        self.k += 1
        
    def Newtoniter(self):
        self.grad_norm_hist.append(np.linalg.norm(self.grad))
        self.p_k = -np.linalg.solve(self.Bk, self.grad)
        if self.is_linesearch == True:
            gamma = self.linesearch()
        else:
            gamma = 1
        s_k = self.alpha*gamma*self.p_k
        self.x += s_k
        new_grad = self.gradf(self.x)
        new_Hessian = self.hessf(self.x)
        self.grad = new_grad
        self.Bk = new_Hessian
        self.k += 1
        
    def run(self):
        self.initialize()
        if self.method == "SD":
            iterfunc = self.SDiter
        elif self.method == "Newton":
            iterfunc = self.Newtoniter
        elif self.method == "BFGS_Bk":
            iterfunc = self.BFGS_Bkiter
        elif self.method == "BFGS_Hk":
            iterfunc = self.BFGS_Hkiter
        elif self.method == "LBFGS":
            iterfunc = self.LBFGSiter
        else:
            print("Method not implemented yet.")
            return
        
        for i in range(self.num_of_iter):
            iterfunc()
        semilogy(self.grad_norm_hist)
            
    def f(self,x):
        if self.prob == "logReg":
            val = (1/self.N)*np.sum(np.log(1 + np.exp(-self.z*np.dot(self.y,x)))) + (self.reg_param/2) * np.dot(x,x)
            return val
        elif self.prob == "rosenbrock":
            val = 0
            a = self.d-1
            for i in range(a):
                val += 100*(x[i+1]- x[i]**2)**2 + (1-x[i])**2
            return val
        elif self.prob == "quad":
            val = 0.5*(np.dot(x,np.dot(self.A,x))) + np.dot(self.b,x)
            return val
        
    def gradf(self,x):
        if self.prob == "logReg":
            grad = (1/self.N)*np.dot(self.y.T,(1-expit(self.z*np.dot(self.y,x)))*(-self.z)) + self.reg_param * x
            return grad
        elif self.prob == "rosenbrock":
            grad = np.zeros(self.d)
            grad[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
            grad[self.d-1] = 200*(x[self.d-1] - x[self.d-2]**2)
            for i in range(self.d-1):
                grad[i] = 200*(x[i]-x[i-1]**2) -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i]);
            return grad
        elif self.prob == "quad":
            grad = np.dot(self.A,x) + self.b
            return grad
    
    def hessf(self,x):
        if self.prob == "logReg":
            h = (1-expit(self.z*np.dot(self.y,x)))*expit(self.z*np.dot(self.y,x))
            h_diag = np.diag(h)
            hess = (1/self.N)*np.dot(np.dot(self.y.T,h_diag),self.y) + self.reg_param*np.identity(self.d)
            return hess
        elif self.prob == "rosenbrock":
            hess = np.zeros((self.d,self.d))
            hess[0,0] = -400*(x[1] - x[0]**2) + 800*x[0]**2 + 2
            hess[self.d-1,self.d-1] = 200
            hess[0,1] = -400*x[0]
            hess[1,0] = hess[0,1]
            for i in range(1,self.d-1):
                hess[i,i] = 200 - 400*(x[i+1] - x[i]**2) + 800*x[i]**2 + 2
                hess[i,i+1] = -400*x[i]
                hess[i+1,i] = hess[i,i+1]
            return hess
        elif self.prob == "quad":
            hess = self.A
            return hess
        
    
        
optimization_obj = Optimization(config)
optimization_obj.run()

        
        
        
        
        
        
        