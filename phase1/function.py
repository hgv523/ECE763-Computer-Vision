# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:03:09 2020

@author: cheng
"""
import numpy as np
import cv2
from scipy.stats import norm
import math
import scipy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import special
import random
#from sklearn.model_selection import StratifiedKFold


def single_Gaussian(x):
    mean=[]
    sigma=[]
    for i in range(10800):
        mean_i, sigma_i = norm.fit(x[:,i])
        mean.append(mean_i)
        sigma.append(sigma_i)
    mean = np.array(mean)
    sigma = np.array(sigma)
    return mean, sigma
    
def parameters(predict_label, test_label, n):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    predict_label = predict_label.reshape(1,n)
    test_label = test_label.reshape(1,n)
    for i in range (n):
        if predict_label[0][i] == 1 and test_label[0][i]==1:
            TP = TP + 1
        if predict_label[0][i] == 1 and test_label[0][i]==0:
            FP = FP + 1
        if predict_label[0][i] == 0 and test_label[0][i]==1:
            FN =FN + 1
        if predict_label[0][i] == 0 and test_label[0][i]==0:
            TN = TN + 1
    print("TP=",TP,"FN=",FN,"FP=",FP,"TN=",TN)
    print("false positive rate:", FP/(FP+TN))
    print("false negative rate:", FN/(TP+FN))
    print("misclassification rate:", (FP+FN)/n)
    
def toImage(x):
    x = x/max(x)
    y = x.reshape((3,3600))
    b = y[0,:].reshape((60,60))
    g = y[1,:].reshape((60,60))
    r = y[2,:].reshape((60,60))
    y = cv2.merge([r,g,b])
    f1=plt.imshow((y * 255 ).astype(np.uint8))
    plt.savefig('./model1/figure.png')
    return f1

def ROCcurve(test_data,predict_score):
    print(test_data[:,-1].shape)
    print(predict_score.shape)
    fpr,tpr,threshold = roc_curve(test_data[:,-1],predict_score)
    roc_auc = auc(fpr,tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("./model1/ROC.png")
    
def log_N_Gaussian(data,mu,sigma):
    predict_score = []
    N = data.shape[0]
    n = data.shape[1]
    for i in range(N):
        p = -n * 0.5 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma)) - 0.5 * np.dot((data[i, :] - mu) / sigma,
                                                                                      (data[i, :] - mu).T) 
        predict_score.append(p)
    predict_score = np.array(predict_score)
    
    return (predict_score)

def p_score(logpdf,lamda):
    logpdf = np.array(logpdf)
    K = logpdf.shape[0]
    N = logpdf.shape[1]
    log_lamda = np.log(lamda).reshape((K,1))
    log_p = logpdf + log_lamda
    p_s = np.zeros((K,N))
    for i in range(K):
        for j in range(N):
            p_s[i,j] = np.exp(log_p[i,j]-scipy.special.logsumexp(log_p[:,j]))
    return p_s

def faverage(data, score):
    mp = np.dot(score, data)
    k = score.shape[0]
    D = data.shape[1]
    mu = np.zeros((k,D))
    for i in range(k):
        mu[i] = mp[i,:]/np.sum(score[i,:])
    return mu

def fsigma(data, mu):
    K = mu.shape[0]
    sigma = np.zeros((K,10800))
    N = data.shape[0]
    D = data.shape[1]
    for i in range(K):
        for j in range(D):
            sigma[i,j] = np.sum(np.square((data[:,j]-mu[i,j])))/N
    return sigma

def flamda(score, lamda_o):
    lamda = np.sum(score,axis=1)/score.shape[1]
    return lamda

def fL(px,pPi):
    sub = np.sum(pPi*px)
    logsub = np.log(sub)
    curL = np.sum(logsub)
    return curL

def stop_iter(threshold,preL,curL):
    return np.sum(np.abs(curL-preL)) < threshold


def EMalgorithm(data, k, flag, f):
    # np.random.seed(1)
    # flag=0 Gaussian, flag=1 T-distribution(pf=10), flag=2 f-Factor Analyzers
    # initialize parameters
    N = data.shape[0]
    D = data.shape[1]
    mu = np.random.rand(k, D)
    sigma = np.random.rand(k, D)
    theta = np.random.rand(D, f)

    lamda_a = np.random.random(k)
    lamda = lamda_a / np.sum(lamda_a)

    # stop EM
    preL = -np.inf
    thresh = 1e-13

    i = 0

    while (True):
        # Calculate logpdf

        # M-step
        logpdf = []
        for m in range(k):
            if flag == 0:
                logpdf.append(log_N_Gaussian(data, mu[m], sigma[m]))
            elif flag == 1:
                logpdf.append(Tlogpdf(data, mu[m], sigma[m]))
            elif flag == 2:
                Eh, Ehh = fh(data, theta, mu[m], sigma[m])
                logpdf.append(FactorAnalysis(data, mu[m], sigma[m], theta, Eh))

        # Calculate score
        score = p_score(logpdf, lamda)

        # E-step
        mu = faverage(data, score)
        sigma = fsigma(data, mu)

        if flag == 2:
            for m in range(k):
                theta = ftheta(data, mu[i], sigma[i], theta, Eh, Ehh)
        inv = []
        for m in range(k):
            inv.append(np.diag(1 / sigma[m]))  # inverse of Modelk
        lamda_t = flamda(score, lamda)
        if stop_iter(thresh, lamda, lamda_t):
            break
        lamda = lamda_t
        i = i + 1
        print(i)
        print(lamda)

    if flag == 2:
        return (mu, sigma, lamda, theta, Eh)
    else:
        return (mu, sigma, lamda)
    
def test(validation, k,l, mu,sigma, lamda, nmu, nsigma, nlamda, flag, theta=0, h=0, ntheta=0, nh=0):
    facepdf = []
    nonfacepdf = []
    if flag==0:
        for i in range(k):
            facepdf.append(log_N_Gaussian(validation[:,:-1], mu[i], sigma[i]))
        for i in range(l):
            nonfacepdf.append(log_N_Gaussian(validation[:,:-1], nmu[i], nsigma[i]))
    elif flag==1:
        for i in range(k):
            facepdf.append(Tlogpdf(validation[:,:-1], mu[i], sigma[i]))
        for i in range(l):
            nonfacepdf.append(Tlogpdf(validation[:,:-1], nmu[i], nsigma[i]))
    elif flag==2:
        for i in range(k):
            facepdf.append(FactorAnalysis(validation[:,:-1], mu[i], sigma[i], theta, h))
        for i in range(l):
            nonfacepdf.append(FactorAnalysis(validation[:,:-1], nmu[i], nsigma[i], ntheta, nh))

    face = np.log(lamda).reshape((k, 1))+facepdf
    nonface = np.log(nlamda).reshape((l, 1))+nonfacepdf
    final_face = np.zeros((1, face.shape[1]))
    final_nonface = np.zeros((1, nonface.shape[1]))
    for i in range(face.shape[1]):
        final_face[0,i] = scipy.special.logsumexp(face[:,i])
        final_nonface[0,i] = scipy.special.logsumexp(nonface[:,i])

    final = np.vstack((final_face, final_nonface))
    log_p = np.zeros((1,face.shape[1]))
    for i in range(face.shape[1]):
        log_p[0,i] = final[0,i]-scipy.special.logsumexp(final[:,i])
    p = np.exp(log_p)
    label = np.zeros((1,face.shape[1]))
    for i in range(face.shape[1]):
        if p[0,i]>0.5:
            label[0,i] = 1
    label = label.astype(int)
    parameters(label, validation[:,-1], 200)
    p = np.array(p)
    ROCcurve(validation, p.T)
    
def Tlogpdf(data,mu,sigma):
    v=10
    predict_score = []
    N = data.shape[0]
    D = data.shape[1]
    for i in range(N):
        x1 = np.log(scipy.special.gammaln((v+D)/2))
        x2 = -0.5*D*np.log(v*np.pi)
        x3 = -0.5*np.sum(np.log(sigma))
        x4 = -np.log(scipy.special.gammaln(v/2))
        x5 = -0.5*(v+D)*np.log(1+np.dot((data[i,:]-mu)/sigma, (data[i,:]-mu).T)/v)
        p = x1+x2+x3+x4+x5
        predict_score.append(p)
    predict_score = np.array(predict_score)
    return(predict_score)

def ftheta(data, mu, sigma, theta, Eh, Ehh):
    N = data.shape[0]
    D = data.shape[1]
    x1 = 0
    x2 = 0
    for i in range(N):
        eh = np.array(Eh[:,i])
        eh = eh.reshape(1, Eh.shape[0])
        x1 = x1 + np.dot(eh.T, eh)
        d = data[i,:]-mu
        d = d.reshape(D,1)
        x2 = x2 + np.dot(d, eh)
    theta_n = np.dot(x2, np.linalg.inv(x1))
    return theta_n

def FactorAnalysis(data, mu, sigma, theta, h):
    predict_score = []
    N = data.shape[0]
    D = data.shape[1]
    K = theta.shape[1]
    for i in range(N):
        x1 = -D*0.5*np.log(2*np.pi)-0.5*np.sum(np.log(sigma))
        x2 = (data[i,:]-mu).T-np.dot(theta,h[:,i])
        x3 = -0.5*np.dot(x2.T/sigma,x2)
        p = x1+x3
        predict_score.append(p)
    predict_score = np.array(predict_score)
    return(predict_score)
            
    
def fh(data, theta, mu, sigma):
    sigma = np.array(sigma)
    Eh = np.zeros((theta.shape[1],data.shape[0]))
    Ehh = []
    for j in range(data.shape[0]):
        x1 = np.dot(np.dot(theta.T,np.diag(1/sigma)),theta) + np.eye(theta.shape[1])
        Eh[:,j] = np.dot(np.dot(np.dot(x1, theta.T), np.diag(1/sigma)),(data[j,:]-mu).T)
        x2 = np.dot(Eh[:,j], Eh[:,j].T)
        Ehh.append(x1+x2)
    Ehh = np.array(Ehh)
    return Eh, Ehh






        
    
