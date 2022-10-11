# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:37:28 2020

@author: cheng
"""

import numpy as np
import cv2
from function import single_Gaussian, parameters, toImage,ROCcurve, EMalgorithm, test
import math
import scipy
from sklearn.model_selection import StratifiedKFold


#Processing training face data
training_face_data = []
for i in range(1000):
    im = cv2.imread('./data/training/face/face{:0>2}'.format(i + 1)+ '.png')

    b, g, r = cv2.split(im)
    b = b.reshape(1,-1)
    g = g.reshape(1,-1)
    r = r.reshape(1,-1)

    x = np.hstack((b,g))
    x = np.hstack((x,r))
    x = np.append(x,1)
    training_face_data.append(x)

training_face_data = np.array(training_face_data)

#Processing training non-face data
training_nonface_data=[]
for i in range(1000):
    im = cv2.imread('./data/training/non-face/non-face{:0>2}'.format(i + 1)+ '.png')
    
    b,g,r = cv2.split(im)
    b = b.reshape(1,-1)
    g = g.reshape(1,-1)
    r = r.reshape(1,-1)
    
    x = np.hstack((b,g))
    x = np.hstack((x,r))
    x = np.append(x,0)
    training_nonface_data.append(x)

training_nonface_data = np.array(training_nonface_data)

#Processing testing face data
test_face_data=[]
for i in range(100):
    im = cv2.imread('./data/test/face/face{:0>2}'.format(i + 1)+ '.png')
    
    b,g,r = cv2.split(im)
    b = b.reshape(1,-1)
    g = g.reshape(1,-1)
    r = r.reshape(1,-1)
    
    x = np.hstack((b,g))
    x = np.hstack((x,r))
    x = np.append(x,1)
    test_face_data.append(x)

test_face_data = np.array(test_face_data)
#Processing testing non-face data
test_nonface_data=[]
for i in range(100):
    im = cv2.imread('./data/test/non-face/non-face{:0>2}'.format(i + 1)+ '.png')
    
    b,g,r = cv2.split(im)
    b = b.reshape(1,-1)
    g = g.reshape(1,-1)
    r = r.reshape(1,-1)
    
    x = np.hstack((b,g))
    x = np.hstack((x,r))
    x = np.append(x,0)
    test_nonface_data.append(x)

test_nonface_data = np.array(test_nonface_data)
test_data = np.vstack((test_face_data,test_nonface_data))


#get the mean and sd for face and non-face using single-Gaussian.
mean_face, sigma_face = single_Gaussian(training_face_data)
mean_nonface, sigma_nonface = single_Gaussian(training_nonface_data)


cov1 = np.diag(1/np.square(sigma_face)) #inverse of sigma_face
cov2 = np.diag(1/np.square(sigma_nonface)) #inverse of sigma_nonface

#model1
predict_label = []
predict_score = []
for i in range(200):
    n = 10800
    p1 = -n / 2 * np.log(2 * math.pi) - 1 / 2 * np.sum(np.log(np.square(sigma_face))) - 1 / 2 * np.dot(
         np.dot((test_data[i, 0:-1] - mean_face), cov1).T, (test_data[i, 0:-1] - mean_face))
    p0 = -n/2*np.log(2*math.pi) - 1/2 * np.sum(np.log(np.square(sigma_nonface))) - 1/2 * np.dot(np.dot(test_data[i,0:-1] - mean_nonface, cov2).T,(test_data[i, 0:-1] - mean_nonface))
    p_v = [p1, p0]
    p = p1 - scipy.special.logsumexp(p_v)
    p=np.exp(p)
    predict_score.append(p)
    if p > 0.5:
        predict_label.append(1)
    else:
       predict_label.append(0)

predict_label = np.array(predict_label)
predict_score = np.array(predict_score)

parameters(predict_label, test_data[:,-1], 200)
ROCcurve(test_data,predict_score)
toImage(mean_face)

#model2
#stratified_folder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)
#face_t = []
#face_v = []
#nonface_t = []
#nonface_v = []
#for train_index, test_index in stratified_folder.split(training_face_data[:,0:-1], training_face_data[:, -1]):
#    face_t.append(training_face_data[train_index])
#    face_v.append(training_face_data[test_index])
#for train_index, test_index in stratified_folder.split(training_nonface_data[:,0:-1], training_nonface_data[:,-1]):
#    nonface_t.append(training_nonface_data[train_index])
#    nonface_v.append(training_nonface_data[test_index])
#k=3
#l=3
#mu,sigma,lamda = EMalgorithm(face_t[0][:,:-1],k,0,1)
#nmu,nsigma,nlamda = EMalgorithm(nonface_t[0][:,:-1],l,0,1)

#validation = np.vstack((face_v[0], nonface_v[0]))
#test(validation,k,l,mu,sigma,lamda,nmu,nsigma,nlamda,0)    
#face_mix_mean=np.zeros(10800)
#face_mix_sigma=np.zeros(10800)
#nonface_mix_mean=np.zeros(10800)
#nonface_mix_sigma=np.zeros(10800)
#for i in range(k):
#    face_mix_mean=face_mix_mean+np.dot(mu[i],lamda[i])
#    face_mix_sigma=face_mix_sigma+np.dot(sigma[i],lamda[i])
#for i in range(l):
#    nonface_mix_mean=nonface_mix_mean+np.dot(nmu[i],nlamda[i])
#    nonface_mix_sigma=nonface_mix_sigma+np.dot(nsigma[i],nlamda[i])


#model3 t-distribution
#k=1
#l=1
#mu, sigma, lamda= EMalgorithm(training_face_data[:,:-1],k,1,1)

#nmu, nsigma, nlamda = EMalgorithm(training_nonface_data[:,:-1],l,1,1)
#test(test_data, k,l, mu,sigma, lamda, nmu, nsigma, nlamda, 1)


#model 4
#k=1  
#mu, sigma, lamda, theta, h = EMalgorithm(training_face_data[:,:-1], 1, 2, 2)
#nmu, nsigma, nlamda, ntheta, nh= EMalgorithm(training_nonface_data[:,:-1], 1, 2, 2)
#test(test_data, 1,1, mu,sigma, lamda, nmu, nsigma, nlamda, 2, theta, h, ntheta, nh)


#model5
#k=8 # k is the number you assume for hidden variables
#mu, sigma, lamda = EMalgorithm(training_face_data[:,:-1], k,1, 2)
#nmu, nsigma, nlamda = EMalgorithm(training_nonface_data[:,:-1],k,1,2)
#test(test_data, k,k, mu,sigma, lamda, nmu, nsigma, nlamda, 1)
#toImage(mean_face)    
#face_t_mean=np.zeros(10800)
#face_t_sigma=np.zeros(10800)
#nonface_t_mean=np.zeros(10800)
#nonface_t_sigma=np.zeros(10800)
#for i in range(k):
#    face_t_mean=face_t_mean+np.dot(mu[i],lamda[i])
#    face_t_sigma=face_t_sigma+np.dot(sigma[i],lamda[i])
#    nonface_t_mean=nonface_t_mean+np.dot(nmu[i],nlamda[i])
#    nonface_t_sigma=nonface_t_sigma+np.dot(nsigma[i],nlamda[i])

#model 6
#k=4  
#mu, sigma, lamda, theta, h = EMalgorithm(training_face_data[:,:-1], k, 2, 2)
#nmu, nsigma, nlamda, ntheta, nh= EMalgorithm(training_nonface_data[:,:-1], k, 2, 2)
#test(test_data, k,k, mu,sigma, lamda, nmu, nsigma, nlamda, 2, theta, h, ntheta, nh)
