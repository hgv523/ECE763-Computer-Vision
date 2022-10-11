import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

Facepath = './face1'
facefiles = [os.path.join(root, name) for root, dirs, files in os.walk(Facepath) for name in files if name.endswith((".png"))]
faceLis = []
for faceimg in facefiles[0:100]:
    face = imread(faceimg)
    faceLis.append(face[:,:,0])
    facearray = np.asarray(faceLis)

    
nonFacepath = './non-face1'
nonfacefiles = [os.path.join(root, name) for root, dirs, files in os.walk(nonFacepath) for name in files if name.endswith((".png"))]
nonfaceLis = []
for nonfaceimg in nonfacefiles[0:100]:
    nonface = imread(nonfaceimg)
    nonfaceLis.append(nonface[:,:,0])
    nonfacearray = np.asarray(nonfaceLis)
    
data = np.concatenate((facearray, nonfacearray))

    


    
