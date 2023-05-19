import cv2
import os
import numpy as np
#from matplotlib import pyplot as plt
from glob import glob
from random import randint

data = glob('/home/psdz/pr/mulgan-cvpr/output/128_shortcut1_inject1_none/sample_testing/*.png')

path = '/home/psdz/money/mask-skip/sagan/re3/new/mulgan/niter_001_034.png'
cimg = cv2.imread(path,1)
print(cimg.shape)

for k in range(12):
	for i in range(22):
		img = cimg[i*128+(1+i)*2:(i+1)*128+(1+i)*2,k*128+(1+k)*2:(k+1)*128+(1+k)*2,:]
		cv2.imwrite('/home/psdz/money/mask-skip/re/'+str(k)+ str(i)+'.jpg',img )

