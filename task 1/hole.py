import cv2
import numpy as np
from random import randint
import sys

file_name=first_arg = sys.argv[1]
path="/home/akhil/Documents/Project/task 1 holw"
print(file_name)
img=cv2.imread(file_name)
img=cv2.resize(img,(256,256))

mask_posix=randint(5,200)
mask_posiy=randint(32,156)
mask_sizelen=randint(25,50)
mask_sizewidth=randint(25,50)


#cv2.imwrite('changed.jpg',img)
new_length=mask_posix+mask_sizewidth
new_width=mask_posiy+mask_sizelen
if(mask_posix+mask_sizewidth>255):
    new_length=255
if(mask_posiy+mask_sizelen>255):
    new_width=255
flag=0
mask_px=mask_posix
mask_py=mask_posiy

print(mask_px,"  ",mask_py)
print(new_length,"  ",new_width)
for i in range(mask_posix,new_length):
    for j in range(mask_posiy,new_width):
        if(flag<2):
            img[i][j]=[255,255,255]
            flag=flag+1
            #print("masking white  ",i,"   ",j)
        else:
            img[i][j]=[0,0,0]
            flag=0
            #print("masking  black ",i,"   ",j)
    flag=0
img[mask_posix][new_width-1]=[255,0,0]
img[new_length-1][mask_posiy]=[255,0,0]
cv2.imwrite('masked_ing.png',img)
