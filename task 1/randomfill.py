import cv2
import numpy as np
from random import randint
import sys
file_name=first_arg = sys.argv[1]
path="/home/akhil/Documents/Project/task 1 holw"
img=cv2.imread(file_name)
def found(x,y):
    flg=1
    d=0
    tsty=0
    for j in range(y,256):
        if(flg<3):
            if(img[x][j][0]!=255 and img[x][j][1]!=255 and img[x][j][2]!=255):
                tsty=1
            else:
                flg=d%3
        if(flg==3):
            if(img[x][j][0]!=0 and img[x][j][1]!=0 and img[x][j][2]!=0):
                tsty=1
            else:
                flg=d%3
        if(tsty==1 and (img[x][j][0]!=255 and img[x][j][1]!=0 and img[x][j][2]!=0)):
            break
            return 0,0
        elif(tsty==1 and (img[x][j][0]==255 and img[x][j][1]==0 and img[x][j][2]==0) ):
            tsty=0
            break
        d=d+1
    tstx=0
    if tsty==0:
        for i in range(x,255):
            if(img[i][y][0]==255 and img[i][y][1]==255 and img[i][y][2]==255):
                tstx=0
            else:
                tstx=1
            if(tstx==1 and (img[i][y][0]==255 and img[i][y][1]==0 and img[i][y][2]==0)):
                return i+1,j+1
                break
def randomfill(srtx,srty,endx,endy):
    for i in range(srtx,endx):
        for j in range(srty,endy):
            img[i][j]=[randint(0,255),randint(0,255),randint(0,255)]
    cv2.imwrite("randomfilled.png",img)
t=0
for i in range(0,255):
    for j in range(0,253):
        if((img[i][j][0]==255 and img[i][j][1]==255 and img[i][j][2]==255 ) and (img[i][j+1][0]==255 and img[i][j+1][1]==255 and img[i][j+1][2]==255 ) and (img[i][j+2][0]==0 and img[i][j+2][1]==0 and img[i][j+2][2]==0)):
            endx,endy=found(i,j)

            if(endx!=0 and endy!=0):
                print(i,"  ",j)
                print(endx,"  ",endy)
                t=1
                strtx=i
                strty=j
                randomfill(strtx,strty,endx,endy)
                break
    if(t==1):
        break
#cv2.imwrite('random_img.png',img)
