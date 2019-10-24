import cv2
import os
import time

start_time = time.time()
s =r'C:\Users\Siddharth\Documents\snek\train\train'                             #where all class folders are
s1 = r'C:\Users\Siddharth\Documents\snek\train\train_resized'             #the new training folder
os.mkdir(s1)
os.chdir(s)
classes = os.listdir()

for label in classes:
    os.mkdir(s1+'\\'+label)
    
for label in classes:
    loc = s+'\\'+label
    os.chdir(loc)
    allimages = os.listdir()
    count=0
    for image in allimages:
        try:
            count+=1
            img = cv2.imread(loc+'\\'+image,cv2.IMREAD_UNCHANGED)
            h=w=128
            dim=(w,h)
            # print(image)
            re = cv2.resize(img,dim)
            gray = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(s1+'\\'+label+'\\'+image,gray)
        except:
            continue
elapsed_time = time.time() - start_time
print(elapsed_time)