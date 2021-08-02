import cv2
import os
import subprocess

f = subprocess.check_output('ls Frames', shell=True).decode("utf-8")
if f:
    os.system('rm Frames/*.png')
cam = cv2.VideoCapture('Videos/spinface.mp4')
currentframe = 0
while (True):
    ret, frame = cam.read()
    if ret:
        name = './Frames/frame'+str(currentframe)+'.jpg'
        print('Creating ...'+name)
        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break
cam.release()
cv2.destroyAllWindows()
os.system('clear')