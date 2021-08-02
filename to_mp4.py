import cv2
import numpy as np
import glob
import subprocess
import re
import os

def getSize():
    size = subprocess.check_output('magick identify Filter/frame0.png',shell=True)
    size = size.split()
    size = size[2].decode("utf-8")
    size = size.split('x')
    size = (int(size[0]), int(size[1]))
    return size
def main():
    frame_size = getSize()
    out = cv2.VideoWriter('progress.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60,frame_size)
    im = [img for img in glob.glob("Filter/*.png")]
    im.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for filename in im:
        img = cv2.imread(filename)
        out.write(img)
    out.release()
    
if __name__=='__main__':
    main()