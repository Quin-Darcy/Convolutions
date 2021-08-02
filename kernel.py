from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
import colorsys
import numpy as np
import subprocess
import time
import re
import glob
import math
import cv2
import os

#============================Distance===========================
def dist(A, B):
    L = len(A)
    s = 0
    for i in range(L):
        s += (A[i]-B[i])**2
    s =math.sqrt(s)
    return s
#=============================X-Scale===========================
def XScale(img, v1, v2, v3, frame_num):
    S = '  '
    if (frame_num >= 100):
        S = ''
    elif (frame_num >= 10):
        S = ' '
    t = 0
    sz = img.size
    imm = Image.new('RGB', sz, color='white')
    pxx = imm.load()
    px = img.load()
    #v1 = float(input('Enter v1: '))
    #v2 = float(input('Enter v2: '))
    #v3 = float(input('Enter v3: '))
    temp = [0,0,0]
    count = 0
    count2 = 0
    t0 = time.time()
    T = time.time()
    rate = 0
    percent_complete = 0
    timeRem = 0
    hr = 0
    mr = 0
    sr = 0
    numPix = (sz[0])*(sz[1])
    pxRem = 0
    v = dist([v1,v2,v3], [0,0,0])
    for y in range(sz[1]):
        for x in range(sz[0]):
            temp = list(px[x, y])
            r = dist(temp, [0,0,0])
            t = r/(v+0.0000000001)
            a = int(t*v1)
            b = int(t*v2)
            c = int(t*v3)
            px[x,y] = (a, b, c)
            count += 1 
            count2 += 1
            percent_complete = (count/numPix)*100
            if (abs(time.time()-t0) >= 2.00):
                rate = count2/(time.time()-t0)
                t0 = time.time()
                count2 = 0
                pxRem = numPix - count
                timeRem = pxRem/rate
                hr = int(timeRem/3600)
                mr = int((timeRem-hr*3600)/60)
                sr = int(timeRem-hr*3600-mr*60)
            h = int((time.time()-T)/3600)
            m = int(((time.time()-T)-h*3600)/60)
            s = int((time.time()-T)-h*3600-m*60)
            print('Frame: '+str(frame_num), S+'|', 'Time Elapsed:',
                    str(f"{h:02d}")+'.'+str(f"{m:02d}")
                    +'.'+str(f"{s:02d}"), '|',
                    'Percent Complete:', 
                    "{:06.2f}".format(percent_complete),
                    '% | Rate:', "{:06.0f}".format(rate),
                    'pixels/sec |', 'Time Remaining:', 
                    str(f"{hr:02d}")+'.'+str(f"{mr:02d}") 
                    +'.'+str(f"{sr:02d}"), end=' \r', flush=True)
    im.save('Kernel/Colors/frame'+str(frame_num)+'.png')
    print('')
#============================RGB-to-HSV=========================
def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None
#=========================Long RGB2HSV========================
def hsvColor(img, frame_num, proc):
    if (frame_num < 10):
        sp = ' '
    else:
        sp = ''
    sz = img.size
    px = img.load()
    current_pixel = [0,0,0]
    pix_count = 0
    count_till_refresh = 0
    frame_count = frame_num
    t0 = time.time()
    T = time.time()
    rate = 0
    percent_complete = 0
    time_remaining = 0
    hr = 0
    mr = 0
    sr = 0
    num_pix = (sz[0])*(sz[1])
    pix_remaining = 0
    for y in range(sz[1]):
        for x in range(sz[0]):
            current_pixel = list(px[x, y])
            h,s,v = colorsys.rgb_to_hsv(current_pixel[0]/255.,current_pixel[1]/255.,current_pixel[2]/255.)
            if (proc == 2 and pix_count%(sz[0]) == 0):
                image = cv2.imread('Kernel/Filter/frame'+str(frame_count)+'.png',cv2.IMREAD_UNCHANGED)
                Rd = np.array(image[:,:,2]).flatten()
                Gr = np.array(image[:,:,1]).flatten()
                Bl = np.array(image[:,:,0]).flatten()
                L = len(Rd)
                C = [[100,100,100] for t in range(L)]
                for t in range(L): # Fuck you this is too slow 
                    C[t] = [Rd[t]/255.0, Gr[t]/255.0, Bl[t]/255.0]
                fig = plt.figure(figsize=(12,9))
                ax = plt.axes(projection='3d') 
                ax.scatter3D(Rd, Gr, Bl, s=0.1, c=C)
                ax.view_init(elev=frame_count/50+20, azim=((frame_count)+45))
                plt.savefig('Kernel/Plot/frame'+str(frame_count)+'.png',
                        bbox_inches='tight')
                plt.close(fig)
                px[x, y] = (int(h*255.), int(s*255.), int(v*255,))
                frame_count += 1
                img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
            else:
                px[x, y] = (int(h*255.), int(s*255.), int(v*255,))
            if (proc == 1 and pix_count%sz[0] == 0):
                img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
                frame_count += 1 
            pix_count += 1
            count_till_refresh += 1
            percent_complete = (pix_count/num_pix)*100
            if (abs(time.time()-t0) >= 2.00):
                rate = count_till_refresh/(time.time()-t0)
                t0 = time.time()
                count_till_refresh = 0
                pix_remaining = num_pix - pix_count
                time_remaining = pix_remaining/rate
                hr = int(time_remaining/3600)
                mr = int((time_remaining-hr*3600)/60)
                sr = int(time_remaining-hr*3600-mr*60)
            h = int((time.time()-T)/3600)
            m = int(((time.time()-T)-h*3600)/60)
            s = int((time.time()-T)-h*3600-m*60)
            print('Frame: ', str(frame_num),sp,'|','Time Elapsed:',
                    str(f"{h:02d}")+'.'+str(f"{m:02d}")
                    +'.'+str(f"{s:02d}"), '|',
                    'Percent Complete:', 
                    "{:06.2f}".format(percent_complete),
                    '% | Rate:', "{:05.0f}".format(rate),
                    'pixels/sec |', 'Time Remaining:', 
                    str(f"{hr:02d}")+'.'+str(f"{mr:02d}") 
                    +'.'+str(f"{sr:02d}"), end=' \r', flush=True)
    #img.save('Kernel/Frames/frame'+str(frame_count)+'.png')
    return [frame_count, time.time()-T]
#=======================Matrix Subtraction=====================
def matrix_sub(A, B):
    L = len(A)
    difference = [[0]*L for i in range(L)]
    for i in range(L):
        for j in range(L):
            difference[i][j] = A[i][j]-B[i][j]
    return difference
#======================Vector Mutliplication=====================
def vec_mult(U, V):
    L = len(U)
    prod = 0
    for j in range(L):
        prod += (U[j])*(V[j])
    return prod
#============================Average=============================
def Average(M): # Make this more general. Not just for 3x3 
    L = len(M)
    avg = 0
    for i in range(L):
        for j in range(L):
            avg += M[i][j]
    avg = int(avg/9)
    return avg
#======================Matrix Multiplication=====================
def matrix_mult(A, B):
    L = len(A)
    prod = [[0]*L for i in range(L)]
    for i in range(L):
        for j in range(L):
            column = [col[j] for col in B]
            prod[i][j] = vec_mult(A[i], column)
    return prod
#==============================Trace==============================
def Trace(M):
    L = len(M)
    trace = 0
    for i in range(L):
        trace += M[i][i]
    return trace
#===========================Lie Bracket===========================
def Bracket(window, kernel):
    bracket = matrix_sub(matrix_mult(window, kernel), matrix_mult(kernel, window))
    return bracket
#==================Convolution Multiplication=====================
def mult(window, kernel):
    sum = 0
    L = len(window)
    for i in range(L):
        for j in range(L):
            sum += (window[L-1-i][L-1-j])*(kernel[i][j])
    return int(sum)
#============================Set Kernel============================
def set_kernel(kSize):
    kernel = [[0]*kSize for i in range(kSize)]
    print('Enter kernel entries:\n')
    for i in range(kSize):
        for j in range(kSize):
            print(i, ', ', j, ': ', end='')
            kernel[i][j] = float(input(' '))
    return kernel
#===============================Blur=============================
def Blur(img, frame_num, proc):
    sp = str()
    if (frame_num < 10):
        sp = ' '
    else:
        sp = ''
    n = 3
    sz = img.size
    px = img.load()
    current_pixel_list = [0]*n
    window_center = [0]*n
    pix_count = 0
    count_till_refresh = 0
    frame_count = frame_num
    t0 = time.time()
    T = time.time()
    rate = 0
    percent_complete = 0
    time_remaining = 0
    hr = 0
    mr = 0
    sr = 0
    num_pix = (sz[0]-2)*(sz[1]-2)
    pix_remaining = 0
    windowR = [[0]*n for i in range(n)]
    windowG = [[0]*n for i in range(n)]
    windowB = [[0]*n for i in range(n)] 
    for y in range(sz[1]-2):
        for x in range(sz[0]-2):
            for i in range(n):
                for j in range(n):
                    current_pixel_list = list(px[x+j, y+i])
                    windowR[i][j] = current_pixel_list[0]
                    windowG[i][j] = current_pixel_list[1]
                    windowB[i][j] = current_pixel_list[2] 
            center_index = int((n+1)/2-1)
            X = int(Average(windowR))
            Y = int(Average(windowG))
            Z = int(Average(windowB))
            px[x+center_index,y+center_index]=(X, Y, Z)
            pix_count += 1 
            count_till_refresh += 1
            if (proc == 1 and pix_count%(sz[0]) == 0):
                img.save('Kernel/Filter/frame'+str(frame_count)+'.png') 
                frame_count += 1
            percent_complete = (pix_count/num_pix)*100
            if (abs(time.time()-t0) >= 2.00):
                rate = count_till_refresh/(time.time()-t0)
                t0 = time.time()
                count_till_refresh = 0
                pix_remaining = num_pix - pix_count
                time_remaining = pix_remaining/rate
                hr = int(time_remaining/3600)
                mr = int((time_remaining-hr*3600)/60)
                sr = int(time_remaining-hr*3600-mr*60)
            h = int((time.time()-T)/3600)
            m = int(((time.time()-T)-h*3600)/60)
            s = int((time.time()-T)-h*3600-m*60)
            print('Frame:', str(frame_num), sp,'|', 'Time Elapsed:',
                    str(f"{h:02d}")+'.'+str(f"{m:02d}")
                    +'.'+str(f"{s:02d}"), '|',
                    'Percent Complete:', 
                    "{:06.2f}".format(percent_complete),
                    '% | Rate:', "{:05.0f}".format(rate),
                    'pixels/sec |', 'Time Remaining:', 
                    str(f"{hr:02d}")+'.'+str(f"{mr:02d}") 
                    +'.'+str(f"{sr:02d}"), end=' \r', flush=True)
    if (proc == 0):
        img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
    else:
        img.save('Images/image.png')
    return [frame_count, time.time()-T]
#==============================Inverse===========================
def inverse(img, frame_num, proc):
    if (frame_num < 10):
        sp = ' '
    else:
        sp = ''
    n = 3
    sz = img.size
    px = img.load()
    current_pixel_list = [0]*n
    window_center = [0]*n
    pix_count = 0
    count_till_refresh = 0
    frame_count = frame_num
    t0 = time.time()
    T = time.time()
    rate = 0
    percent_complete = 0
    time_remaining = 0
    hr = 0
    mr = 0
    sr = 0
    num_pix = (sz[0]-2)*(sz[1]-2)
    pix_remaining = 0
    windowR = [[0]*n for i in range(n)]
    windowG = [[0]*n for i in range(n)]
    windowB = [[0]*n for i in range(n)]
    for y in range(sz[1]-2):
        for x in range(sz[0]-2):
            for i in range(n):
                for j in range(n):
                    temp = list(px[x+j, y+i])
                    windowR[i][j] = temp[0]
                    windowG[i][j] = temp[1]
                    windowB[i][j] = temp[2]
            R = np.array(windowR)
            G = np.array(windowG)
            B = np.array(windowB)
            if (np.linalg.det(R) == 0):
                Rinv = R
            else:
                Rinv = np.linalg.inv(R)
            if (np.linalg.det(G) == 0):
                Ginv = G
            else:
                Ginv = np.linalg.inv(G)
            if (np.linalg.det(B) == 0):
                Binv = B
            else:
                Binv = np.linalg.inv(B)
            for i in range(n):
                for j in range(n):
                    px[x+j, y+i] = (int(Rinv[i][j])%255, int(Ginv[i][j])%255,
                            int(Binv[i][j])%255)
            pix_count += 1 
            count_till_refresh += 1
            if (proc == 1 and pix_count%(sz[0]) == 0):
                img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
                frame_count += 1
            percent_complete = (pix_count/num_pix)*100
            if (abs(time.time()-t0) >= 2.00):
                rate = count_till_refresh/(time.time()-t0)
                t0 = time.time()
                count_till_refresh = 0
                pix_remaining = num_pix - pix_count
                time_remaining = pix_remaining/rate
                hr = int(time_remaining/3600)
                mr = int((time_remaining-hr*3600)/60)
                sr = int(time_remaining-hr*3600-mr*60)
            h = int((time.time()-T)/3600)
            m = int(((time.time()-T)-h*3600)/60)
            s = int((time.time()-T)-h*3600-m*60)
            print('Frame: ',str(frame_num), sp,'|', 'Time Elapsed:',
                    str(f"{h:02d}")+'.'+str(f"{m:02d}")
                    +'.'+str(f"{s:02d}"), '|',
                    'Percent Complete:', 
                    "{:06.2f}".format(percent_complete),
                    '% | Rate:', "{:05.0f}".format(rate),
                    'pixels/sec |', 'Time Remaining:', 
                    str(f"{hr:02d}")+'.'+str(f"{mr:02d}") 
                    +'.'+str(f"{sr:02d}"), end=' \r', flush=True)
    return [frame_count, time.time()]
#======================Bracket/Determinant=======================
def Br_Det(img, frame_num, proc, effect):
    sp = str()
    if (frame_num < 10):
        sp = ' '
    else:
        sp = ''
    #n = int(input('Enter kernel size: '))
    n = 3
    test_kernel = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    sz = img.size
    px = img.load()
    current_pixel_list = [0]*n
    window_center = [0]*n
    pix_count = 0
    count_till_refresh = 0
    frame_count = frame_num
    t0 = time.time()
    T = time.time()
    rate = 0
    percent_complete = 0
    time_remaining = 0
    hr = 0
    mr = 0
    sr = 0
    num_pix = (sz[0]-2)*(sz[1]-2)
    pix_remaining = 0
    windowR = [[0]*n for i in range(n)]
    windowG = [[0]*n for i in range(n)]
    windowB = [[0]*n for i in range(n)]
    for y in range(sz[1]-2):
        for x in range(sz[0]-2):
            for i in range(n):
                for j in range(n):
                    current_pixel_list = list(px[x+j, y+i])
                    windowR[i][j] = current_pixel_list[0]
                    windowG[i][j] = current_pixel_list[1]
                    windowB[i][j] = current_pixel_list[2]
            R = np.array(Bracket(windowR, test_kernel))
            G = np.array(Bracket(windowG, test_kernel))
            B = np.array(Bracket(windowB, test_kernel))
            if (0 == 1):
                detR = np.linalg.det(R)
                detG = np.linalg.det(G)
                detB = np.linalg.det(B)
                center_index = int((n+1)/2-1)
                X = 255-int(detR)
                Y = 255-int(detG)
                Z = 255-int(detB)
                if (proc == 2 and pix_count%(sz[0]) == 0):
                    image = cv2.imread('Kernel/Filter/frame'+str(frame_count)+'.png',cv2.IMREAD_UNCHANGED)
                    Rd = np.array(image[:,:,2]).flatten()
                    Gr = np.array(image[:,:,1]).flatten()
                    Bl = np.array(image[:,:,0]).flatten()
                    L = len(Rd)
                    C = [[100,100,100] for t in range(L)]
                    for t in range(L): # Fuck you this is too slow 
                        C[t] = [Rd[t]/255.0, Gr[t]/255.0, Bl[t]/255.0]
                    fig = plt.figure(figsize=(12,9))
                    ax = plt.axes(projection='3d')
                    ax.scatter3D(Rd, Gr, Bl, s=0.1, c=C)
                    ax.view_init(elev=20, azim=((frame_count)/10+45)%360)
                    plt.savefig('Kernel/Plot/frame'+str(frame_count)+'.png',
                            bbox_inches='tight')
                    plt.close(fig)
                    px[x+center_index, y+center_index] = (X,Y,Z)
                    frame_count += 1
                    img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
                else:
                    px[x+center_index,y+center_index]=(X, Y, Z)
            elif (6 == 5):
                #detR = np.linalg.det(R)
                #detG = np.linalg.det(G)
                #detB = np.linalg.det(B)
                R = Trace(windowR)
                G = Trace(windowG)
                B = Trace(windowB)
                center_index = int((n+1)/2-1)
                win = px[x+center_index, y+center_index]
                try:
                    X = int(math.exp(R))%255
                except OverflowError:
                    X = win[0]
                try:
                    Y = int(math.exp(G))%255
                except OverflowError:
                    Y = win[1]
                try:
                    Z = int(math.exp(B))%255
                except OverflowError:
                    Z = win[2]
                px[x+center_index,y+center_index]=(int(X), int(Y), int(Z))
            elif (effect%4 == 5):
                detR = np.linalg.det(R)
                detB = np.linalg.det(B)
                center_index = int((n+1)/2-1)
                window_center = px[x+center_index, y+center_index]
                X = 255-int(detR)
                Y = 255-window_center[1]
                Z = 255-int(detB)
                px[x+center_index,y+center_index]=(X, Y, Z)
            else:
                detR = np.linalg.det(R)
                detB = np.linalg.det(B)
                center_index = int((n+1)/2-1)
                window_center = px[x+center_index, y+center_index]
                X = int(detR)
                Y = window_center[1]
                Z = int(detB) 
                if (proc == 2 and pix_count%(sz[0]) == 0):
                    image = cv2.imread('Kernel/Filter/frame'+str(frame_count)+'.png',cv2.IMREAD_UNCHANGED)
                    Rd = np.array(image[:,:,2]).flatten()
                    Gr = np.array(image[:,:,1]).flatten()
                    Bl = np.array(image[:,:,0]).flatten()
                    L = len(Rd)
                    C = [[100,100,100] for t in range(L)]
                    for t in range(L): # Fuck you this is too slow 
                        C[t] = [Rd[t]/255.0, Gr[t]/255.0, Bl[t]/255.0]
                    fig = plt.figure(figsize=(12,9))
                    ax = plt.axes(projection='3d') 
                    ax.scatter3D(Rd, Gr, Bl, s=0.1, c=C)
                    ax.view_init(elev=frame_count/50+20, azim=((frame_count)+45))
                    plt.savefig('Kernel/Plot/frame'+str(frame_count)+'.png',
                            bbox_inches='tight')
                    plt.close(fig)
                    px[x+center_index, y+center_index] = (X,Y,Z)
                    frame_count += 1
                    img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
                else:
                    px[x+center_index,y+center_index]=(X, Y, Z)
            pix_count += 1 
            count_till_refresh += 1
            if (proc == 1 and pix_count%(sz[0]) == 0):
                img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
                frame_count += 1
                effect = frame_count
            percent_complete = (pix_count/num_pix)*100
            if (abs(time.time()-t0) >= 2.00):
                rate = count_till_refresh/(time.time()-t0)
                t0 = time.time()
                count_till_refresh = 0
                pix_remaining = num_pix - pix_count
                time_remaining = pix_remaining/rate
                hr = int(time_remaining/3600)
                mr = int((time_remaining-hr*3600)/60)
                sr = int(time_remaining-hr*3600-mr*60)
            h = int((time.time()-T)/3600)
            m = int(((time.time()-T)-h*3600)/60)
            s = int((time.time()-T)-h*3600-m*60)
            print('Frame:', str(frame_num), sp,'|', 'Time Elapsed:',
                    str(f"{h:02d}")+'.'+str(f"{m:02d}")
                    +'.'+str(f"{s:02d}"), '|',
                    'Percent Complete:', 
                    "{:06.2f}".format(percent_complete),
                    '% | Rate:', "{:05.0f}".format(rate),
                    'pixels/sec |', 'Time Remaining:', 
                    str(f"{hr:02d}")+'.'+str(f"{mr:02d}") 
                    +'.'+str(f"{sr:02d}"), end=' \r', flush=True)
    if (proc == 0):
        img.save('Kernel/Filter/frame'+str(frame_count)+'.png')
    else:
        img.save('Images/image.png')
    return [frame_count, time.time()-T]
#===============================VIDEO==================================
def video():
    f = subprocess.check_output('ls Kernel/Filter', shell=True).decode("utf-8")
    if f:
        os.system('rm Kernel/Filter/*.png')
    T = 0
    frames = []
    im = [img for img in glob.glob("Kernel/Frames/*.jpg")]
    im.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for i in im:
        temp = Image.open(i)
        new_frame = temp.copy()
        frames.append(new_frame)
        temp.close()
    for i in range(len(frames)):
        T += Br_Det(frames[i], i, 0, i)[1]
    print('')
    h = int(T/3600)
    m = int((T-h*3600)/60)
    s = int(T-h*3600-m*60)
    print('Total Time: ', str(f"{h:02d}")+'.'+str(f"{m:02d}")+'.'+str(f"{s:02d}"))
#==============================IMAGE===================================
def image():
    #f = subprocess.check_output('ls Kernel/Filter', shell=True).decode("utf-8")
    #if f:
    #    os.system('rm Kernel/Filter/*.png')
    im = Image.open('Kernel/Filter/frame0.png')
    frame = hsvColor(im, 0, 2)[0]
    im = Image.open('Kernel/Filter/frame'+str(frame-1)+'.png')
    frame = Br_Det(im, frame, 2, 0)
    #for i in range(3):
    #    frame = Br_Det(im, frame, 1, 0)[0]
    #    im = Image.open('Kernel/Filter/frame'+str(frame-1)+'.png')
    #    frame = hsvColor(im, frame, 1)[0]
    #    im = Image.open('Kernel/Filter/frame'+str(frame-1)+'.png')
    #    frame = inverse(im, frame, 1)[0]
    #    im = Image.open('Kernel/Filter/frame'+str(frame-1)+'.png') 
    print('')
#==============================MAIN====================================
def main():
    #video()
    image()
#======================================================================
if __name__=='__main__':
    main()

