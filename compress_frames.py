import os
import subprocess
import glob

def get_size(im):
    size = subprocess.check_output('magick identify '+im, shell=True)
    size = size.split()
    size = size[2].decode("utf-8")
    return size

def main():
    im = [i for i in glob.glob('Frames/*.jpg')]
    L = len(im)
    print('Original frame size: ', end='')
    print(get_size(im[0]))
    new_size = str(input('Enter new width: '))
    for i in range(L):
        os.system('convert -filter Cubic -resize '+new_size+' '+im[i]+' '+im[i])
    print('Done.')
if __name__=='__main__':
    main()