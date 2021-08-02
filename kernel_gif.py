import glob
from PIL import Image
import re

frames = []
im = [img for img in glob.glob("Plot/*.png")]
im.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
for i in im:
    temp = Image.open(i)
    new_frame = temp.copy()
    frames.append(new_frame)
    temp.close()
for i in range(len(frames)-2, -1, -1):
    frames.append(frames[i])
frames[0].save('rgb.gif', format='GIF', append_images=frames[1:],
        save_all=True, duration=40, loop=0)