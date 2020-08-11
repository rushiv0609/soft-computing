import numpy as np
import cv2
from PIL import Image

r = np.array(Image.open('1.gif')).flatten()
g = np.array(Image.open('2.gif')).flatten()
b = np.array(Image.open('3.gif')).flatten()
i = np.array(Image.open('4.gif')).flatten()

r = r - r.mean()
g = g - g.mean()
b = b - b.mean()
i = i - i.mean()

cov = np.cov([r,g,b,i])
_, vec = np.linalg.eig(cov)
PC = [[],[],[],[]]

for j in range(r.shape[0]):
    for k in range(4):
        PC[k].append(np.dot(vec[:,k], [r[j],g[j],b[j],i[j]]))
        
new_r = np.array(PC[0]).reshape((512,512)).astype(np.uint8)
new_g = np.array(PC[1]).reshape((512,512)).astype(np.uint8)
new_b = np.array(PC[2]).reshape((512,512)).astype(np.uint8)
new_i = np.array(PC[3]).reshape((512,512)).astype(np.uint8)

img_r = Image.fromarray(new_r)
img_g = Image.fromarray(new_g)
img_b = Image.fromarray(new_b)
img_i = Image.fromarray(new_i)

img_r.save("PCA_1.png")
img_g.save("PCA_2.png")
img_b.save("PCA_3.png")
img_i.save("PCA_4.png")
