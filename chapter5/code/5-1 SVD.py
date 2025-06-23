import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
# 1.定义恢复函数，由分解后的矩阵恢复到原矩阵
def restore(u, s, v, K): 
    n, p = len(u), len(v[0])
    a = np.zeros((n, p))
    for k in range(K):
        uk = u[:, k].reshape(n, 1)
        vk = v[k].reshape(1, p)
        a += s[k] * np.dot(uk, vk)   
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')
# 2.读取图像
A = np.array(Image.open("./svd.jpg", 'r'))
# 3.对RGB图像进行奇异值分解
u_r, s_r, v_r = np.linalg.svd(A[:, :, 0])    
u_g, s_g, v_g = np.linalg.svd(A[:, :, 1])
u_b, s_b, v_b = np.linalg.svd(A[:, :, 2])
# 4.保存奇异值为10，20，30，40，50的图像，并输出特征数
selected_svd_values = [10, 20, 30, 40, 50]
output_images = []
original_image = Image.fromarray(A)
output_images.append(original_image)
for k in selected_svd_values:
    R = restore(u_r, s_r, v_r, k)
    G = restore(u_g, s_g, v_g, k)
    B = restore(u_b, s_b, v_b, k)
    I = np.stack((R, G, B), axis=2)   
    img = Image.fromarray(I)
    output_images.append(img)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(output_images[i])
    if i == 0:
        ax.set_title('原图', fontname='SimSun', fontsize=40)
    else:
        ax.set_title(f'k={selected_svd_values[i-1]}', fontname='Times New Roman', fontsize=40)
    ax.axis('off')  
plt.tight_layout()
plt.savefig('svd_images_output.jpg', dpi=600)
plt.show()