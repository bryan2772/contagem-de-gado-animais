import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2

# Carregar a imagem
image = io.imread('datasets/CattleSegment.v7i.yolov8/test/images/00000211_jpg.rf.fbd1317b656f3d8f9efd34c507403a5f.jpg')
video_path = '/home/bryan/Área de Trabalho/projetos/tcc1/videos/20241019142228.mp4'


# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture(video_path)

'''#cap
# Exibir a imagem original
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.show()
'''

from skimage.segmentation import slic
from skimage.color import label2rgb

# Aplicar a segmentação SLIC
# n_segments é o número de superpixels que queremos, e compactness controla a regularidade dos segmentos
segments = slic(image, n_segments=100000, compactness=200, start_label=1)

# Converter os segmentos em uma imagem de rótulos colorida
segmented_image = label2rgb(segments, image, kind='avg')

# Exibir a imagem segmentada
plt.figure(figsize=(6, 6))
plt.imshow(segmented_image)
plt.axis('off')
plt.show()


