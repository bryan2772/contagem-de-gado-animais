import cv2
import os

# Caminho para o vídeo
video_path = '/home/bryan/Área de Trabalho/projetos/tcc1/videos/VID_20240111_161152325.mp4'

# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture(video_path)

# Verifique se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Loop para ler cada frame do vídeo
frame_count = 0
while True:
    # Leia o próximo frame
    ret, frame = cap.read()

    print("frame abrindo")
    
    # Verifique se o frame foi lido corretamente
    if not ret:
        break

    # Incrementa o contador de frames
    frame_count +=  1
    
# Libere o objeto de captura de vídeo
cap.release()

print(f"Frames ok: {frame_count}")
