import torch
import cv2
import numpy as np
from collections import defaultdict
#from yolact import YOLACT

# Caminho para o vídeo
video_path = '/home/bryan/Área de Trabalho/projetos/tcc1/videos/VID_20240111_161152325.mp4'

# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture(video_path)
'''
# Carregue o modelo YOLACT
model = YOLACT.from_pretrained("yolact_resnet50_54_800000.pth")
model.eval()'''

# Carregue o modelo YOLACT diretamente do GitHub
model = torch.hub.load('dbolya/yolact', 'yolact_resnet50_54_800000', pretrained=True)
model.eval()

track_history = defaultdict(lambda: [])
seguir = True
max_bois_por_frame = 0

while True:
    success, img = cap.read()
    if success:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)

        with torch.no_grad():
            preds = model(img)

        # Aqui você precisará adaptar a lógica para lidar com as segmentações de instâncias
        # Por exemplo, você pode querer extrair contornos das segmentações e contar instâncias de objetos específicos

        # Visualize os resultados na imagem
        img = cv2.cvtColor(img.squeeze(0).permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Tela", img)

        # Detecta se a tecla 'q' foi pressionada na janela do frame, se sim fecha
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()