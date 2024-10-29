import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# Abrir o vídeo
video_input_path = 'videos/20241019142228.mp4'
video_output_path = 'video/video_segmentado_slic.mp4'

# Ler o vídeo
cap = cv2.VideoCapture(video_input_path)

# Verificar se o vídeo abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

# Obter informações do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Definir codec e criar o objeto para escrever o novo vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame para o formato RGB (o OpenCV usa BGR por padrão)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Aplicar SLIC
    segments = slic(frame_rgb, n_segments=8000, compactness=10, start_label=1)

    # Converter os segmentos em uma imagem de rótulos colorida
    segmented_frame = label2rgb(segments, frame_rgb, kind='avg')

    # Converter de volta para BGR para salvar o vídeo corretamente
    segmented_frame_bgr = cv2.cvtColor((segmented_frame * 255).astype('uint8'), cv2.COLOR_RGB2BGR)

    # Escrever o frame segmentado no vídeo de saída
    out.write(segmented_frame_bgr)

    # Opcional: exibir o frame segmentado (pressione 'q' para parar)
    cv2.imshow('SLIC Video Segmentation', segmented_frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar tudo
cap.release()
out.release()
cv2.destroyAllWindows()
