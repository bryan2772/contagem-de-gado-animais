from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import re

# origens possíveis: image, screenshot, URL, video, YouTube, Streams -> ESP32 / Intelbras / Cameras On-Line
# mais informações em https://docs.ultralytics.com/modes/predict/#inference-sources
# Caminho para o vídeo
video_path = '/home/bryan/Área de Trabalho/projetos/tcc1/videos/VID_20240111_161152325.mp4'

# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture(video_path)

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Usa modelo da Yolo
# Model	    size    mAPval  Speed       Speed       params  FLOPs
#           (pixels) 50-95  CPU ONNX A100 TensorRT   (M)     (B)
#                           (ms)        (ms)
# YOLOv8n	640	    37.3	80.4	    0.99	    3.2	    8.7
# YOLOv8s	640	    44.9	128.4	    1.20	    11.2	28.6
# YOLOv8m	640	    50.2	234.7	    1.83	    25.9	78.9
# YOLOv8l	640	    52.9	375.2	    2.39	    43.7	165.2
# YOLOv8x	640	    53.9	479.1	    3.53	    68.2	257.8

model = YOLO("yolov8n.pt")

track_history = defaultdict(lambda: [])
seguir = True
max_bois_por_frame =  0
num_cows = 0
while True:

    success, img = cap.read()

    if success:

        if seguir:

            results = model.track(img, persist=True)
            #print(model.track)
            #print(model.track.predict)
            for result in results:

                if result.boxes is not None:

                    #print(result.names)
                    #print(result.boxes.cls)
                    #print(result.boxes)
                    test=result.verbose()
                    #test = "1 person, 5 cows"
                    #print(test)
                    # Use uma expressão regular para encontrar o número de vacas
                    #num_cows = int(re.search(r'(\d+) cows', test).group(1))
                    # Use uma expressão regular para encontrar o número de vacas

                    match = re.search(r'(\d+) cow', test)
                    if match:
                        num_cows = int(match.group(1))
                    #else:
                    #   num_cows = 0

                    #num_cows = sum(1 for cls in result.boxes.cls if result.names[int(cls)] == "cow")
                    print(f"num cows: {num_cows}")
                    max_bois_por_frame = max(max_bois_por_frame, num_cows)
                    print(f"Máximo de bois em um frame: {max_bois_por_frame}")
            
        else:

            results = model(img)
           
        # Process results list
        for result in results:
            # Visualize the results on the frame
            img = result.plot()
           
        #cv2.imshow("Tela", img)#abri a tela mostrando os frames/video

    #detecta se o q foi precionado na janela do frame, se sim fecha
    k = cv2.waitKey(1)

    if k == ord('q'):

        print("tecla pressionada")
        break

cap.release()
cv2.destroyAllWindows()

print(f"Máximo de bois em um frame: {max_bois_por_frame}")

