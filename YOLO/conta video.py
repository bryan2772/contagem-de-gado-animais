from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

# origens possíveis: image, screenshot, URL, video, YouTube, Streams -> ESP32 / Intelbras / Cameras On-Line
# mais informações em https://docs.ultralytics.com/modes/predict/#inference-sources
# Caminho para o vídeo
video_path = '/home/bryan/Área de Trabalho/projetos/tcc1/videos/20241019142228.mp4'



# URL do vídeo da câmera DVR
# Substitua 'http://seu_ip_da_câmera:porta/caminho_do_stream' pela URL real da sua câmera
video_url = 'rtsp://admin:Gol2772*@192.268.15.110:554/live/ch00_0'



video_url = 'rtsp://admin:test@192.168.05.110:554/live/ch00_0'



video_imag= '/home/bryan/Área de Trabalho/projetos/tcc1/videos/IMG_20240313_161951579.jpg'
# Inicialize o objeto de captura de vídeo com a URL da câmera
#cap = cv2.VideoCapture(video_url)

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

model = YOLO("yolov8x.pt")

track_history = defaultdict(lambda: [])
seguir = True
max_bois_por_frame =  0

while True:

    success, img = cap.read()

    if success:

        '''if seguir:

            results = model.track(img, persist=True)
            #print(model.track)
            #print(model.track.predict)
            for result in results:

                if result.boxes is not None:

                    #print(result.names)
                    #print(result.boxes.cls)
                    #print(result.boxes)
                    #print(result)
                    num_cows = sum(1 for cls in result.boxes.cls if result.names[int(cls)] == "cow")
                    print(f"num cows: {num_cows}")
                    max_bois_por_frame = max(max_bois_por_frame, num_cows)
                    print(f"Máximo de bois em um frame: {max_bois_por_frame}")
            
        else:'''
        

        if seguir:

            results = model.track(img, persist=True)
            
            for result in results:
                if result.boxes is not None:

                    # Considerando apenas detecções com confiança acima de 60%
                    num_cows = sum(1 for cls, conf in zip(result.boxes.cls, result.boxes.conf) 
                                if result.names[int(cls)] == "cow" and conf > 0.63)
                    
                    print(f"num cows: {num_cows}")
                    
                    max_bois_por_frame = max(max_bois_por_frame, num_cows)
                    print(f"Máximo de bois em um frame: {max_bois_por_frame}")

        else:
            # Lógica do else aqui (caso necessário)


            results = model(img)
           
        # Process results list
        for result in results:
            # Visualize the results on the frame
            img = result.plot()
            # Adiciona o texto com o valor de max_bois_por_frame na imagem
            cv2.putText(img, f"quantidade de bois em um frame: {num_cows}", (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f"Maximo de bois detectados em um frame: {max_bois_por_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
           
        cv2.imshow("Tela", img)#abri a tela mostrando os frames/video

    #detecta se o q foi precionado na janela do frame, se sim fecha
    k = cv2.waitKey(1)

    if k == ord('q'):

        print("tecla pressionada")
        break

cap.release()
cv2.destroyAllWindows()

print(f"Máximo de bois em um frame: {max_bois_por_frame}")

