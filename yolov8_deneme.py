import cv2
from ultralytics import YOLO
import time

# YOLOv8 modelini yükle
model = YOLO("dosya-yolu/best.pt") #modelinizin bulunduğu dosya yolunu buraya giriniz
model.to("cuda") #modeli cuda ile kullanmak için cuda, cpu ile kullanmak için cpu yazınız. Eğer cuda yüklü değilse cpu ile çalıştırabilirsiniz.

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı belirtir; eğer harici bir kamera kullanıyorsan, cihaz ID'sini belirtebilirsin. örn 0,1,2,3 vb...


# FPS hesaplamak için değişkenler
prev_frame_time         = 0
new_frame_time          = 0
fps_display_interval    = 1  # FPS bilgilerini her saniyede bir güncelle
fps_display_time        = time.time()


# Görüntümü kaydetmek için bazı ayarlar
fourcc                  = cv2.VideoWriter_fourcc(*'mp4v')  # Kodlayıcı ayarı
out                     = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))  # Çıktı dosyası

# Skorlama eşik değeri
score_threshold         = 0.5

while True:
    ret, frame = cap.read()  # Kameradan bir kare oku
    
    
    if not ret: # kamera açık mı diye kontrol et
        break


    results         = model(frame) #modelimizi çalıştır
    new_frame_time  = time.time()
    
    
    # Sonuçları filtreleme
    annotated_frame = frame.copy()  # Sonuçları orijinal kareye çizmeye başlamadan önce bir kopyasını al
    
    for result in results:
        # Sonuçların her bir nesnesi için
        for det in result.boxes.data:
            if det[4] > score_threshold:  # Skor eşiğinin üzerinde olanları kontrol et
                x1, y1, x2, y2 = map(int, det[:4])
                label = f"Class: {int(det[5])}, Score: {det[4]:.2f}"

                # Nesneyi kare üzerinde çiz
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # FPS hesaplama
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # FPS bilgisini güncelleme belirli aralıklarla
    if time.time() - fps_display_time > fps_display_interval:
        fps_display_time = time.time()
        fps_text = f"FPS: {int(fps)}"
    else:
        fps_text = f"FPS: {int(fps)}"  # Son hesaplanan FPS değeri

    # FPS'i görüntüye yazdır
    cv2.putText(annotated_frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Görüntüyü ekrana yansıt
    cv2.imshow('YOLOv8 Live Detection', annotated_frame)
    out.write(annotated_frame)
    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı, pencereleri ve kaydı serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()
