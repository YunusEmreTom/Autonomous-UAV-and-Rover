import cv2  # OpenCV kütüphanesini içe aktar
from ultralytics import YOLO  # YOLO modelini içe aktar
import time  # Zaman işlemleri için time modülünü içe aktar

model = YOLO("best.engine")  # YOLO modelini yükle
cap = cv2.VideoCapture("denemevideo2.mp4")  # Video dosyasını aç
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video kodek formatını belirle
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))  # Çıkış video yazıcısını oluştur

score_threshold = 0.5  # Tespit eşik değerini belirle
fps_update_interval = 1  # FPS güncelleme aralığını belirle
last_fps_update = time.time()  # Son FPS güncellemesinin zamanını kaydet
frame_count = 0  # Kare sayacını başlat

while True:  # Ana döngü
    
    ret, frame = cap.read()  # Videodan bir kare oku
    current_time = time.time()  # Şu anki zamanı al
    if not ret:  # Eğer kare okunamazsa döngüyü sonlandır
        break
     
    results = model(frame, stream=True)  # YOLO modelini kullanarak nesne tespiti yap
    
    for result in results:  # Her bir sonuç için
        for box in result.boxes:  # Her bir tespit kutusu için
            if box.conf > score_threshold:  # Eğer güven skoru eşik değerinden yüksekse
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Kutu koordinatlarını al
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Tespit kutusunu çiz
                label = f"Class: {int(box.cls[0])}, Score: {box.conf[0].item():.2f}"  # Etiket metnini oluştur
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Etiketi çiz

    
    
    fps = 1 / (current_time - last_fps_update)  # FPS hesapla
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # FPS'i ekrana yaz
    last_fps_update = current_time  # Son FPS güncelleme zamanını güncelle
    

    cv2.imshow('YOLOv8 Live Detection', frame)  # Sonuç karesini göster
    out.write(frame)  # Kareyi çıkış videosuna yaz
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basılırsa döngüyü sonlandır
        break

cap.release()  # Video yakalayıcıyı serbest bırak
out.release()  # Video yazıcıyı serbest bırak
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat
