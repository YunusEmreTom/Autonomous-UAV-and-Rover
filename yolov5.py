import torch  # PyTorch kütüphanesini içe aktar
import cv2  # OpenCV kütüphanesini içe aktar
import time  # Zaman işlemleri için time modülünü içe aktar
import numpy as np  # NumPy kütüphanesini içe aktar

# YOLOv5 modelini yükle
device = torch.device('cuda')  # CUDA cihazını seç
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best (2).pt") # veya yolov5s - yolov5x6  # Özel YOLOv5 modelini yükle
model.to(device).float().eval()  # Modeli GPU'ya taşı, float tipine dönüştür ve değerlendirme moduna al
# Kamerayı başlat
cap = cv2.VideoCapture(r"C:\Users\TOM\Documents\Projeler\İTU_Gökbörü\yolov8_deneme\denemevideo2.mp4")  # Video kaynağını belirle

prev_frame_time = 0  # Önceki kare zamanını başlat
new_frame_time = 0  # Yeni kare zamanını başlat
fps_display_interval = 1  # FPS bilgilerini her saniyede bir güncelle
fps_display_time = time.time()  # FPS gösterim zamanını başlat

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodlayıcı ayarı
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))  # Çıktı dosyası

# Skorlama eşik değeri
score_threshold = 0.5  # Tespit eşik değerini ayarla

while True:  # Sonsuz döngü başlat
    ret, frame = cap.read()  # Kameradan bir kare oku
    
    if not ret:  # Eğer kare okunamazsa
        break  # Döngüden çık
    
    # Modeli kare üzerinde çalıştır
    results = model(frame)  # YOLOv5 modelini kare üzerinde çalıştır

    new_frame_time = time.time()  # Yeni kare zamanını al
    
    # Sonuçları filtreleme
    annotated_frame = frame.copy()  # Sonuçları orijinal kareye çizmeye başlamadan önce bir kopyasını al
    
    # YOLOv5 sonuçlarını işleme
    for det in results.xyxy[0]:  # sonuçları [x1, y1, x2, y2, confidence, class]
        if det[4] > score_threshold:  # Skor eşiğinin üzerinde olanları kontrol et
            x1, y1, x2, y2 = map(int, det[:4])  # Sınırlayıcı kutu koordinatlarını al
            label = f"Class: {int(det[5])}, Score: {det[4]:.2f}"  # Etiket metnini oluştur

            # Nesneyi kare üzerinde çiz
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Sınırlayıcı kutuyu çiz
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)  # Etiketi yaz

    # FPS hesaplama
    fps = 1 / (new_frame_time - prev_frame_time)  # FPS'i hesapla
    prev_frame_time = new_frame_time  # Önceki kare zamanını güncelle

    # FPS bilgisini güncelleme belirli aralıklarla
    if time.time() - fps_display_time > fps_display_interval:  # FPS gösterim zamanı geldiyse
        fps_display_time = time.time()  # FPS gösterim zamanını güncelle
        fps_text = f"FPS: {int(fps)}"  # FPS metnini oluştur
    else:
        fps_text = f"FPS: {int(fps)}"  # Son hesaplanan FPS değeri

    # FPS'i görüntüye yazdır
    cv2.putText(annotated_frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # FPS metnini görüntüye yaz

    # Görüntüyü ekrana yansıt
    cv2.imshow('YOLOv5 Live Detection', annotated_frame)  # İşlenmiş kareyi göster
    out.write(annotated_frame)  # İşlenmiş kareyi video dosyasına yaz
    
    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldıysa
        break  # Döngüden çık

# Kamerayı ve pencereleri serbest bırak
cap.release()  # Kamera kaynağını serbest bırak
out.release()  # Video yazıcıyı serbest bırak
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat
