from qrdet import QRDetector  # QRDetector sınıfını içe aktar
import cv2  # OpenCV kütüphanesini içe aktar
from pyzbar.pyzbar import decode  # QR kod çözme fonksiyonunu içe aktar
import numpy as np  # NumPy kütüphanesini içe aktar
import time  # Zaman işlemleri için time modülünü içe aktar

detector = QRDetector(model_size='s')  # QR kod dedektörü oluştur

# Video dosyasını aç
  # Video dosyanızın yolunu buraya girin
cap = cv2.VideoCapture(0)  # Kamerayı aç (0 numaralı kamera)
start=time.perf_counter()  # Başlangıç zamanını al
while True:  # Sonsuz döngü başlat
    # Videodan bir kare al
    ret, frame = cap.read()  # Kameradan bir kare oku
    if not ret:  # Eğer kare okunamazsa
        print("Video bitti veya bir hata oluştu.")  # Hata mesajı yazdır
        break  # Döngüden çık

    # Gösterilecek kareyi kopyala
    display_frame = frame.copy()  # Gösterilecek kareyi kopyala

    # QR kodu tespit et
    detections = detector.detect(image=frame, is_bgr=True)  # QR kodunu tespit et

    # Eğer QR kodu tespit edildiyse
    if detections:  # Eğer QR kodu tespit edildiyse
        for detection in detections:  # Her bir tespit için
            x1, y1, x2, y2 = map(int, detection['bbox_xyxy'])  # Tespit edilen alanın koordinatlarını al
            
            # Tespit edilen alanı kırp
            qr_region = frame[y1:y2, x1:x2]  # Tespit edilen QR kod bölgesini kırp
            
            # Görüntü işleme teknikleri uygula
            gray = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevir
            
            # Adaptif eşikleme uygula
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # Adaptif eşikleme uygula
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Gürültü azaltma
            binary = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)  # Gürültü azaltma uygula
            
            # Görüntüyü büyüt
            binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Görüntüyü büyüt
            
            # QR kodunu okumaya çalış
            decoded_objects = decode(binary)  # QR kodunu çözmeye çalış
            
            if decoded_objects:  # Eğer QR kodu başarıyla çözüldüyse
                # QR kodu başarıyla okundu
                qr_data = decoded_objects[0].data.decode('utf-8')  # QR kod verisini al
                print(f"QR kodu okundu: {qr_data}")  # QR kod verisini yazdır
                
                # Orijinal görüntüde QR kodunun yerini göster
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # QR kodunun etrafına yeşil dikdörtgen çiz
                cv2.putText(display_frame, qr_data, (x1, y1-10),   # QR kod verisini görüntüye yaz
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                # Görüntüyü kaydet
                cv2.imwrite('detected_qr.jpg', display_frame)  # Tespit edilen QR kodlu görüntüyü kaydet
                print("Görüntü kaydedildi.")  # Kayıt mesajını yazdır
                
                # Programı sonlandır
                
            else:  # Eğer QR kodu çözülemediyse
                # QR kodu tespit edildi ama okunamadı
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # QR kodunun etrafına kırmızı dikdörtgen çiz
                cv2.putText(display_frame, "QR tespit edildi, okunamadi", (x1, y1-10),   # Hata mesajını görüntüye yaz
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            break  # İç döngüden çık
        
    
    end=time.perf_counter()  # Bitiş zamanını al
    fps=1/(end-start)  # FPS hesapla
    start=time.perf_counter()  # Yeni başlangıç zamanını al
    print(fps)  # FPS'i yazdır
    # Görüntüyü göster
    cv2.imshow('QR Code Detection', display_frame)  # İşlenmiş görüntüyü göster

    # 'q' tuşuna basılırsa veya video biterse döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basılırsa
        break  # Döngüden çık

# Kaynakları serbest bırak
cap.release()  # Kamerayı serbest bırak
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat