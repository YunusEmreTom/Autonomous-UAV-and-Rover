import cv2  # OpenCV kütüphanesini içe aktar
import numpy as np  # NumPy kütüphanesini içe aktar
import socket  # Socket kütüphanesini içe aktar
import torch  # PyTorch kütüphanesini içe aktar
import time  # Zaman işlemleri için time modülünü içe aktar
from pyzbar.pyzbar import decode  # QR kod çözme fonksiyonunu içe aktar
from qrdet import QRDetector  # QRDetector sınıfını içe aktar
from utils.general import non_max_suppression  # YOLOv5'ten non-max suppression fonksiyonunu içe aktar
import subprocess  # Alt süreç yönetimi için subprocess modülünü içe aktar

# Initialize YOLOv5 UAV detection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU varsa CUDA'yı, yoksa CPU'yu kullan
model = torch.load('C:\\Users\\rmznt\\Desktop\\yolov5\\best (3).pt', map_location=device)['model']  # YOLOv5 modelini yükle
model.to(device).float().eval()  # Modeli belirlenen cihaza taşı, float tipine dönüştür ve değerlendirme moduna al

# Initialize QR code detector
qr_detector = QRDetector(model_size='s')  # QR kod dedektörünü başlat

# Start video capture
cap = cv2.VideoCapture(0)  # Varsayılan kamerayı aç
input_size = (640, 480)  # Giriş boyutunu ayarla

# Mode flag (True for UAV detection, False for QR detection)
is_uav_mode = True  # Başlangıçta UAV tespit modunu etkinleştir

# Set up FFmpeg process
ffmpeg_cmd = [  # FFmpeg komut satırı argümanlarını ayarla
    'ffmpeg', '-re', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x480', '-i', '-',
    '-c:v', 'libx264', '-f', 'mpegts', 'udp://192.168.209.172:1557'
]
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)  # FFmpeg işlemini başlat

# Function to perform UAV detection
def detect_uav(frame, model, device, input_size):  # UAV tespiti yapan fonksiyon
    frame_resized = cv2.resize(frame, input_size)  # Kareyi yeniden boyutlandır
    img_tensor = torch.from_numpy(frame_resized[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)  # Kareyi tensor'a dönüştür

    with torch.no_grad():  # Gradyan hesaplamayı devre dışı bırak
        results = model(img_tensor)[0]  # Modeli kullanarak tahmin yap
        results = non_max_suppression(results, conf_thres=0.4, iou_thres=0.5)  # Non-max suppression uygula

    if results and len(results[0]) > 0:  # Eğer sonuç varsa
        for result in results[0]:  # Her bir sonuç için
            x1, y1, x2, y2 = map(int, result[:4])  # Sınırlayıcı kutu koordinatlarını al
            confidence = float(result[4])  # Güven skorunu al
            class_id = int(result[5])  # Sınıf ID'sini al
            if confidence > 0.6:  # Eğer güven skoru 0.6'dan büyükse
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Sınırlayıcı kutuyu çiz
                cv2.putText(frame, f'UAV Class: {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Sınıf etiketini yaz

# Function to perform QR detection and reading
def detect_qr(frame, qr_detector):  # QR kodu tespit eden ve okuyan fonksiyon
    qr_detections = qr_detector.detect(image=frame, is_bgr=True)  # QR kodunu tespit et
    if qr_detections:  # Eğer QR kodu tespit edildiyse
        for detection in qr_detections:  # Her bir tespit için
            x1, y1, x2, y2 = map(int, detection['bbox_xyxy'])  # Sınırlayıcı kutu koordinatlarını al
            qr_region = frame[y1:y2, x1:x2]  # QR kod bölgesini kırp
            gray = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevir
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptif eşikleme uygula
            binary = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)  # Gürültü azaltma uygula
            binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Görüntüyü büyüt
            decoded_objects = decode(binary)  # QR kodunu çözmeye çalış
            if decoded_objects:  # Eğer QR kodu başarıyla çözüldüyse
                qr_data = decoded_objects[0].data.decode('utf-8')  # QR kod verisini al
                print(f"QR Code Detected: {qr_data}")  # QR kod verisini yazdır
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # QR kodunun etrafına yeşil dikdörtgen çiz
                cv2.putText(frame, qr_data, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # QR kod verisini görüntüye yaz
            else:  # QR kodu çözülemediyse
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # QR kodunun etrafına kırmızı dikdörtgen çiz
                cv2.putText(frame, "QR Detected, Not Readable", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Okunamadı mesajını yaz

# Variables to track time and FPS
prev_time = time.time()  # Önceki zaman değerini kaydet

while cap.isOpened():  # Kamera açık olduğu sürece döngüyü sürdür
    ret, frame = cap.read()  # Kameradan bir kare oku
    if not ret:  # Eğer kare okunamazsa
        break  # Döngüden çık

    # Calculate FPS
    current_time = time.time()  # Şu anki zamanı al
    fps = 1 / (current_time - prev_time)  # FPS'i hesapla
    prev_time = current_time  # Önceki zaman değerini güncelle

    # Print FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # FPS değerini kareye yaz

    # Toggle between UAV and QR detection modes
    if is_uav_mode:  # Eğer UAV tespit modu etkinse
        detect_uav(frame, model, device, input_size)  # UAV tespiti yap
    else:  # QR tespit modu etkinse
        detect_qr(frame, qr_detector)  # QR kodu tespiti yap

    # Send frame to FFmpeg process
    ffmpeg_process.stdin.write(frame.tobytes())  # Kareyi FFmpeg işlemine gönder

    # Display the frame
    cv2.imshow('Detection', frame)  # Kareyi göster

    # Check for key press
    key = cv2.waitKey(1) & 0xFF  # Tuş basımını kontrol et
    if key == ord('q'):  # Press 'q' to exit
        break  # 'q' tuşuna basılırsa döngüden çık
    elif key == ord('r'):  # Press 'r' to switch between UAV and QR detection modes
        is_uav_mode = not is_uav_mode  # Tespit modunu değiştir
        if is_uav_mode:  # Eğer UAV tespit modu etkinleştirildiyse
            print("Switched to UAV detection mode")  # UAV tespit moduna geçildiğini yazdır
        else:  # QR tespit modu etkinleştirildiyse
            print("Switched to QR detection mode")  # QR tespit moduna geçildiğini yazdır

# Release resources
cap.release()  # Kamerayı serbest bırak
cv2.destroyAllWindows()  # Tüm açık pencereleri kapat
ffmpeg_process.stdin.close()  # FFmpeg işleminin giriş akışını kapat
ffmpeg_process.wait()  # FFmpeg işleminin tamamlanmasını bekle
