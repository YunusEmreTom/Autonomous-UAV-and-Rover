import torch  # PyTorch kütüphanesini içe aktar
import numpy as np  # NumPy kütüphanesini içe aktar
import cv2  # OpenCV kütüphanesini içe aktar
from time import time  # time modülünden time fonksiyonunu içe aktar
from ultralytics import YOLO  # ultralytics kütüphanesinden YOLO sınıfını içe aktar

class BottleDetector:  # BottleDetector sınıfını tanımla
    

    def __init__(self, capture_index, model_name):  # Sınıfın yapıcı metodunu tanımla
        """
        hangi kamerayı kullancağımız, hangi modeli kullanacağımız ekran kartı mı yoksa işlemci mi kullanacağız
        ve bazı değişkenlere atama yapıyoruz
        """
        self.capture_index = capture_index  # Kamera indeksini ayarla
        self.model = self.load_model(model_name)  # Modeli yükle
        self.classes = self.model.names  # Model sınıflarını al
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU varsa CUDA'yı, yoksa CPU'yu kullan
        print("Using Device: ", self.device)  # Kullanılan cihazı yazdır

    def get_video_capture(self):  # Video yakalama nesnesini döndüren metod
        """
        kameradan görüntü alıyoruz
        """
      
        return cv2.VideoCapture(self.capture_index)  # Belirtilen indeksteki kamerayı aç ve döndür

    def load_model(self, model_name):  # Modeli yükleyen metod
        """
        Pytorch hub'dan Yolov5 modelini indiriyoruz
        ve bunu modüle geri döndürüyoruz 
        """
        if model_name:  # Eğer model adı belirtilmişse
            model = YOLO("best.engine")  # YOLO modelini "best.engine" dosyasından yükle
        else:  # Model adı belirtilmemişse
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # YOLOv5s modelini PyTorch hub'dan yükle
        return model  # Yüklenen modeli döndür

    def score_frame(self, frame):  # Kareyi değerlendiren metod
        """
        kameradan aldığı görüntüyü modele sokarak ondan tahmin oranı alıyoruz 
        """
        self.model.to(self.device)  # Modeli belirlenen cihaza taşı
        frame = [frame]  # Kareyi liste içine al
        results = self.model(frame)  # Modeli kullanarak tahmin yap
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]  # Etiketleri ve koordinatları ayır
        return labels, cord  # Etiketleri ve koordinatları döndür

    def class_to_label(self, x):  # Sınıf indeksini etikete dönüştüren metod
        """
        classlarımızı labela dönüştürüyoruz.
        """
        return self.classes[int(x)]  # Verilen indeksteki sınıf adını döndür

    def plot_boxes(self, results, frame):  # Tespit kutularını çizen metod
        """
        aranan objenin hangi konumlar içinde olduğunu buluyoruz.
        """
        labels, cord = results  # Sonuçları etiketlere ve koordinatlara ayır
        n = len(labels)  # Etiket sayısını al
        x_shape, y_shape = frame.shape[1], frame.shape[0]  # Karenin boyutlarını al
        for i in range(n):  # Her bir tespit için döngü
            row = cord[i]  # Koordinatları al
            if row[4] >= 0.3:  # Eğer güven skoru 0.3'ten büyükse
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)  # Koordinatları piksel cinsinden hesapla
                bgr = (0, 255, 0)  # Renk değerini ayarla (yeşil)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)  # Tespit kutusunu çiz
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)  # Sınıf etiketini yaz
                

        return frame  # İşlenmiş kareyi döndür

    def __call__(self):  # Sınıfın çağrılabilir metodu
        
        """
        kameramızı açarak aranan nesnenin nerede olduğunu hangi nesne olduğunu ve % kaç olasılıkla onun olduğunu yazıyoruz.
        """
        cap = self.get_video_capture()  # Video yakalama nesnesini al
        assert cap.isOpened()  # Kameranın açık olduğundan emin ol
      
        while True:  # Sonsuz döngü başlat

            start_time = time()  # Başlangıç zamanını al
            print(start_time)  # Başlangıç zamanını yazdır

            ret, frame = cap.read()  # Kameradan bir kare oku
            assert ret  # Karenin başarıyla okunduğundan emin ol
            
            frame = cv2.resize(frame, (416,416))  # Kareyi yeniden boyutlandır
            
            results = self.score_frame(frame)  # Kareyi değerlendir
            frame = self.plot_boxes(results, frame)  # Tespit kutularını çiz
            
            end_time = time()  # Bitiş zamanını al

            fps = 1/np.round(end_time - start_time, 2)  # FPS'i hesapla
            #print(f"her saniye frame yaz : {fps}")
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)  # FPS'i ekrana yaz
            
            cv2.imshow('YOLOv5 Detection', frame)  # İşlenmiş kareyi göster
 
            if cv2.waitKey(5) & 0xFF == ord('q'):  # 'q' tuşuna basılırsa döngüden çık
                break
      
        cap.release()  # Kamera kaynağını serbest bırak
        cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapat
        
# yeni bir obje oluşturarak çalıştırıyoruz.

detector = BottleDetector(capture_index=0, model_name='best.pt')  # BottleDetector nesnesini oluştur
detector()  # Nesneyi çağır ve çalıştır