"""
Kamera test kodu.

"""

import cv2  # OpenCV kütüphanesini içe aktar

def list_active_cameras():  # Aktif kameraları listeleyen fonksiyon
    # Kamera indeksleri için bir liste oluştur
    active_cameras = []  # Aktif kameraları tutacak boş liste
    # 0'dan 9'a kadar olan cihazları kontrol et
    for i in range(10):  # 0'dan 9'a kadar döngü
        # Kamerayı açmaya çalış
        cap = cv2.VideoCapture(i)  # i indeksli kamerayı aç
        if cap.isOpened():  # Kameranın açılıp açılmadığını kontrol et
            active_cameras.append(i)  # Aktif olan kamerayı listeye ekle
            cap.release()  # Kamerayı serbest bırak

    return active_cameras  # Aktif kameraların listesini döndür


cameras = list_active_cameras()  # Aktif kameraları listele
if cameras:  # Eğer aktif kamera varsa
    print("Aktif Kameralar: ", cameras)  # Aktif kameraları yazdır
else:  # Aktif kamera yoksa
    print("Hiçbir aktif kamera bulunamadı.")  # Uyarı mesajı yazdır

cap = cv2.VideoCapture(0) # Kamera id'si ne ise o konulmalı. Örneğin 0 yerine 1,2,3 gibi, Bazen de direk port verebilirsiniz. Örn: COM6, ACM0 gibi... 

while True:  # Sonsuz döngü
    _,img = cap.read()  # Kameradan bir kare oku


    cv2.imshow("img",img)  # Okunan kareyi göster
    if cv2.waitKey(1) ==27: # ESC çıkma kodu
        break  # ESC tuşuna basılırsa döngüden çık

cv2.destroyAllWindows()  # Tüm açık pencereleri kapat
cap.release()  # Kamerayı serbest bırak