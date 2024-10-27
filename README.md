

# YOLO Modeli ile Nesne Tespiti 📸✨

Bu dosyalar, nesne tespiti yapan **YOLO (You Only Look Once)** modelinin farklı versiyonlarını (YOLOv5 ve YOLOv8) kullanan Python scriptleridir. Her bir dosyanın genel amacını açıklayarak projeyi daha iyi anlamanızı sağlayalım:

## 1. **cevirme_tensorrt.py** 🛠️
- **Amaç**: YOLO modelini **TensorRT** formatına dönüştürmek için kullanılan bir script.
- **Özellikler**:
  - **TensorRT**, NVIDIA'nın GPU üzerinde daha hızlı çıkarım (inference) yapabilmek için geliştirdiği bir optimizasyon aracıdır.
  - Modeli **ONNX** (Open Neural Network Exchange) formatına çevirme ve **nicelendirme (quantization)** işlemleri içerir.
  - Bu dönüşüm, modelin daha verimli çalışmasını ve daha hızlı sonuçlar üretmesini sağlar.

## 2. **anadeneme.py** 🎥
- **Amaç**: TensorRT'ye dönüştürülmüş YOLO modelini kullanarak video üzerinde nesne tespiti yapmak.
- **Özellikler**:
  - Video dosyasından okuma yapar ve tespit edilen nesnelerin sonuçlarını yeni bir video dosyasına kaydeder.
  - **FPS (Frames Per Second)** hesaplaması yaparak, işlem performansını ölçer.
  - Video görüntüleme imkanı sunarak kullanıcı etkileşimini artırır.

## 3. **Hiz_deneme_yolo.py** 🚀
- **Amaç**: `BottleDetector` sınıfı içeren, şişe tespiti için özelleştirilmiş bir script.
- **Özellikler**:
  - Kameradan canlı görüntü alarak anlık nesne tespiti yapar.
  - **GPU/CPU** seçimi yaparak sistemin donanımına uygun performans ölçümü gerçekleştirir.
  - Kullanıcı dostu bir arayüz ile tespit edilen nesneleri izleme olanağı sunar.

## 4. **yoloqrson.py** 📦
- **Amaç**: Hem **UAV (İnsansız Hava Aracı)** hem de **QR kod** tespiti yapabilen hibrit bir sistem.
- **Özellikler**:
  - İki mod arasında geçiş yapabilme özelliği ile esneklik sağlar.
  - **FFmpeg** kullanarak video akışını **UDP** üzerinden iletme özelliği içerir, bu sayede uzaktan izleme imkanları sunar.

## 5. **yolov5.py** 🔍
- **Amaç**: **YOLOv5** modelini kullanarak video üzerinde nesne tespiti yapmak.
- **Özellikler**:
  - Tespit edilen nesneleri sınırlayıcı kutular ve etiketlerle görselleştirir.
  - **FPS** ölçümü yaparak işlem hızı hakkında bilgi verir ve video kaydı yapar.
  - Kullanıcıya gerçek zamanlı geri bildirim sağlar.

## 6. **yolov8_deneme.py** 🚀
- **Amaç**: **YOLOv8** modelini kullanarak kameradan canlı görüntü üzerinde nesne tespiti yapmak.
- **Özellikler**:
  - Daha yeni ve daha gelişmiş **YOLOv8** mimarisini kullanarak daha yüksek doğruluk sağlar.
  - Tespit sonuçlarını görselleştirme ve kaydetme özellikleri içerir, böylece kullanıcı sonuçları analiz edebilir.

---

## Genel Özellikler 🌟
Bu scriptler, nesne tespiti, görüntü işleme, performans ölçümü (FPS), video kaydı ve gerçek zamanlı görselleştirme gibi temel işlevleri yerine getirmektedir. Farklı YOLO versiyonları ve çeşitli optimizasyon teknikleri (TensorRT gibi) kullanarak, çeşitli kullanım senaryolarına uygun çözümler sunmaktadırlar. Bu sayede, kullanıcılar farklı senaryolara göre en iyi çözümü bulma şansı elde ederler.

