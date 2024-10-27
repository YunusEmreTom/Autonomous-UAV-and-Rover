from ultralytics import YOLO  # YOLO sınıfını ultralytics kütüphanesinden içe aktar

model = YOLO("best.pt")  # 'best.pt' dosyasından YOLO modelini yükle
model.export(format="engine")  # modeli TensorRT engine formatına dönüştür ve kaydet

# Run inference
model = YOLO("best.engine")  # TensorRT engine formatındaki modeli yükle
results = model("https://ultralytics.com/images/bus.jpg")  # Belirtilen URL'deki görüntü üzerinde çıkarım yap

"""
def quantize_onnx_model(onnx_model_path, quantized_model_path):  # ONNX modelini nicelendirmek için bir fonksiyon tanımla
    from onnxruntime.quantization import quantize_dynamic, QuantType  # Gerekli modülleri içe aktar
    import onnx  # ONNX kütüphanesini içe aktar
    onnx_opt_model = onnx.load(onnx_model_path)  # ONNX modelini yükle
    quantize_dynamic(onnx_model_path,  # Dinamik nicelendirme uygula
    quantized_model_path,
    weight_type=QuantType.QUInt8) #chnage QInt8 to QUInt8 
    
quantize_onnx_model("best.onnx","best1.onnx")  # Fonksiyonu çağırarak ONNX modelini nicele
"""

"""
from ultralytics import YOLO  # YOLO sınıfını ultralytics kütüphanesinden içe aktar

model = YOLO("best.pt")  # 'best.pt' dosyasından YOLO modelini yükle
model.export(  # Modeli belirli parametrelerle dışa aktar
    format="engine",  # TensorRT engine formatında dışa aktar
    dynamic=True,  # Dinamik şekil kullan
      
    int8=True  # INT8 nicelendirmesi kullan
)

# Load the exported TensorRT INT8 model
model = YOLO("yolov8n.engine", task="detect")  # Dışa aktarılan TensorRT INT8 modelini yükle

# Run inference
result = model.predict("https://ultralytics.com/images/bus.jpg")  # Belirtilen URL'deki görüntü üzerinde çıkarım yap
"""