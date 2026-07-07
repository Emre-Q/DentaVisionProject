# =============================================================================
# DentaVision - src/inference.py
# =============================================================================
# AMAÇ:
#   Uygulamanın farklı model formatlarında (PyTorch, ONNX, TFLite) çalışabilmesi
#   için "Strateji (Strategy)" ve "Fabrika (Factory)" tasarım kalıplarını uygular.
#   app.py arka planda hangi motorun çalıştığını bilmez, sadece tahmin_et() çağırır.
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TespitSonucu:
    """Tek bir patoloji tespitini (ör: Çürük, Kemik Kaybı) ve maskesini tutar."""
    sinif_id: int
    sinif_adi: str
    guven_skoru: float  # 0.0 - 1.0 arası
    bbox: tuple  # (x1, y1, x2, y2)
    mask: np.ndarray  # (H, W) boyutunda 0/1 ikili (binary) matris


# -----------------------------------------------------------------------------
# 1. ORTAK ARAYÜZ (Interface)
# -----------------------------------------------------------------------------
class CikarimMotoru(ABC):
    @abstractmethod
    def tahmin_et(self, goruntu_bgr: np.ndarray, guven_esigi: float = 0.35) -> list:
        """Tüm motorlar BGR matris alır, TespitSonucu listesi döndürür."""
        pass


# -----------------------------------------------------------------------------
# 2. PYTORCH MOTORU (Geliştirme ve Eğitim Sonrası Hızlı Test İçin)
# -----------------------------------------------------------------------------
class PyTorchMotoru(CikarimMotoru):
    def __init__(self, model_yolu: str, sinif_isimleri: dict):
        from ultralytics import YOLO
        self.model = YOLO(model_yolu)
        self.sinif_isimleri = sinif_isimleri

    def tahmin_et(self, goruntu_bgr: np.ndarray, guven_esigi: float = 0.35) -> list:
        # verbose=False ile terminalin gereksiz loglarla dolması engellenir
        sonuclar = self.model.predict(goruntu_bgr, conf=guven_esigi, verbose=False)[0]
        return _ultralytics_sonucunu_donustur(sonuclar, self.sinif_isimleri)


# -----------------------------------------------------------------------------
# 3. ONNX MOTORU (Üretim/Sunucu Ortamı İçin Optimize Edilmiş Hızlı Motor)
# -----------------------------------------------------------------------------
class ONNXMotoru(CikarimMotoru):
    def __init__(self, model_yolu: str, sinif_isimleri: dict, imgsz: int = 640):
        import onnxruntime as ort
        # Edge cihazlar ve standart sunucular için CPU sağlayıcısı varsayılandır
        self.session = ort.InferenceSession(model_yolu, providers=["CPUExecutionProvider"])
        self.input_adi = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.sinif_isimleri = sinif_isimleri

    def tahmin_et(self, goruntu_bgr: np.ndarray, guven_esigi: float = 0.35) -> list:
        # Görüntü hazırlığı (YOLOv8 formatı: NCHW, float32, normalize)
        import cv2
        h, w = goruntu_bgr.shape[:2]
        olcek = self.imgsz / max(h, w)
        yeni_h, yeni_w = int(h * olcek), int(w * olcek)

        yeniden_boyutlu = cv2.resize(goruntu_bgr, (yeni_w, yeni_h))
        tuval = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        tuval[:yeni_h, :yeni_w] = yeniden_boyutlu

        rgb = tuval[:, :, ::-1]
        nchw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        girdi = np.ascontiguousarray(nchw)

        # Çıkarım
        cikti = self.session.run(None, {self.input_adi: girdi})
        return _onnx_ciktisini_decode_et(cikti, olcek, (h, w), self.sinif_isimleri, guven_esigi)


# -----------------------------------------------------------------------------
# 4. TFLITE MOTORU (Mobil ve Düşük Güçlü IoT/Klinik Cihazları İçin)
# -----------------------------------------------------------------------------
class TFLiteMotoru(CikarimMotoru):
    def __init__(self, model_yolu: str, sinif_isimleri: dict, imgsz: int = 640):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_yolu)
        self.interpreter.allocate_tensors()
        self.girdi_detay = self.interpreter.get_input_details()
        self.cikti_detay = self.interpreter.get_output_details()
        self.imgsz = imgsz
        self.sinif_isimleri = sinif_isimleri

    def tahmin_et(self, goruntu_bgr: np.ndarray, guven_esigi: float = 0.35) -> list:
        import cv2
        yeniden_boyutlu = cv2.resize(goruntu_bgr, (self.imgsz, self.imgsz))
        rgb = yeniden_boyutlu[:, :, ::-1].astype(np.float32) / 255.0
        girdi = rgb[None, ...]  # TFLite genelde NHWC formatı bekler

        self.interpreter.set_tensor(self.girdi_detay[0]["index"], girdi)
        self.interpreter.invoke()
        ciktilar = [self.interpreter.get_tensor(d["index"]) for d in self.cikti_detay]

        return _onnx_ciktisini_decode_et(ciktilar, 1.0, goruntu_bgr.shape[:2], self.sinif_isimleri, guven_esigi)


# -----------------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def _ultralytics_sonucunu_donustur(sonuclar, sinif_isimleri: dict) -> list:
    """PyTorch ham çıktısını uygulamamızın beklediği TespitSonucu formatına çevirir."""
    tespitler = []
    # Görüntüde hiçbir anomali bulunamazsa (maske yoksa) boş liste dön
    if sonuclar.masks is None or sonuclar.boxes is None:
        return tespitler

    for i in range(len(sonuclar.boxes)):
        kutu = sonuclar.boxes[i]
        maske = sonuclar.masks.data[i].cpu().numpy()
        sinif_id = int(kutu.cls.item())

        tespitler.append(TespitSonucu(
            sinif_id=sinif_id,
            sinif_adi=sinif_isimleri.get(sinif_id, f"sinif_{sinif_id}"),
            guven_skoru=float(kutu.conf.item()),
            bbox=tuple(kutu.xyxy[0].cpu().numpy().tolist()),
            mask=(maske > 0.5).astype(np.uint8),
        ))
    return tespitler


def _onnx_ciktisini_decode_et(ham_cikti, olcek, orijinal_boyut, sinif_isimleri, guven_esigi):
    """
    DİKKAT: YOLOv8-seg'in ONNX/TFLite ham çıktıları doğrudan koordinat vermez.
    Katsayılar (coefficients) ve proto-maskelerin matris çarpımı yapılarak
    manuel NMS (Non-Max Suppression) uygulanması gerekir.
    (Üretim ortamında Ultralytics 'ops.process_mask' fonksiyonu baz alınarak doldurulmalıdır).
    """
    raise NotImplementedError(
        "ONNX/TFLite çıktılarını manuel decode etme modülü henüz entegre edilmedi. "
        "Testler için lütfen DENTAVISION_BACKEND=pytorch ortam değişkenini kullanın."
    )


# -----------------------------------------------------------------------------
# FABRİKA (FACTORY) YÖNTEMİ
# -----------------------------------------------------------------------------
def motor_yukle(backend: str, model_yolu: str, sinif_isimleri: dict) -> CikarimMotoru:
    """app.py tarafından çağrılan, ortam değişkenine göre doğru motoru veren üretici."""
    backend = backend.lower()
    if backend == "pytorch":
        return PyTorchMotoru(model_yolu, sinif_isimleri)
    elif backend == "onnx":
        return ONNXMotoru(model_yolu, sinif_isimleri)
    elif backend == "tflite":
        return TFLiteMotoru(model_yolu, sinif_isimleri)
    else:
        raise ValueError(f"Geçersiz backend: {backend}. (pytorch, onnx, tflite)")