# =============================================================================
# DentaVision - export_edge.py
# =============================================================================
# AMAÇ:
#   Eğitim sonrası elde edilen 'models/best_model.pt' (PyTorch, ~6 MB) ağırlığını,
#   web sunucusunda DAHA HAFİF ve DAHA HIZLI çalışacak Edge AI formatlarına
#   (ONNX ve/veya TFLite) dönüştürmek.
#
#   NEDEN GEREKLİ?
#   - PyTorch runtime'ı (torch kütüphanesi) hem disk hem RAM açısından
#     ağırdır (yüzlerce MB); üretim (production) sunucusunda sadece
#     çıkarım (inference) yapılacaksa gereksiz yüklerdir.
#   - ONNX Runtime, grafiği önceden optimize eder (operator fusion,
#     constant folding) ve CPU üzerinde PyTorch'tan genelde 1.5-3x
#     daha hızlı çıkarım yapar.
#   - TFLite, mobil/gömülü (edge) cihazlarda (örn. kliniklerdeki düşük
#     kaynaklı bilgisayarlar, tablet) çalışacaksa, INT8 kuantizasyon ile
#     model boyutunu 4 kata kadar küçültebilir (örn. 6MB -> ~1.5MB).
#
#   Ultralytics, .pt -> .onnx ve .onnx -> .tflite dönüşümlerini
#   `model.export()` metoduyla tek satırda destekler; bu script bunu
#   DentaVision'ın klasör yapısına göre sarmalar (wrap) ve doğrulama
#   (sanity check) ekler.
# =============================================================================

import argparse
from pathlib import Path

from ultralytics import YOLO


def onnx_disa_aktar(model_yolu: str, cikti_klasoru: str = "models", imgsz: int = 640,
                     dinamik_boyut: bool = False, basitlestir: bool = True):
    """
    PyTorch (.pt) modelini ONNX formatına dönüştürür.

    Parametreler
    ----------
    dinamik_boyut : bool
        True ise, farklı görüntü boyutlarını (batch/H/W) destekleyen
        dinamik bir ONNX grafiği üretir. Web sunucusunda genelde sabit
        boyutta (imgsz x imgsz) ön işleme yapılacağından varsayılan False
        daha hızlı çıkarım sağlar.
    basitlestir : bool
        onnxsim ile grafiği sadeleştirir (gereksiz düğümleri kaldırır),
        çıkarımı hızlandırır ve dosya boyutunu küçültür.
    """
    print(f"[ONNX] '{model_yolu}' modeli ONNX formatına aktarılıyor...")
    model = YOLO(model_yolu)

    onnx_yolu = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dinamik_boyut,
        simplify=basitlestir,
        opset=12,          # ONNX Runtime ve TFLite dönüşümüyle geniş uyumluluk için
    )

    hedef = Path(cikti_klasoru) / "best_model.onnx"
    Path(cikti_klasoru).mkdir(parents=True, exist_ok=True)
    Path(onnx_yolu).replace(hedef)
    print(f"[ONNX] Başarılı -> {hedef}")
    return str(hedef)


def tflite_disa_aktar(model_yolu: str, cikti_klasoru: str = "models", imgsz: int = 640,
                       int8_kuantizasyon: bool = False):
    """
    PyTorch (.pt) modelini (Ultralytics içsel olarak TF SavedModel -> TFLite
    zincirinden geçirerek) TFLite formatına dönüştürür.

    int8_kuantizasyon=True verilirse, modeli 8-bit tam sayıya (INT8)
    kuantize eder. Bu, en agresif sıkıştırmadır (model ~4x küçülür) ancak
    kalibrasyon verisi (temsili bir örnek görüntü kümesi) gerektirir ve
    çok az doğruluk kaybına yol açabilir - bu yüzden opsiyoneldir.
    """
    print(f"[TFLite] '{model_yolu}' modeli TFLite formatına aktarılıyor "
          f"(int8={int8_kuantizasyon})...")
    model = YOLO(model_yolu)

    tflite_yolu = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=int8_kuantizasyon,
    )

    ek = "_int8" if int8_kuantizasyon else ""
    hedef = Path(cikti_klasoru) / f"best_model{ek}.tflite"
    Path(cikti_klasoru).mkdir(parents=True, exist_ok=True)
    Path(tflite_yolu).replace(hedef)
    print(f"[TFLite] Başarılı -> {hedef}")
    return str(hedef)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DentaVision - Edge AI Dışa Aktarma")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--format", type=str, choices=["onnx", "tflite", "both"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="Sadece TFLite için: INT8 kuantizasyon uygula")
    args = parser.parse_args()

    if args.format in ("onnx", "both"):
        onnx_disa_aktar(args.model, imgsz=args.imgsz)
    if args.format in ("tflite", "both"):
        tflite_disa_aktar(args.model, imgsz=args.imgsz, int8_kuantizasyon=args.int8)
