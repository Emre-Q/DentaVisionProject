# =============================================================================
# DentaVision - train.py
# =============================================================================
# AMAÇ:
#   Kaggle Dental Disease Panoramic Dataset (31 sınıf) üzerinde YOLOv8n
#   (Detection) modelini deterministik (tekrarlanabilir) şekilde eğitmek.
#
# ÖZELLİKLER:
#   1) Tam Determinizm: Aynı kod + aynı veri = Her zaman aynı sonuç.
#   2) Otomatik Checkpoint: En iyi doğrulama skoruna sahip modeli otomatik
#      olarak 'models/best_model.pt' yoluna kopyalar.
# =============================================================================

import os
import random
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# 1) DETERMİNİZM: Rastgelelik Kaynaklarını Sabitleme
# -----------------------------------------------------------------------------
def seed_sabitle(seed: int = 42):
    """
    Eğitimin tekrarlanabilir olması için Python, NumPy ve PyTorch'un
    (CPU/GPU) tüm rastgele sayı üreteçlerini aynı tohum değerine sabitler.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN algoritma seçimini donanımdan bağımsız hale getirir
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(f"[SEED] Deterministik mod aktif. Tüm rastgelelik kaynakları seed={seed} ile sabitlendi.")


# -----------------------------------------------------------------------------
# 2) EN İYİ MODELİ KAYDEDEN CALLBACK SİNİFİ
# -----------------------------------------------------------------------------
class EnIyiModelKaydedici:
    """
    Her epoch sonunda Ultralytics'in skorlarını kontrol eder.
    İyileşme varsa modeli proje köküne 'best_model.pt' olarak yedekler.
    """

    def __init__(self, hedef_yol: str = "models/best_model.pt"):
        self.hedef_yol = Path(hedef_yol)
        self.hedef_yol.parent.mkdir(parents=True, exist_ok=True)
        self.en_iyi_skor = -1.0  # Başlangıç skoru

    def __call__(self, trainer):
        mevcut_fitness = float(trainer.fitness) if trainer.fitness is not None else -1.0

        if mevcut_fitness > self.en_iyi_skor:
            self.en_iyi_skor = mevcut_fitness
            kaynak = Path(trainer.best)

            if kaynak.exists():
                shutil.copy(kaynak, self.hedef_yol)
                print(f"[BEST-CHECKPOINT] Epoch {trainer.epoch}: "
                      f"Yeni skor = {mevcut_fitness:.5f} -> '{self.hedef_yol}' kaydedildi.")


# -----------------------------------------------------------------------------
# 3) ANA EĞİTİM AKIŞI
# -----------------------------------------------------------------------------
def egit(
        veri_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        seed: int = 42,
        model_agirlik: str = "yolov8n.pt",  # MİMARİ GÜNCELLEME: seg silindi
        proje_adi: str = "dentavision_egitim",
):
    # 1) Determinizmi başlat
    seed_sabitle(seed)

    # 2) Edge cihazlar için hafif olan Nano (n) model ağırlıklarını yükle
    model = YOLO(model_agirlik)

    # 3) Checkpoint mekanizmasını Ultralytics'e bağla
    en_iyi_kaydedici = EnIyiModelKaydedici()
    model.add_callback("on_fit_epoch_end", en_iyi_kaydedici)

    # 4) Eğitimi başlat
    sonuc = model.train(
        task="detect",       # MİMARİ GÜNCELLEME: Segmentasyon yerine Kutu Tespiti zorunlu kılındı
        data=veri_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        project=proje_adi,
        name="run",
        exist_ok=True,
        patience=20,

        # --- Medikal Görüntülere (X-Ray) Özel Augmentasyon Ayarları ---
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
    )

    print("\n" + "=" * 70)
    print(f"Eğitim tamamlandı. En iyi skor: {en_iyi_kaydedici.en_iyi_skor:.5f}")
    print(f"Model şuraya kaydedildi: models/best_model.pt")
    print("=" * 70)

    # 5) Doğrulama seti metrikleri (MİMARİ GÜNCELLEME: .seg yerine .box kullanıldı)
    metrikler = model.val()
    print(f"mAP50(box): {metrikler.box.map50:.4f} | mAP50-95(box): {metrikler.box.map:.4f}")

    return sonuc


# -----------------------------------------------------------------------------
# 4) TERMİNAL (CLI) ARAYÜZÜ
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DentaVision - Kaggle YOLOv8n Eğitim Scripti")
    parser.add_argument("--data", type=str, default="data/dataset.yaml",
                        help="Kaggle veri setinin dataset.yaml yolu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="yolov8n.pt") # MİMARİ GÜNCELLEME
    args = parser.parse_args()

    egit(
        veri_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        model_agirlik=args.weights,
    )