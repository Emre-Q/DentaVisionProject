# =============================================================================
# DentaVision - train.py
# =============================================================================
# AMAÇ:
#   Kaggle Dental Disease Panoramic Dataset (31 sınıf) üzerinde YOLOv8n
#   (Detection) modelini deterministik (tekrarlanabilir) şekilde eğitmek.
#
# ÖZELLİKLER:
#   1) Tam Determinizm: Aynı kod + aynı veri = Her zaman aynı sonuç.
#   2) Otomatik Checkpoint: En iyi modeli 'models/best_model.pt' yoluna kopyalar.
#   3) RESUME (Devam Etme): Kopan/durdurulan eğitimlere kaldığı yerden devam eder.
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
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[SEED] Deterministik mod aktif. Tüm rastgelelik kaynakları seed={seed} ile sabitlendi.")


# -----------------------------------------------------------------------------
# 2) EN İYİ MODELİ KAYDEDEN CALLBACK SINIFI
# -----------------------------------------------------------------------------
class EnIyiModelKaydedici:
    def __init__(self, hedef_yol: str = "models/best_model.pt"):
        self.hedef_yol = Path(hedef_yol)
        self.hedef_yol.parent.mkdir(parents=True, exist_ok=True)
        self.en_iyi_skor = -1.0

    def __call__(self, trainer):
        mevcut_fitness = float(trainer.fitness) if trainer.fitness is not None else -1.0
        if mevcut_fitness > self.en_iyi_skor:
            self.en_iyi_skor = mevcut_fitness
            kaynak = Path(trainer.best)
            if kaynak.exists():
                shutil.copy(kaynak, self.hedef_yol)
                print(f"[BEST-CHECKPOINT] Epoch {trainer.epoch}: Yeni skor = {mevcut_fitness:.5f} -> kaydedildi.")


# -----------------------------------------------------------------------------
# 3) ANA EĞİTİM AKIŞI
# -----------------------------------------------------------------------------
def egit(
        veri_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        seed: int = 42,
        model_agirlik: str = "yolov8n.pt",
        proje_adi: str = "dentavision_egitim",
        resume: bool = False  # <--- YENİ EKLENEN YETENEK
):
    seed_sabitle(seed)

    # Modeli yükle (Eğer resume true ise last.pt yüklenecek, false ise yolov8n.pt)
    model = YOLO(model_agirlik)

    en_iyi_kaydedici = EnIyiModelKaydedici()
    model.add_callback("on_fit_epoch_end", en_iyi_kaydedici)

    # Eğitimi başlat
    sonuc = model.train(
        task="detect",
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
        resume=resume,  # <--- EĞİTİME KALDIĞI YERDEN DEVAM ETME EMRİ

        # Medikal Görüntülere (X-Ray) Özel Augmentasyon Ayarları
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
    print("=" * 70)

    # Doğrulama seti metrikleri
    metrikler = model.val()
    print(f"mAP50(box): {metrikler.box.map50:.4f} | mAP50-95(box): {metrikler.box.map:.4f}")

    return sonuc


# -----------------------------------------------------------------------------
# 4) TERMİNAL (CLI) ARAYÜZÜ
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DentaVision - Kaggle YOLOv8n Eğitim Scripti")
    parser.add_argument("--data", type=str, default="data/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default="yolov8n.pt")

    # Koda dışarıdan müdahale edebilmek için --resume anahtarı eklendi
    parser.add_argument("--resume", action="store_true", help="Eğitime kaldığı yerden devam eder")

    args = parser.parse_args()

    egit(
        veri_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        model_agirlik=args.weights,
        resume=args.resume  # Terminalden gelen emri fonksiyona iletiyoruz
    )