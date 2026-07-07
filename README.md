# DentaVision

Panoramik diş röntgenlerinde YOLOv8n-seg tabanlı çoklu-sınıf anomali tespiti ve segmentasyonu yapan uçtan uca web uygulaması. UFBA-UESC Dental Images veri seti üzerinde eğitim için tasarlanmıştır.

## Proje Klasör Yapısı

```
dentavision/
├── app.py                     # FastAPI ana sunucu (İş akışının yönetildiği orkestra şefi)
├── train.py                   # Deterministik eğitim scripti (Seed sabitlenmiş, tekrarlanabilir)
├── export_edge.py             # PyTorch -> ONNX / TFLite dönüştürücü (Uç cihaz uyumluluğu için)
├── requirements.txt           # Bağımlılıklar (cv2 sadece I/O için headless olarak eklidir)
├── README.md
│
├── src/
│   ├── preprocessing.py       # SIFIRDAN MATEMATİK: Histogram germe, eşitleme (NumPy tabanlı)
│   ├── postprocessing.py      # Maske temizleme (Mantıksal VE ile erozyon) ve görsel işaretleme
│   ├── inference.py           # Çıkarım Motoru (PyTorch/ONNX/TFLite Strategy Pattern)
│   ├── report.py              # Hasta bilgisi ve tespitlerden PDF Ön Teşhis Raporu üretimi
│   └── schemas.py             # Pydantic veri doğrulama modelleri
│
├── templates/
│   └── index.html             # Tıbbi lightbox (röntgen kutusu) temalı doktor arayüzü
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── data/
│   └── dataset.yaml           # Kaggle 31-Sınıflı veri seti için YOLO konfigürasyon dosyası
├── models/                    # Eğitilmiş model ağırlıkları (best_model.pt/.onnx/.tflite)
├── uploads/                   # Runtime sırasında işlenen hasta röntgenleri
└── reports/                   # Runtime sırasında üretilen PDF raporları
```

## Mimari Akış

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────────────┐
│   Doktor     │────▶│  FastAPI (app.py) │────▶│ 1. cv2.imdecode (sadece   │
│  (index.html)│     │  POST /analiz      │     │    dosya I/O)             │
└─────────────┘     └──────────────────┘     │ 2. preprocessing.py:      │
                                              │    manuel histogram       │
                                              │    germe/eşitleme         │
                                              └──────────┬────────────────┘
                                                         ▼
                                              ┌───────────────────────────┐
                                              │ inference.py               │
                                              │ (pytorch | onnx | tflite)  │
                                              │ YOLOv8n-seg çıkarımı       │
                                              └──────────┬────────────────┘
                                                         ▼
                                              ┌───────────────────────────┐
                                              │ postprocessing.py          │
                                              │ - manuel erosion (mask)    │
                                              │ - manuel alfa-karışım      │
                                              │   overlay + bbox çizimi    │
                                              └──────────┬────────────────┘
                                                         ▼
                                              ┌───────────────────────────┐
                                              │ report.py                  │
                                              │ Hasta bilgisi + tespitler  │
                                              │ -> PDF Ön Teşhis Raporu    │
                                              └──────────┬────────────────┘
                                                         ▼
                                              JSON yanıt: işaretli görüntü
                                              URL'i + tespit listesi + PDF URL'i
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Model Eğitimi

Kaggle üzerinden indirilen panoramik röntgen verisini data/images/{train,val} ve data/labels/{train,val} altına yerleştirdikten sonra eğitimi başlatın

```bash
python train.py --data data/dataset.yaml --epochs 100 --imgsz 640 --batch 16 --seed 42
```

- Eğitim tamamen deterministiktir (aynı seed = aynı sonuç).
- En iyi doğrulama (validation) skoruna sahip ağırlık otomatik olarak
  `models/best_model.pt` içine kaydedilir.

## 2) Edge AI Formatına Dönüştürme (opsiyonel ama önerilir)

```bash
# Sunucu tarafı, hafif CPU çıkarımı için ONNX (önerilen varsayılan)
python export_edge.py --model models/best_model.pt --format onnx

# Kaynak kısıtlı klinik içi cihazlar için TFLite (opsiyonel INT8 kuantizasyon)
python export_edge.py --model models/best_model.pt --format tflite --int8
```

## 3) Sunucuyu Başlatma

```bash
# Varsayılan: PyTorch backend
uvicorn app:app --reload

# ONNX backend ile (production önerilen)
DENTAVISION_BACKEND=onnx DENTAVISION_MODEL_PATH=models/best_model.onnx uvicorn app:app

# TFLite backend ile (Edge)
DENTAVISION_BACKEND=tflite DENTAVISION_MODEL_PATH=models/best_model.tflite uvicorn app:app
```

Tarayıcıda `http://localhost:8000` adresini açın.

## Ortam Değişkenleri

| Değişken                     | Açıklama                                      | Varsayılan            |
|-------------------------------|------------------------------------------------|------------------------|
| `DENTAVISION_BACKEND`         | `pytorch` \| `onnx` \| `tflite`                | `pytorch`              |
| `DENTAVISION_MODEL_PATH`      | Model ağırlık dosyası yolu                     | backend'e göre otomatik|
| `DENTAVISION_PREPROCESS`      | `germe` \| `esitleme` \| `hibrit`               | `hibrit`               |
| `DENTAVISION_CONF_THRESHOLD`  | Minimum güven skoru eşiği                       | `0.35`                 |

## Önemli Mühendislik Notları

1. **Ön işleme (src/preprocessing.py):** Histogram germe, histogram eşitleme
   ve morfolojik erozyon `cv2`'nin hazır fonksiyonları KULLANILMADAN, doğrudan
   NumPy dizileri üzerinde piksel matematiği ile (CDF hesabı, mantıksal VE)
   sıfırdan yazılmıştır. Dosya içindeki yorumlarda formüllerin matematiksel
   gerekçesi ayrıntılı olarak açıklanmıştır.
2. **Determinizm (train.py):** `random`, `numpy`, `torch` (CPU+CUDA) seed'leri
   ve `cudnn.deterministic=True` ile eğitim tam tekrarlanabilirdir.
3. **Edge AI (export_edge.py, src/inference.py):** Aynı arayüz üzerinden
   PyTorch/ONNX/TFLite arasında ortam değişkeniyle geçiş yapılabilir; kod
   değişikliği gerekmez (Strategy Pattern).

## Tıbbi/Etik Sorumluluk Reddi

DentaVision'ın ürettiği rapor bir **ön değerlendirme/tarama** aracı çıktısıdır,
kesin tıbbi teşhis YERİNE GEÇMEZ. Bu ibare, üretilen her PDF raporunda
otomatik olarak yer alır.
