# =============================================================================
# DentaVision - app.py
# =============================================================================
# AMAÇ:
#   FastAPI tabanlı ana sunucu ve iş akışı yöneticisi.
#   Görüntüyü alır, sıfırdan yazılmış matematiksel ön işleme (preprocessing)
#   sokar, YOLO ile çıkarım (inference) yapar, hedef sınıfları filtreler ve
#   işaretli röntgen ile PDF raporunu döndürür.
# =============================================================================

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.preprocessing import rontgen_on_isle, gri_to_bgr
from src.postprocessing import maskeleri_temizle, gorseli_isaretle
from src.report import rapor_olustur
from src.schemas import HastaBilgisi
from src.inference import motor_yukle

# -----------------------------------------------------------------------------
# KONFİGÜRASYON VE SINIF TANIMLARI
# -----------------------------------------------------------------------------
INFERENCE_BACKEND = os.getenv("DENTAVISION_BACKEND", "pytorch")  # pytorch | onnx | tflite
MODEL_YOLU = os.getenv(
    "DENTAVISION_MODEL_PATH",
    {
        "pytorch": "models/best_model.pt",
        "onnx": "models/best_model.onnx",
        "tflite": "models/best_model.tflite",
    }[INFERENCE_BACKEND],
)
ON_ISLEME_YONTEMI = os.getenv("DENTAVISION_PREPROCESS", "hibrit")
GUVEN_ESIGI = float(os.getenv("DENTAVISION_CONF_THRESHOLD", "0.35"))

# Kaggle Veri Seti (31 Sınıf)
SINIF_ISIMLERI = {
    0: "Caries", 1: "Crown", 2: "Filling", 3: "Implant", 4: "Malaligned",
    5: "Mandibular Canal", 6: "Missing teeth", 7: "Periapical lesion",
    8: "Retained root", 9: "Root Canal Treatment", 10: "Root Piece",
    11: "Impacted tooth", 12: "Maxillary sinus", 13: "Bone Loss",
    14: "Fracture teeth", 15: "Permanent Teeth", 16: "Supra Eruption",
    17: "TAD", 18: "Abutment", 19: "Attrition", 20: "Bone defect",
    21: "Gingival former", 22: "Metal band", 23: "Orthodontic brackets",
    24: "Permanent retainer", 25: "Post-core", 26: "Plating", 27: "Wire",
    28: "Cyst", 29: "Root resorption", 30: "Primary teeth"
}

# Sadece hekime gösterilecek / rapora eklenecek KRİTİK patolojiler
HEDEF_SINIFLAR = [
    "Caries", "Periapical lesion", "Impacted tooth", "Bone Loss",
    "Fracture teeth", "Bone defect", "Cyst", "Root resorption"
]

UPLOAD_KLASORU = Path("uploads")
RAPOR_KLASORU = Path("reports")
UPLOAD_KLASORU.mkdir(exist_ok=True)
RAPOR_KLASORU.mkdir(exist_ok=True)

app = FastAPI(title="DentaVision API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
templates = Jinja2Templates(directory="templates")

# -----------------------------------------------------------------------------
# SİNGLETON MODEL YÜKLEME (Performans için sadece başlangıçta 1 kez çalışır)
# -----------------------------------------------------------------------------
_cikarim_motoru = None


@app.on_event("startup")
def modeli_yukle():
    global _cikarim_motoru
    if not Path(MODEL_YOLU).exists():
        print(f"[UYARI] Model bulunamadı: {MODEL_YOLU}. Sunucu modelsiz başlatılıyor.")
        return
    _cikarim_motoru = motor_yukle(INFERENCE_BACKEND, MODEL_YOLU, SINIF_ISIMLERI)
    print(f"[BAŞLANGIÇ] '{INFERENCE_BACKEND}' backend aktif. Model yüklendi.")


# -----------------------------------------------------------------------------
# ENDPOİNTLER
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def anasayfa(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analiz")
async def analiz_yap(
        ad: str = Form(...), soyad: str = Form(...), tc_kimlik: str = Form(...),
        yas: int = Form(...), sikayet: str = Form(""), xray_dosyasi: UploadFile = File(...)
):
    # 1) Kullanıcı Verisini Doğrula
    try:
        hasta = HastaBilgisi(ad=ad, soyad=soyad, tc_kimlik=tc_kimlik, yas=yas, sikayet=sikayet)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Geçersiz veri: {e}")

    if _cikarim_motoru is None:
        raise HTTPException(status_code=503, detail="Yapay zeka modeli henüz yüklenmedi.")

    # 2) Görüntüyü Belleğe Oku (Sadece I/O, işlem yok)
    islem_id = str(uuid.uuid4())[:8]
    ham_dizi = np.frombuffer(await xray_dosyasi.read(), dtype=np.uint8)
    orijinal_bgr = cv2.imdecode(ham_dizi, cv2.IMREAD_COLOR)
    if orijinal_bgr is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı.")

    # 3) Manuel Ön İşleme (Sıfırdan yazılmış matris matematiği)
    islenmis_gri = rontgen_on_isle(orijinal_bgr, yontem=ON_ISLEME_YONTEMI)
    islenmis_bgr = gri_to_bgr(islenmis_gri)

    # 4) YZ Çıkarımı (Strategy Pattern ile backend bağımsız)
    tespitler = _cikarim_motoru.tahmin_et(islenmis_bgr, guven_esigi=GUVEN_ESIGI)

    # 5) Maske Temizliği (Logical AND ile erozyon)
    tespitler = maskeleri_temizle(tespitler, erozyon_kernel=3)

    # 5.1) FİLTRELEME: Hekimin görmesine gerek olmayan sınıfları (tel, dolgu vb.) ele
    filtrelenmis_tespitler = [t for t in tespitler if t.sinif_adi in HEDEF_SINIFLAR]

    # 6) Orijinal Röntgene Çizim Yap (Etik gereği doktor işlenmemiş orjinali görmeli)
    isaretlenmis_bgr = gorseli_isaretle(orijinal_bgr, filtrelenmis_tespitler, guven_esigi_goster=GUVEN_ESIGI)

    isaretli_yol = UPLOAD_KLASORU / f"isaretli_{islem_id}.jpg"
    cv2.imwrite(str(isaretli_yol), isaretlenmis_bgr)

    # 7) PDF Rapor Üretimi
    tespit_sozlukleri = [
        {"sinif_adi": t.sinif_adi, "guven_skoru": t.guven_skoru, "bbox": list(t.bbox)}
        for t in filtrelenmis_tespitler
    ]
    rapor_yolu = RAPOR_KLASORU / f"rapor_{islem_id}.pdf"
    rapor_olustur(
        hasta=hasta.model_dump(), tespitler=tespit_sozlukleri,
        isaretlenmis_goruntu_yolu=str(isaretli_yol), cikti_pdf_yolu=str(rapor_yolu),
        on_isleme_yontemi=ON_ISLEME_YONTEMI
    )

    # 8) İstemciye Yanıt Dön
    return JSONResponse({
        "hasta": hasta.model_dump(),
        "tespitler": tespit_sozlukleri,
        "toplam_tespit_sayisi": len(tespit_sozlukleri),
        "isaretlenmis_goruntu_url": f"/uploads/isaretli_{islem_id}.jpg",
        "rapor_pdf_url": f"/reports/rapor_{islem_id}.pdf",
    })


@app.get("/saglik-kontrolu")
def saglik_kontrolu():
    return {"durum": "aktif", "backend": INFERENCE_BACKEND}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)