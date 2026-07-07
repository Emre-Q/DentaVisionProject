# =============================================================================
# DentaVision - src/postprocessing.py
# =============================================================================
# AMAÇ:
#   Modelden dönen tespit (maske + bbox) verilerini, orijinal görüntü üzerine
#   matematiksel alfa-karışım (alpha blending) ile sıfırdan çizer.
#   Ayrıca preprocessing.py'deki manuel erozyonu çağırarak gürültüleri temizler.
# =============================================================================

import numpy as np
import cv2  # SADECE çizgi ve metin eklemek içindir, piksel manipülasyonu yasaktır.

from .preprocessing import morfolojik_erozyon

# Hekime gösterilecek patolojiler için BGR (Blue, Green, Red) renk paleti.
# DİKKAT: main.js dosyasındaki HEX kodlarının OpenCV için BGR'a dönüştürülmüş halidir.
SINIF_RENK_PALETI = {
    "Caries": (107, 107, 255),  # Kırmızımsı
    "Periapical lesion": (77, 157, 255),  # Kehribar
    "Impacted tooth": (217, 156, 177),  # Mor
    "Bone Loss": (0, 204, 255),  # Sarı
    "Fracture teeth": (70, 57, 230),  # Koyu Kırmızı
    "Bone defect": (97, 162, 244),  # Turuncu
    "Cyst": (157, 123, 69),  # Koyu Mavi
    "Root resorption": (106, 196, 233)  # Soluk Sarı
}


def maskeleri_temizle(tespitler: list, erozyon_kernel: int = 3) -> list:
    """
    Model çıktısındaki maskelere, kütüphanesiz yazılmış olan 'morfolojik erozyon'
    (Logical AND) işlemini uygular. Kenarlardaki pürüzleri ve gürültüleri temizler.
    """
    for tespit in tespitler:
        tespit.mask = morfolojik_erozyon(tespit.mask, kernel_boyutu=erozyon_kernel)
    return tespitler


def gorseli_isaretle(orijinal_bgr: np.ndarray, tespitler: list,
                     guven_esigi_goster: float = 0.35) -> np.ndarray:
    """
    Orijinal röntgen üzerine poligon maskeleri, kutuları ve etiketleri ekler.

    KURAL: cv2.addWeighted hazır fonksiyonu tüm görüntüyü karıştırır.
    Biz SADECE maske olan (1) piksellerde manuel formül uyguluyoruz:
    piksel = piksel * (1 - alpha) + renk * alpha
    """
    cikti = orijinal_bgr.copy().astype(np.float64)
    alpha = 0.4  # Maske saydamlık oranı

    for tespit in tespitler:
        if tespit.guven_skoru < guven_esigi_goster:
            continue

        renk = np.array(SINIF_RENK_PALETI.get(tespit.sinif_adi, (255, 255, 255)), dtype=np.float64)
        mask_bool = tespit.mask.astype(bool)

        # --- MANUEL ALFA-KARIŞIM (Sadece maske bölgesinde seçici matris işlemi) ---
        for k in range(3):
            kanal = cikti[:, :, k]
            kanal[mask_bool] = kanal[mask_bool] * (1 - alpha) + renk[k] * alpha
            cikti[:, :, k] = kanal

        # --- Çizim İşlemleri (cv2 istatistik/kontrast için değil, sadece çizgi için kullanılır) ---
        x1, y1, x2, y2 = map(int, tespit.bbox)
        cikti_uint8 = cikti.astype(np.uint8)

        # Bounding Box (Sınırlayan Kutu)
        cv2.rectangle(cikti_uint8, (x1, y1), (x2, y2), tuple(int(c) for c in renk), 2)

        # Etiket Kutusu ve Yazı
        etiket = f"{tespit.sinif_adi} %{tespit.guven_skoru * 100:.0f}"
        (etiket_w, etiket_h), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(cikti_uint8, (x1, y1 - etiket_h - 8), (x1 + etiket_w + 4, y1),
                      tuple(int(c) for c in renk), -1)
        cv2.putText(cikti_uint8, etiket, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Sonraki tespit işlemleri için güncel matrisi float64'e geri al
        cikti = cikti_uint8.astype(np.float64)

    return cikti.astype(np.uint8)