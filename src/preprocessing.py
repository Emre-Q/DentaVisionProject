# =============================================================================
# DentaVision - src/preprocessing.py
# =============================================================================
# AMAÇ:
#   Röntgen görüntülerindeki düşük kontrast problemini çözmek ve maskeleri
#   temizlemek. cv2.equalizeHist, cv2.erode vb. hazır fonksiyonlar YASAKTIR.
#   Tüm işlemler NumPy ile sıfırdan piksel matematiği kullanılarak yazılmıştır.
#
# MATEMATİKSEL KONSEPTLER:
#   1) Germe (Stretching): Piksel aralığını [0, 255]'e doğrusal olarak yayar.
#   2) Eşitleme (Equalization): Kümülatif Dağılım Fonksiyonu (CDF) kullanarak
#      histogramı doğrusal olmayan şekilde düzleştirir, kontrastı agresif artırır.
#   3) Erozyon: İkili matrisler üzerinde kayan bir yapılandırma elemanıyla
#      uygulanan Mantıksal VE (Logical AND) işlemidir.
# =============================================================================

import numpy as np


# -----------------------------------------------------------------------------
# 1) HİSTOGRAM GERME (Linear Contrast Stretching)
# -----------------------------------------------------------------------------
def histogram_germe(img: np.ndarray, alt_persentil: float = 1.0,
                    ust_persentil: float = 99.0) -> np.ndarray:
    """Görüntünün dinamik aralığını [0, 255]'e doğrusal olarak gerer."""
    img_f = img.astype(np.float64)

    # Uçlardaki gürültüleri filtrelemek için persentil hesaplaması
    i_min = np.percentile(img_f, alt_persentil)
    i_max = np.percentile(img_f, ust_persentil)

    epsilon = 1e-6
    payda = (i_max - i_min) if (i_max - i_min) > epsilon else epsilon

    # I_out = (I_in - I_min) * (255 / (I_max - I_min))
    img_gerilmis = (img_f - i_min) * (255.0 / payda)
    img_gerilmis = np.clip(img_gerilmis, 0, 255)

    return img_gerilmis.astype(np.uint8)


# -----------------------------------------------------------------------------
# 2) HİSTOGRAM EŞİTLEME (CDF Tabanlı Histogram Equalization)
# -----------------------------------------------------------------------------
def histogram_esitleme(img: np.ndarray) -> np.ndarray:
    """CDF kullanarak histogramı düzleştirir ve yerel kontrastı artırır."""
    L = 256
    img_flat = img.flatten().astype(np.int64)
    N = img_flat.size

    # 1. Histogram (frekans)
    histogram = np.bincount(img_flat, minlength=L)

    # 2. Olasılık Dağılımı ve Kümülatif Dağılım Fonksiyonu (CDF)
    olasilik = histogram / float(N)
    cdf = np.cumsum(olasilik)

    # 3. CDF'yi [0, 255] aralığına ölçekle (Arama Tablosu - LUT oluştur)
    lut = np.round(cdf * (L - 1)).astype(np.uint8)

    # 4. Orijinal pikselleri yeni CDF değerleriyle eşleştir
    img_esitlenmis = lut[img.astype(np.uint8)]

    return img_esitlenmis.astype(np.uint8)


# -----------------------------------------------------------------------------
# 3) MORFOLOJİK EROZYON (Mantıksal VE)
# -----------------------------------------------------------------------------
def morfolojik_erozyon(binary_mask: np.ndarray, kernel_boyutu: int = 3) -> np.ndarray:
    """
    Kayan pencere altındaki komşu piksellerin tümü 1 ise merkez pikseli 1 yapar.
    Maske kenarlarındaki titremeleri ve gürültüleri (noise) temizler.
    """
    assert kernel_boyutu % 2 == 1, "Kernel boyutu tek sayı olmalı."

    mask = (binary_mask > 0).astype(np.uint8)
    h, w = mask.shape
    yaricap = kernel_boyutu // 2

    # Taşmaları önlemek için manuel zero-padding
    dolgulu = np.zeros((h + 2 * yaricap, w + 2 * yaricap), dtype=np.uint8)
    dolgulu[yaricap:yaricap + h, yaricap:yaricap + w] = mask

    # Vektörize edilmiş Mantıksal VE (Logical AND) işlemi
    sonuc = np.ones((h, w), dtype=bool)

    for di in range(kernel_boyutu):
        for dj in range(kernel_boyutu):
            komsu_dilim = dolgulu[di:di + h, dj:dj + w]
            sonuc = np.logical_and(sonuc, komsu_dilim.astype(bool))

    return sonuc.astype(np.uint8)


# -----------------------------------------------------------------------------
# 4) ANA YÖNLENDİRİCİ
# -----------------------------------------------------------------------------
def rontgen_on_isle(gri_goruntu: np.ndarray, yontem: str = "esitleme") -> np.ndarray:
    """Görüntüyü gri tonlamaya zorlar ve seçilen kontrast iyileştirmesini uygular."""

    # Eğer 3 kanallı (BGR) ise manuel luma (gri) dönüşümü: Y = 0.299R + 0.587G + 0.114B
    if gri_goruntu.ndim == 3:
        r, g, b = gri_goruntu[:, :, 2], gri_goruntu[:, :, 1], gri_goruntu[:, :, 0]
        gri_goruntu = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    if yontem == "germe":
        return histogram_germe(gri_goruntu)
    elif yontem == "esitleme":
        return histogram_esitleme(gri_goruntu)
    elif yontem == "hibrit":
        # Önce sıkışık histogramı ger, sonra yayılmış histogramı eşitle (En iyi sonuç)
        gerilmis = histogram_germe(gri_goruntu)
        return histogram_esitleme(gerilmis)
    else:
        raise ValueError("Yöntem 'germe', 'esitleme' veya 'hibrit' olmalıdır.")


def gri_to_bgr(gri_goruntu: np.ndarray) -> np.ndarray:
    """YOLO modelinin beklediği 3 kanallı format için matrisi çoğaltır."""
    return np.stack([gri_goruntu, gri_goruntu, gri_goruntu], axis=-1)