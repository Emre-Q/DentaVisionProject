# =============================================================================
# DentaVision - src/schemas.py
# =============================================================================
# AMAÇ:
#   FastAPI'nin otomatik veri doğrulaması (validation) ve OpenAPI/Swagger
#   dokümantasyonu için Pydantic veri modelleri.
# =============================================================================

from pydantic import BaseModel, Field, field_validator


class HastaBilgisi(BaseModel):
    """Doktorun arayüzden girdiği hasta bilgileri."""
    ad: str = Field(..., min_length=2, max_length=50, description="Hastanın adı")
    soyad: str = Field(..., min_length=2, max_length=50, description="Hastanın soyadı")
    tc_kimlik: str = Field(..., description="TC Kimlik Numarası (11 hane)")
    yas: int = Field(..., ge=0, le=120, description="Hasta yaşı")
    sikayet: str = Field(default="", max_length=500, description="Hastanın şikayeti / kliniğe geliş nedeni")

    @field_validator("tc_kimlik")
    @classmethod
    def tc_kimlik_dogrula(cls, v: str) -> str:
        """
        TC Kimlik No algoritmik (checksum) doğrulaması.
        KVKK/Güvenlik Notu: Bu hassas veri at-rest şifrelenmeli ve loglanmamalıdır.
        """
        if not v.isdigit() or len(v) != 11:
            raise ValueError("TC Kimlik Numarası 11 haneli rakamlardan oluşmalıdır.")

        haneler = [int(d) for d in v]
        if haneler[0] == 0:
            raise ValueError("TC Kimlik Numarası 0 ile başlayamaz.")

        tek_toplam = sum(haneler[0:9:2])
        cift_toplam = sum(haneler[1:8:2])

        if ((tek_toplam * 7) - cift_toplam) % 10 != haneler[9]:
            raise ValueError("Geçersiz TC Kimlik Numarası (10. hane checksum hatası).")

        if sum(haneler[0:10]) % 10 != haneler[10]:
            raise ValueError("Geçersiz TC Kimlik Numarası (11. hane checksum hatası).")

        return v


class TespitCiktisi(BaseModel):
    """Tek bir anomali tespitinin API yanıt formatı."""
    sinif_adi: str
    guven_skoru: float
    bbox: list[float]


class AnalizYaniti(BaseModel):
    """FastAPI /analiz endpoint'inin tam yanıt şeması."""
    hasta: HastaBilgisi
    tespitler: list[TespitCiktisi]
    isaretlenmis_goruntu_url: str
    rapor_pdf_url: str
    on_isleme_yontemi: str
    toplam_tespit_sayisi: int