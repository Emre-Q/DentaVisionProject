# DentaVision src paketi
# =============================================================================
# DentaVision - src/__init__.py
# =============================================================================
# AMAÇ:
#   'src' dizinini bir Python paketi olarak tanımlamak ve paketin dışarıya
#   açtığı temel arayüzü (Public API) tek bir noktada toplamak.
# =============================================================================

__version__ = "1.0.0"

# Paketin dışarıdan erişilebilecek temel fonksiyonları
from .preprocessing import rontgen_on_isle, gri_to_bgr
from .postprocessing import maskeleri_temizle, gorseli_isaretle
from .inference import motor_yukle
from .report import rapor_olustur

# Sadece bu listedeki fonksiyonların dışarıdan import edilmesine izin ver
__all__ = [
    "rontgen_on_isle",
    "gri_to_bgr",
    "maskeleri_temizle",
    "gorseli_isaretle",
    "motor_yukle",
    "rapor_olustur"
]