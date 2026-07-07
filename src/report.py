# =============================================================================
# DentaVision - src/report.py
# =============================================================================
# AMAÇ:
#   Hasta bilgileri ve YZ tespitlerini birleştirip profesyonel bir
#   "Ön Teşhis Raporu" PDF'i oluşturmak.
#
# TIBBİ UYARI:
#   Bu rapor kesin tanı belgesi değildir. Sistem sadece bir karar destek
#   mekanizmasıdır ve bu durum raporda (yasal olarak) açıkça belirtilir.
# =============================================================================

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Kaggle etiketlerini (İngilizce) PDF'te klinik Türkçeye çevirmek için sözlük
SINIF_CEVIRILERI = {
    "Caries": "Çürük",
    "Periapical lesion": "Kök Ucu İltihabı / Lezyon",
    "Impacted tooth": "Gömülü Diş",
    "Bone Loss": "Kemik Kaybı (Periodontal)",
    "Fracture teeth": "Kırık Diş",
    "Bone defect": "Kemik Defekti",
    "Cyst": "Kist",
    "Root resorption": "Kök Erimesi"
}


def rapor_olustur(hasta: dict, tespitler: list, isaretlenmis_goruntu_yolu: str,
                  cikti_pdf_yolu: str, on_isleme_yontemi: str) -> str:
    """Hasta ve tespit verilerini kullanarak PDF rapor üretir."""
    Path(cikti_pdf_yolu).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(cikti_pdf_yolu, pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm)
    stiller = getSampleStyleSheet()
    baslik_stili = ParagraphStyle("Baslik", parent=stiller["Heading1"],
                                  textColor=colors.HexColor("#1a4f7a"), fontSize=18)
    alt_baslik_stili = ParagraphStyle("AltBaslik", parent=stiller["Heading2"],
                                      textColor=colors.HexColor("#2c3e50"), fontSize=13)
    uyari_stili = ParagraphStyle("Uyari", parent=stiller["Normal"],
                                 textColor=colors.HexColor("#8a1c1c"), fontSize=9,
                                 backColor=colors.HexColor("#fdecea"))

    icerik = []

    # --- Başlık ---
    icerik.append(Paragraph("DentaVision — Yapay Zeka Destekli Ön Teşhis Raporu", baslik_stili))
    icerik.append(Paragraph(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}", stiller["Normal"]))
    icerik.append(Spacer(1, 0.5 * cm))

    # --- Hasta Bilgileri ---
    icerik.append(Paragraph("Hasta Bilgileri", alt_baslik_stili))
    hasta_tablosu = Table([
        ["Ad Soyad:", f"{hasta['ad']} {hasta['soyad']}"],
        ["TC Kimlik No:", _tc_maskele(hasta["tc_kimlik"])],
        ["Yaş:", str(hasta["yas"])],
        ["Şikayet:", hasta.get("sikayet", "-") or "-"],
    ], colWidths=[4 * cm, 12 * cm])

    hasta_tablosu.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
    ]))
    icerik.append(hasta_tablosu)
    icerik.append(Spacer(1, 0.6 * cm))

    # --- İşaretlenmiş Röntgen ---
    icerik.append(Paragraph("Analiz Edilen Panoramik Röntgen", alt_baslik_stili))
    icerik.append(Paragraph(f"<i>Uygulanan kontrast iyileştirme yöntemi: {on_isleme_yontemi}</i>", stiller["Normal"]))
    icerik.append(Spacer(1, 0.3 * cm))
    icerik.append(RLImage(isaretlenmis_goruntu_yolu, width=16 * cm, height=8 * cm, kind="proportional"))
    icerik.append(Spacer(1, 0.6 * cm))

    # --- Tespit Tablosu (Türkçe Çevirilerle Birlikte) ---
    icerik.append(Paragraph("Tespit Edilen Bulgular", alt_baslik_stili))
    if tespitler:
        satirlar = [["#", "Bulgu / Sınıf", "Güven Skoru"]]
        for i, t in enumerate(tespitler, start=1):
            # Arka plandan gelen İngilizce etiketi, Türkçe klinik terime çevir
            turkce_sinif = SINIF_CEVIRILERI.get(t["sinif_adi"], t["sinif_adi"])
            satirlar.append([str(i), turkce_sinif, f"%{t['guven_skoru'] * 100:.1f}"])

        tespit_tablosu = Table(satirlar, colWidths=[1.5 * cm, 10 * cm, 4.5 * cm])
        tespit_tablosu.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4f7a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
        ]))
        icerik.append(tespit_tablosu)
    else:
        icerik.append(
            Paragraph("Seçili patoloji kategorilerinde belirgin bir anomali tespit edilmemiştir.", stiller["Normal"]))

    icerik.append(Spacer(1, 0.8 * cm))

    # --- KVKK ve Yasal Uyarı (Zorunlu) ---
    icerik.append(Paragraph(
        "⚠ ÖNEMLİ UYARI: Bu rapor, yapay zeka destekli bir ön tarama/karar destek "
        "aracının otomatik çıktısıdır. Kesin tıbbi teşhis niteliği TAŞIMAZ ve "
        "yetkili bir diş hekiminin klinik değerlendirmesinin YERİNE GEÇMEZ. "
        "Nihai teşhis ve tedavi planı için mutlaka bir uzmana başvurunuz.",
        uyari_stili
    ))

    doc.build(icerik)
    return cikti_pdf_yolu


def _tc_maskele(tc: str) -> str:
    """KVKK uyumu için TC kimlik numarasını maskeler (Örn: 123*****789)."""
    if len(tc) != 11:
        return tc
    return f"{tc[:3]}{'*' * 5}{tc[8:]}"