// =============================================================================
// DentaVision - static/js/main.js
// =============================================================================
// AMAÇ: 
// 1. Form ve dosya yükleme işlemlerini yönetmek.
// 2. FastAPI sunucusuyla (/analiz) haberleşmek.
// 3. Arka plandan gelen evrensel İngilizce etiketleri klinik Türkçeye çevirip,
//    tema uyumlu renklerle DOM'a (arayüze) basmak.
// =============================================================================

const form = document.getElementById("analiz-formu");
const dropzoneInput = document.getElementById("xray_dosyasi");
const onizlemeGoruntu = document.getElementById("onizleme-goruntu");
const yuklemeIpucu = document.getElementById("yukleme-ipucu");
const analizButonu = document.getElementById("analiz-butonu");
const bosDurum = document.getElementById("bos-durum");
const sonucIcerik = document.getElementById("sonuc-icerik");
const hataDurum = document.getElementById("hata-durum");

// SADECE hekime gösterilecek (app.py'deki HEDEF_SINIFLAR) patolojiler için renk paleti
const SINIF_RENKLERI = {
    "Caries": "#ff6b6b",             // Kırmızı (Kritik)
    "Periapical lesion": "#ff9d4d",  // Kehribar (Uyarı)
    "Impacted tooth": "#b19cd9",     // Mor (Cerrahi)
    "Bone Loss": "#ffcc00",          // Sarı (Dikkat)
    "Fracture teeth": "#e63946",     // Koyu Kırmızı (Acil)
    "Bone defect": "#f4a261",        // Turuncu
    "Cyst": "#457b9d",               // Koyu Mavi (Lezyon)
    "Root resorption": "#e9c46a"     // Soluk Sarı
};

// Evrensel Kaggle etiketlerini klinik Türkçeye çeviren sözlük
const SINIF_CEVIRILERI = {
    "Caries": "Çürük",
    "Periapical lesion": "Kök Ucu İltihabı / Lezyon",
    "Impacted tooth": "Gömülü Diş",
    "Bone Loss": "Kemik Kaybı (Periodontal)",
    "Fracture teeth": "Kırık Diş",
    "Bone defect": "Kemik Defekti",
    "Cyst": "Kist",
    "Root resorption": "Kök Erimesi"
};

// --- SUNUCU SAĞLIK KONTROLÜ ---
fetch("/saglik-kontrolu")
    .then((r) => r.json())
    .then((durum) => {
        const etiket = document.getElementById("backend-etiketi");
        if (durum.model_yuklendi) {
            etiket.textContent = `SİSTEM AKTİF (${durum.backend.toUpperCase()})`;
            etiket.style.color = "var(--c-basari)";
        } else {
            etiket.textContent = "MODEL YÜKLENMEDİ";
            etiket.style.color = "var(--c-amber)";
        }
    })
    .catch(() => {
        document.getElementById("backend-etiketi").textContent = "BAĞLANTI HATASI";
    });

// --- DOSYA SEÇİMİ VE ÖNİZLEME ---
dropzoneInput.addEventListener("change", () => {
    const dosya = dropzoneInput.files[0];
    if (!dosya) return;
    const url = URL.createObjectURL(dosya);
    onizlemeGoruntu.src = url;
    onizlemeGoruntu.hidden = false;
    yuklemeIpucu.hidden = true;
});

// --- FORM GÖNDERİMİ VE API İSTEĞİ ---
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hataDurum.hidden = true;

    const formData = new FormData(form);

    // Buton yükleniyor durumuna geçiş
    analizButonu.disabled = true;
    analizButonu.querySelector(".btn-metin").hidden = true;
    analizButonu.querySelector(".btn-yukleniyor").hidden = false;

    try {
        const yanit = await fetch("/analiz", { method: "POST", body: formData });
        const veri = await yanit.json();

        if (!yanit.ok) {
            throw new Error(veri.detail || "Analiz sırasında sunucu kaynaklı bir hata oluştu.");
        }

        sonucuGoster(veri);
    } catch (hata) {
        hataDurum.textContent = `Hata: ${hata.message}`;
        hataDurum.hidden = false;
    } finally {
        // Butonu normal duruma al
        analizButonu.disabled = false;
        analizButonu.querySelector(".btn-metin").hidden = false;
        analizButonu.querySelector(".btn-yukleniyor").hidden = true;
    }
});

// --- SONUÇLARI ARAYÜZE (DOM) YANSITMA ---
function sonucuGoster(veri) {
    bosDurum.hidden = true;
    sonucIcerik.hidden = false;

    // Görsel ve link güncellemeleri
    document.getElementById("sonuc-goruntu").src = veri.isaretlenmis_goruntu_url;
    document.getElementById("tespit-sayisi").textContent = veri.toplam_tespit_sayisi;
    document.getElementById("pdf-link").href = veri.rapor_pdf_url;

    const liste = document.getElementById("tespit-listesi");
    liste.innerHTML = "";

    if (veri.tespitler.length === 0) {
        const bosLi = document.createElement("li");
        bosLi.textContent = "Seçili patoloji kategorilerinde belirgin bir anomali tespit edilmedi.";
        bosLi.style.color = "var(--c-metin-soluk)";
        liste.appendChild(bosLi);
        return;
    }

    // Tespitleri listeye ekle (Çeviri ve Renk eşleştirmesi ile)
    veri.tespitler.forEach((t) => {
        const li = document.createElement("li");
        const renk = SINIF_RENKLERI[t.sinif_adi] || "#ffffff";
        const turkceIsim = SINIF_CEVIRILERI[t.sinif_adi] || t.sinif_adi; // Çeviri yoksa orijinalini bas

        li.innerHTML = `
      <span class="tespit-sinif">
        <span class="renk-noktasi" style="background:${renk}; box-shadow: 0 0 8px ${renk}66;"></span>
        ${turkceIsim}
      </span>
      <span class="tespit-guven">%${(t.guven_skoru * 100).toFixed(1)}</span>
    `;
        liste.appendChild(li);
    });
}