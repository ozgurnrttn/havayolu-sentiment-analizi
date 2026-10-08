"""Toplanan yorumlardan analiz kümesini çıkarır ve etiketleme dosyalarını hazırlar.

Girdi : data/youtube_yorumlar.csv, data/videolar.csv   (01_yorum_topla.py çıktısı)
        video_turleri.csv                              (her videonun elle verilmiş türü)
Çıktı : data/analiz_kumesi.csv        etiketlenecek yorumlar (metinle birlikte; depoya girmez)
        etiket/anahtar.csv            numara -> yorum_id eşlemesi
        etiket/girdi/b01.txt ...      200'er yorumluk etiketleme dosyaları
        etiket/girdi/k1.txt, k2.txt   ikinci geçiş için rastgele 400 yorum
        outputs/hazirlik.json         her adımda kalan yorum sayısı

Kurulum: pip install pandas numpy lingua-language-detector
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

DILLER = [Language.TURKISH, Language.ENGLISH, Language.GERMAN, Language.RUSSIAN, Language.ARABIC,
          Language.SPANISH, Language.AZERBAIJANI, Language.FRENCH, Language.INDONESIAN,
          Language.PORTUGUESE, Language.PERSIAN, Language.HINDI, Language.ITALIAN, Language.DUTCH]
TUR_ADI = {"resmi": "reklam ya da marka filmi", "kaza": "kaza anlatısı",
           "haber": "haber, analiz ya da bilgi", "ucus": "uçuş videosu ya da inceleme"}
PARTI = 200          # bir etiketleme dosyasındaki yorum sayısı
IKINCI_GECIS = 400   # tutarlılık ölçümü için ikinci kez etiketlenen yorum sayısı


def dil_bul(dedektor, metin: str) -> str:
    """Bağlantı ve kullanıcı adları atıldıktan sonra en az 3 harf kalmıyorsa 'bos'."""
    s = re.sub(r"https?://\S+|@\S+", "", str(metin))
    if sum(c.isalpha() for c in s) < 3:
        return "bos"
    r = dedektor.detect_language_of(s)
    return r.iso_code_639_1.name.lower() if r else "?"


def dosya_yaz(df: pd.DataFrame, yol: Path) -> None:
    """Yorumlar video başlığının altında gruplanır: etiketleyen bağlamı görsün."""
    satirlar, son = [], None
    for r in df.itertuples():
        if r.video_id != son:
            satirlar.append(f"\n## VIDEO: {r.baslik} | kanal: {r.kanal} | marka: {r.marka} | tür: {TUR_ADI[r.tur]}")
            son = r.video_id
        satirlar.append(f"{r.n}\t{r.t}")
    yol.write_text("\n".join(satirlar).strip() + "\n", encoding="utf-8")


def main() -> None:
    ham = pd.read_csv("data/youtube_yorumlar.csv")
    turler = pd.read_csv("video_turleri.csv")[["video_id", "tur", "baslik", "kanal"]]
    adim = [("Toplanan yorum", len(ham))]

    y = ham.drop_duplicates("yorum_id")
    adim.append(("Tekrarlar atıldı", len(y)))

    dedektor = LanguageDetectorBuilder.from_languages(*DILLER).build()
    y = y.assign(dil=y["metin"].map(lambda t: dil_bul(dedektor, t)))
    dil_dagilimi = y["dil"].value_counts().to_dict()
    y = y[y["dil"] == "tr"]
    adim.append(("Türkçe olanlar", len(y)))

    y = y.merge(turler, on="video_id")
    y = y[y["tur"] != "alakasiz"]
    adim.append(("Markayla ilgisiz videolar çıkarıldı", len(y)))

    # Yanıtlar başka bir yorumcuya yazılır; üstteki yorumu görmeden etiketlemek güvenilir olmaz.
    a = y[y["yanit_mi"] == 0].copy()
    adim.append(("Yanıtlar çıkarıldı (analiz kümesi)", len(a)))

    a["t"] = a["metin"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 500)
    a = a.sort_values(["marka", "video_id"], kind="stable").reset_index(drop=True)
    a["n"] = np.arange(1, len(a) + 1)

    Path("etiket/girdi").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    a[["n", "yorum_id", "marka", "video_id", "tur"]].to_csv("etiket/anahtar.csv", index=False)
    a[["n", "yorum_id", "marka", "video_id", "tur", "baslik", "kanal", "begeni", "tarih", "t"]].to_csv(
        "data/analiz_kumesi.csv", index=False, encoding="utf-8-sig")

    for i in range(int(np.ceil(len(a) / PARTI))):
        dosya_yaz(a.iloc[i * PARTI:(i + 1) * PARTI], Path(f"etiket/girdi/b{i + 1:02d}.txt"))
    ikinci = a.sample(IKINCI_GECIS, random_state=7).sort_values("n")
    ikinci[["n"]].to_csv("etiket/ikinci_gecis_n.csv", index=False)
    for i in range(2):
        dosya_yaz(ikinci.iloc[i * 200:(i + 1) * 200], Path(f"etiket/girdi/k{i + 1}.txt"))

    v = pd.read_csv("data/videolar.csv").merge(turler[["video_id", "tur"]], on="video_id")
    ozet = {
        "adimlar": [{"adim": ad, "yorum": int(n)} for ad, n in adim],
        "dil": {k: int(n) for k, n in dil_dagilimi.items()},
        "video": {"bulunan": int(len(v)), "yorumsuz": int((v["yorum_sayisi"] == 0).sum()),
                  "tur": {k: int(n) for k, n in v["tur"].value_counts().items()},
                  "analizde": int(a["video_id"].nunique())},
    }
    Path("outputs/hazirlik.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    for ad, n in adim:
        print(f"{n:>7,}  {ad}")
    print(f"{a['video_id'].nunique()} video, {int(np.ceil(len(a) / PARTI))} etiketleme dosyası")


if __name__ == "__main__":
    main()
