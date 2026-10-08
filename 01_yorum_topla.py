"""YouTube'dan havayolu videolarının Türkçe yorumlarını toplar.

Çıktı:
    data/youtube_yorumlar.csv   her satır bir yorum
    data/videolar.csv           taranan videolar (yorumu kapalı olanlar dahil)

Kurulum:
    pip install youtube-comment-downloader scrapetube pandas

Kullanım:
    python 01_yorum_topla.py                       # marka başına 40 video, video başına 300 yorum
    python 01_yorum_topla.py --video 60 --yorum 500

API anahtarı gerekmez. Yarıda kesilirse tekrar çalıştır: bitmiş videoları atlar.
"""
import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import scrapetube
from youtube_comment_downloader import SORT_BY_RECENT, YoutubeCommentDownloader

# Marka -> başlıkta aranacak yazılışlar. Başka sektör için sadece burayı değiştir.
MARKALAR = {
    "THY": ["thy", "türk hava yolları", "turk hava yollari", "turkish airlines"],
    "Pegasus": ["pegasus"],
    "AJet": ["ajet", "anadolujet", "anadolu jet"],
}

# Arama adı: aramada markanın nasıl yazılacağı.
ARAMA_ADI = {"THY": "Türk Hava Yolları", "Pegasus": "Pegasus Hava Yolları", "AJet": "AJet"}

# Her markaya aynı kalıplar soruluyor ki karşılaştırma adil olsun.
# Bilerek nötr: "rötar", "şikayet" gibi kelimeler sonucu olumsuza çekerdi.
KALIPLAR = ["{m}", "{m} uçuş deneyimi", "{m} inceleme", "{m} ile uçmak", "{m} yolculuk vlog"]

VIDEO_ALAN = ["marka", "video_id", "baslik", "kanal", "yayin", "izlenme", "sorgu", "diger_marka", "yorum_sayisi", "durum"]
YORUM_ALAN = ["marka", "video_id", "yorum_id", "metin", "begeni", "yanit_sayisi", "yanit_mi", "zaman_metni", "tarih"]


def kucuk(s: str) -> str:
    """Türkçe büyük/küçük harf: İ -> i, I -> ı."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def metin(d: dict, anahtar: str) -> str:
    """YouTube alanları ya {'simpleText': ..} ya da {'runs': [{'text': ..}]} gelir."""
    v = d.get(anahtar) or {}
    if "simpleText" in v:
        return v["simpleText"]
    return "".join(r.get("text", "") for r in v.get("runs", []))


def markalar_basliktan(baslik: str) -> list[str]:
    b = kucuk(baslik)
    return [m for m, yazilis in MARKALAR.items() if any(y in b for y in yazilis)]


def video_bul(marka: str, hedef: int, haric: set[str]) -> list[dict]:
    """Aramalardan, başlığında markanın adı geçen ve daha önce alınmamış videoları toplar."""
    bulunan: dict[str, dict] = {}
    for kalip in KALIPLAR:
        if len(bulunan) >= hedef:
            break
        sorgu = kalip.format(m=ARAMA_ADI[marka])
        try:
            for v in scrapetube.get_search(sorgu, limit=60, sleep=1):
                vid = v.get("videoId")
                baslik = metin(v, "title")
                gecen = markalar_basliktan(baslik)
                if not vid or vid in bulunan or vid in haric or marka not in gecen:
                    continue
                bulunan[vid] = {
                    "marka": marka,
                    "video_id": vid,
                    "baslik": baslik,
                    "kanal": metin(v, "ownerText") or metin(v, "longBylineText"),
                    "yayin": metin(v, "publishedTimeText"),
                    "izlenme": metin(v, "viewCountText"),
                    "sorgu": sorgu,
                    "diger_marka": ",".join(m for m in gecen if m != marka),
                }
                if len(bulunan) >= hedef:
                    break
        except Exception as e:  # tek bir arama bozulursa diğerleri devam etsin
            print(f"    arama atlandı ({sorgu}): {e}")
        print(f"    {sorgu!r}: toplam {len(bulunan)} video")
    return list(bulunan.values())


def yorum_cek(dl: YoutubeCommentDownloader, vid: str, ust_sinir: int) -> list[dict]:
    """En yeniden eskiye. Yazar adı ve profil fotoğrafı bilerek alınmıyor."""
    satirlar = []
    # language='en': "3 months ago" gibi zaman metni tarihe güvenilir çevrilsin diye.
    for c in dl.get_comments(vid, sort_by=SORT_BY_RECENT, language="en", sleep=0.2):
        ts = c.get("time_parsed")
        satirlar.append({
            "video_id": vid,
            "yorum_id": c["cid"],
            "metin": c["text"],
            "begeni": c.get("votes", "0"),
            "yanit_sayisi": c.get("replies", 0),
            "yanit_mi": int(bool(c.get("reply"))),
            "zaman_metni": c.get("time", ""),
            "tarih": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") if ts else "",
        })
        if len(satirlar) >= ust_sinir:
            break
    return satirlar


def ekle(yol: Path, alanlar: list[str], satirlar: list[dict]) -> None:
    yeni = not yol.exists()
    with yol.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        if yeni:
            w.writeheader()
        w.writerows(satirlar)


def bitenler(yol: Path) -> dict[str, str]:
    """Önceki çalıştırmada bitmiş videolar: video_id -> marka."""
    if not yol.exists():
        return {}
    with yol.open(encoding="utf-8-sig", newline="") as f:
        return {r["video_id"]: r["marka"] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, default=40, help="marka başına video sayısı")
    ap.add_argument("--yorum", type=int, default=300, help="video başına en çok yorum")
    ap.add_argument("--klasor", default="data")
    args = ap.parse_args()

    klasor = Path(args.klasor)
    klasor.mkdir(parents=True, exist_ok=True)
    v_yol, y_yol = klasor / "videolar.csv", klasor / "youtube_yorumlar.csv"
    onceki = bitenler(v_yol)
    gorulen = set(onceki)
    dl = YoutubeCommentDownloader()
    toplam = {}

    for marka in MARKALAR:
        eksik = args.video - sum(1 for m in onceki.values() if m == marka)
        if eksik <= 0:
            print(f"\n{marka}: önceki çalıştırmada tamamlanmış")
            toplam[marka] = 0
            continue
        print(f"\n{marka}: video aranıyor")
        videolar = video_bul(marka, eksik, gorulen)
        gorulen.update(v["video_id"] for v in videolar)
        n = 0
        for i, v in enumerate(videolar, 1):
            try:
                satirlar = yorum_cek(dl, v["video_id"], args.yorum)
            except Exception as e:  # kaydetme: tekrar çalıştırınca yeniden denensin
                print(f"  {i:>2}/{len(videolar)}  atlandı ({e})  {v['baslik'][:50]}")
                continue
            durum = "tamam" if satirlar else "yorum yok ya da kapalı"
            for s in satirlar:
                s["marka"] = marka
            ekle(y_yol, YORUM_ALAN, satirlar)
            ekle(v_yol, VIDEO_ALAN, [{**v, "yorum_sayisi": len(satirlar), "durum": durum}])
            n += len(satirlar)
            print(f"  {i:>2}/{len(videolar)}  {len(satirlar):>4} yorum  {v['baslik'][:60]}")
            time.sleep(1)  # sunucuyu yormamak için
        toplam[marka] = n

    print("\nBu çalıştırmada toplanan yorum:")
    for m, n in toplam.items():
        print(f"  {m:<8} {n}")
    if onceki:
        print(f"  (önceki çalıştırmadan {len(onceki)} video duruyor)")
    print(f"\nDosya: {y_yol.resolve()}")
    if not sum(toplam.values()) and not onceki:
        raise SystemExit("Hiç yorum gelmedi. İnternet bağlantısını kontrol edip tekrar dene.")


if __name__ == "__main__":
    main()
