"""Etiketlenmiş yorumların analizi ve hafif sınıflandırıcı denemesi.

Girdi : data/analiz_kumesi.csv   (02_hazirla.py çıktısı; yorum metinleri)
        etiket/etiketler.csv     (yorum başına duygu, hedef, konu; 400 yorumda ikinci geçiş)
Çıktı : outputs/summary.json, outputs/*.png

Kurulum: pip install pandas numpy scikit-learn matplotlib
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, KFold

SEED = 42
OUT = Path("outputs")
MARKALAR = ["THY", "Pegasus", "AJet"]
TURLER = ["resmi", "haber", "ucus", "kaza"]
TUR_ADI = {"resmi": "Reklam ve marka filmi", "haber": "Haber ve analiz", "ucus": "Uçuş videosu", "kaza": "Kaza anlatısı"}
HEDEFLER = ["marka", "video", "diger"]
HEDEF_ADI = {"marka": "Marka", "video": "Video", "diger": "Diğer"}
DUYGULAR = ["olumlu", "notr", "olumsuz"]
KONULAR = ["genel", "guvenlik", "yonetim", "fiyat", "operasyon", "hizmet", "konfor"]
KONU_ADI = {"genel": "Genel", "guvenlik": "Güvenlik", "yonetim": "Yönetim", "fiyat": "Fiyat",
            "operasyon": "Operasyon", "hizmet": "Hizmet", "konfor": "Konfor"}
ACCENT, GRAY, WARM = "#6d28d9", "#8a8799", "#d95926"

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e1ec", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "font.size": 10})

# Sözlük tabanı: en basit yaklaşım, olumlu ve olumsuz kelime köklerini saymak.
OLUMLU = ("güzel harika süper mükemmel muhteşem efsane bravo tebrik teşekkür sağol helal gurur başarı "
          "sev bayıl beğen iyi kaliteli rahat güvenli memnun keyif mutlu duygulan ağla tüyler "
          "en iyi kral müthiş şahane enfes eline emeğine yürek ❤ 👏 😍 🥰 👍").split()
OLUMSUZ = ("kötü berbat rezil rezalet felaket saçma yazık utan ayıp pahalı kazık soygun rötar iptal "
           "gecik kayıp mağdur şikayet şikâyet saygısız kaba ilgisiz bıktım nefret iğrenç korkunç "
           "tehlike kaza hata ihmal beceriksiz liyakat batır yalan dolandır boykot bir daha asla "
           "binmem uçmam maalesef sorun problem skandal 👎 😡 🤮").split()


def kucuk(s: str) -> str:
    return str(s).replace("İ", "i").replace("I", "ı").lower()


def oran(x: float, basamak: int = 1) -> float:
    return round(float(x) * 100, basamak)


def sozluk_tahmin(metinler: pd.Series) -> np.ndarray:
    out = []
    for t in metinler.map(kucuk):
        p = sum(t.count(k) for k in OLUMLU)
        n = sum(t.count(k) for k in OLUMSUZ)
        out.append("olumlu" if p > n else "olumsuz" if n > p else "notr")
    return np.array(out)


def kume_bootstrap(df: pd.DataFrame, deger: str, tekrar: int = 2000, seed: int = SEED) -> np.ndarray:
    """Yorumlar videolara kümelenmiş olduğu için yorum değil video yeniden örneklenir."""
    g = df.groupby("video_id")[deger].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(tekrar, len(g)))
    return g[idx, 0].sum(axis=1) / g[idx, 1].sum(axis=1)


def aralik(df: pd.DataFrame, deger: str) -> dict:
    b = kume_bootstrap(df, deger)
    return {"n": int(len(df)), "video": int(df["video_id"].nunique()), "oran": oran(df[deger].mean()),
            "alt": oran(np.percentile(b, 2.5)), "ust": oran(np.percentile(b, 97.5))}


def ozellik(egitim: pd.DataFrame, test: pd.DataFrame, tur_ekle: bool):
    kelime = TfidfVectorizer(preprocessor=kucuk, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    harf = TfidfVectorizer(preprocessor=kucuk, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    Xe = [kelime.fit_transform(egitim["t"]), harf.fit_transform(egitim["t"])]
    Xt = [kelime.transform(test["t"]), harf.transform(test["t"])]
    if tur_ekle:
        for X, d in ((Xe, egitim), (Xt, test)):
            X.append(csr_matrix(np.column_stack([(d["tur"] == t).to_numpy(float) for t in TURLER]) * 2.0))
    return hstack(Xe).tocsr(), hstack(Xt).tocsr()


def cv_tahmin(d: pd.DataFrame, hedef: str, katlar, tur_ekle: bool = False, egitim_n: int | None = None) -> np.ndarray:
    """Kat dışı tahminler. egitim_n verilirse her katta eğitim kümesinden o kadar yorum örneklenir."""
    tahmin = np.empty(len(d), dtype=object)
    for i, (e, t) in enumerate(katlar):
        if egitim_n is not None and egitim_n < len(e):
            e = np.random.default_rng(SEED + i).choice(e, egitim_n, replace=False)
        Xe, Xt = ozellik(d.iloc[e], d.iloc[t], tur_ekle)
        m = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced")
        tahmin[t] = m.fit(Xe, d[hedef].iloc[e]).predict(Xt)
    return tahmin


def skor(gercek, tahmin) -> dict:
    return {"dogruluk": oran(accuracy_score(gercek, tahmin)), "makro_f1": round(float(f1_score(gercek, tahmin, average="macro")), 3)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    S = json.loads((OUT / "hazirlik.json").read_text(encoding="utf-8")) if (OUT / "hazirlik.json").exists() else {}
    a = pd.read_csv("data/analiz_kumesi.csv")
    e = pd.read_csv("etiket/etiketler.csv")
    # Veri yeniden toplandıysa yalnızca etiketi olan yorumlar kullanılır.
    d = a.merge(e[["yorum_id", "duygu", "hedef", "konu", "duygu_2", "hedef_2", "konu_2"]], on="yorum_id", validate="1:1")
    print(f"{len(d):,} etiketli yorum bulundu (etiket dosyasında {len(e):,})")
    d["neg"] = (d["duygu"] == "olumsuz").astype(int)
    d["poz"] = (d["duygu"] == "olumlu").astype(int)
    d["begeni_n"] = d["begeni"].astype(str).map(
        lambda s: int(float(s[:-1]) * {"K": 1e3, "M": 1e6}[s[-1]]) if s[-1:] in ("K", "M") else int(float(s or 0)))
    m = d[d["hedef"] == "marka"]

    S["genel"] = {"yorum": int(len(d)), "video": int(d["video_id"].nunique()),
                  "marka_yorum": {b: int((d["marka"] == b).sum()) for b in MARKALAR},
                  "marka_video": {b: int(d.loc[d["marka"] == b, "video_id"].nunique()) for b in MARKALAR},
                  "tur_yorum": {t: int((d["tur"] == t).sum()) for t in TURLER},
                  "marka_tur": {b: {t: int(((d["marka"] == b) & (d["tur"] == t)).sum()) for t in TURLER} for b in MARKALAR},
                  "tarih": [d["tarih"].min(), d["tarih"].max()],
                  "son_iki_yil": oran((pd.to_datetime(d["tarih"]).dt.year >= 2025).mean())}

    # ---------- Etiketlerin tutarlılığı: 400 yorum ikinci kez, bağımsız etiketlendi ----------
    k = d.dropna(subset=["duygu_2"])
    S["tutarlilik"] = {"n": int(len(k))}
    for c in ("duygu", "hedef", "konu"):
        S["tutarlilik"][c] = {"uyum": oran((k[c] == k[c + "_2"]).mean()), "kappa": round(float(cohen_kappa_score(k[c], k[c + "_2"])), 2)}
    uc = (k["duygu"] == k["duygu_2"]) & (k["hedef"] == k["hedef_2"]) & (k["konu"] == k["konu_2"])
    S["tutarlilik"]["ucu_birden"] = oran(uc.mean())

    # ---------- Yorum kimin hakkında? ----------
    S["hedef"] = {"hepsi": {h: oran((d["hedef"] == h).mean()) for h in HEDEFLER},
                  "n": {h: int((d["hedef"] == h).sum()) for h in HEDEFLER},
                  "tur": {t: {h: oran((d.loc[d["tur"] == t, "hedef"] == h).mean()) for h in HEDEFLER} for t in TURLER},
                  "marka": {b: {h: oran((d.loc[d["marka"] == b, "hedef"] == h).mean()) for h in HEDEFLER} for b in MARKALAR}}
    S["hedef_duygu"] = {h: {s: oran((d.loc[d["hedef"] == h, "duygu"] == s).mean()) for s in DUYGULAR} for h in HEDEFLER}

    # ---------- Hedefi yok sayınca ne değişiyor? ----------
    S["saf_hedefli"] = {}
    for b in MARKALAR:
        hep, hed = d[d["marka"] == b], m[m["marka"] == b]
        S["saf_hedefli"][b] = {
            "saf": {"n": int(len(hep)), "olumlu": oran(hep["poz"].mean()), "olumsuz": oran(hep["neg"].mean())},
            "hedefli": {"n": int(len(hed)), "olumlu": oran(hed["poz"].mean()), "olumsuz": oran(hed["neg"].mean()),
                        "notr": oran((hed["duygu"] == "notr").mean())},
            "aralik": aralik(hed, "neg")}

    # ---------- Marka mı, video türü mü? ----------
    S["tur_olumsuz"] = {t: aralik(m[m["tur"] == t], "neg") for t in TURLER}
    S["marka_tur"] = {}
    for b in MARKALAR:
        S["marka_tur"][b] = {}
        for t in TURLER:
            x = m[(m["marka"] == b) & (m["tur"] == t)]
            S["marka_tur"][b][t] = aralik(x, "neg") if len(x) >= 30 else {"n": int(len(x)), "video": int(x["video_id"].nunique())}
    # Aynı tür içinde marka farkı (haber): video bootstrap ile fark aralığı
    S["haber_fark"] = {}
    h = m[m["tur"] == "haber"]
    for b in ("Pegasus", "AJet"):
        f = kume_bootstrap(h[h["marka"] == b], "neg", seed=1) - kume_bootstrap(h[h["marka"] == "THY"], "neg", seed=2)
        S["haber_fark"][b] = {"fark": round(float(h.loc[h["marka"] == b, "neg"].mean() - h.loc[h["marka"] == "THY", "neg"].mean()) * 100, 1),
                              "alt": round(float(np.percentile(f, 2.5)) * 100, 1), "ust": round(float(np.percentile(f, 97.5)) * 100, 1)}
    # Standardizasyon: marka, THY'nin video türü karışımına sahip olsaydı olumsuz payı ne olurdu?
    # Yalnızca markanın en az 30 yorumu olan türler kullanılır; THY de aynı türlerle yeniden hesaplanır.
    S["standart"] = {}
    thy = m[m["marka"] == "THY"]
    for b in ("Pegasus", "AJet"):
        x = m[m["marka"] == b]
        ortak = [t for t in TURLER if (x["tur"] == t).sum() >= 30]
        w = np.array([(thy["tur"] == t).sum() for t in ortak], dtype=float)
        w /= w.sum()
        nokta = float(sum(wi * x.loc[x["tur"] == t, "neg"].mean() for wi, t in zip(w, ortak)))
        boot = sum(wi * kume_bootstrap(x[x["tur"] == t], "neg", seed=10 + i) for i, (wi, t) in enumerate(zip(w, ortak)))
        to = thy[thy["tur"].isin(ortak)]
        S["standart"][b] = {"turler": ortak, "ham": oran(x["neg"].mean()), "ham_ortak": oran(x.loc[x["tur"].isin(ortak), "neg"].mean()),
                            "standart": oran(nokta), "alt": oran(np.percentile(boot, 2.5)), "ust": oran(np.percentile(boot, 97.5)),
                            "thy": oran(to["neg"].mean()), "thy_n": int(len(to))}
    S["tur_karisimi"] = {b: {t: oran(((m["marka"] == b) & (m["tur"] == t)).sum() / (m["marka"] == b).sum()) for t in TURLER} for b in MARKALAR}
    # Yoğunlaşma: en çok yorum alan üç videonun payı
    S["yogunlasma"] = {b: oran(m.loc[m["marka"] == b, "video_id"].value_counts().head(3).sum() / (m["marka"] == b).sum()) for b in MARKALAR}

    # ---------- Konular ----------
    S["konu"] = {"hepsi": [], "marka": {}}
    for kk in KONULAR:
        x = m[m["konu"] == kk]
        S["konu"]["hepsi"].append({"konu": kk, "n": int(len(x)), "pay": oran(len(x) / len(m)), "olumsuz": oran(x["neg"].mean()), "olumlu": oran(x["poz"].mean())})
    for b in MARKALAR:
        x = m[m["marka"] == b]
        S["konu"]["marka"][b] = {kk: {"n": int((x["konu"] == kk).sum()), "pay": oran((x["konu"] == kk).mean()),
                                      "olumsuz_pay": oran(((x["konu"] == kk) & (x["neg"] == 1)).sum() / max(x["neg"].sum(), 1))}
                                 for kk in KONULAR}

    # ---------- Beğeniler ----------
    S["begeni"] = {s: {"ortalama": round(float(m.loc[m["duygu"] == s, "begeni_n"].mean()), 1),
                       "en_az_bir": oran((m.loc[m["duygu"] == s, "begeni_n"] > 0).mean())} for s in DUYGULAR}

    # ---------- Hafif sınıflandırıcı ----------
    d = d.reset_index(drop=True)
    grup = list(GroupKFold(n_splits=5).split(d, groups=d["video_id"]))
    rastgele = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(d))
    d["marka_olumsuz"] = np.where((d["hedef"] == "marka") & (d["neg"] == 1), "evet", "hayir")
    M = {}
    for hedef in ("duygu", "hedef"):
        cogunluk = np.full(len(d), d[hedef].mode().iat[0])
        M[hedef] = {"cogunluk": skor(d[hedef], cogunluk)}
        tah = cv_tahmin(d, hedef, grup)
        d[hedef + "_tah"] = tah
        M[hedef]["model"] = skor(d[hedef], tah)
        M[hedef]["model_tur"] = skor(d[hedef], cv_tahmin(d, hedef, grup, tur_ekle=True))
        M[hedef]["model_rastgele"] = skor(d[hedef], cv_tahmin(d, hedef, rastgele))
    M["duygu"]["sozluk"] = skor(d["duygu"], sozluk_tahmin(d["t"]))
    # Öğrenme eğrisi: kaç etiket yeter?
    M["egri"] = []
    for n in (250, 500, 1000, 2000, 4000, None):
        M["egri"].append({"n": n or int(np.mean([len(e) for e, _ in grup])),
                          "duygu": skor(d["duygu"], cv_tahmin(d, "duygu", grup, egitim_n=n))["makro_f1"],
                          "hedef": skor(d["hedef"], cv_tahmin(d, "hedef", grup, egitim_n=n))["makro_f1"]})
    # Asıl soru: model, markaya yönelik olumsuz payını tutturuyor mu?
    M["toplam"] = {}
    for b in MARKALAR:
        x = d[d["marka"] == b]
        gh, th = x[x["hedef"] == "marka"], x[x["hedef_tah"] == "marka"]
        M["toplam"][b] = {"etiket": oran(gh["neg"].mean()), "model": oran((th["duygu_tah"] == "olumsuz").mean()),
                          "etiket_n": int(len(gh)), "model_n": int(len(th)),
                          "sozluk_saf": oran((sozluk_tahmin(x["t"]) == "olumsuz").mean())}
    S["model"] = M

    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding="utf-8")
    grafikler(d, m, S)
    print(json.dumps({k: S[k] for k in ("tutarlilik", "saf_hedefli", "standart")}, ensure_ascii=False, indent=1))


def grafikler(d: pd.DataFrame, m: pd.DataFrame, S: dict) -> None:
    # 1) Yorum kimin hakkında: video türüne göre
    fig, ax = plt.subplots(figsize=(7, 2.9))
    sol = np.zeros(len(TURLER))
    for h, renk in zip(HEDEFLER, (ACCENT, "#b9a3e8", "#d9d6e2")):
        v = np.array([S["hedef"]["tur"][t][h] for t in TURLER])
        ax.barh([TUR_ADI[t] for t in TURLER], v, left=sol, color=renk, height=0.6, label=HEDEF_ADI[h], edgecolor="white", linewidth=1.5)
        for i, (x0, w) in enumerate(zip(sol, v)):
            ax.text(x0 + w / 2, i, f"%{w:.0f}", ha="center", va="center", fontsize=9, color="white" if h == "marka" else "#2a2833")
        sol += v
    ax.set(xlim=(0, 100), xlabel="Yorumların payı (%)", title="Yorum kimin hakkında?")
    ax.invert_yaxis(); ax.grid(False); ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    plt.tight_layout(); fig.savefig(OUT / "1_hedef.png"); plt.close(fig)

    # 2) Hedefi yok sayınca olumsuz payı
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for i, b in enumerate(MARKALAR):
        s, h = S["saf_hedefli"][b]["saf"]["olumsuz"], S["saf_hedefli"][b]["hedefli"]["olumsuz"]
        ax.plot([s, h], [i, i], color="#c9c5d6", linewidth=2, zorder=1)
        ax.scatter([s], [i], color=GRAY, s=55, zorder=2, label="Bütün yorumlar" if i == 0 else None)
        ax.scatter([h], [i], color=ACCENT, s=55, zorder=2, label="Yalnızca markaya yönelik" if i == 0 else None)
        ax.text(s - 2, i, f"%{s:.0f}", ha="right", va="center", fontsize=9); ax.text(h + 2, i, f"%{h:.0f}", ha="left", va="center", fontsize=9)
    ax.set(yticks=range(3), yticklabels=MARKALAR, xlim=(0, 80), xlabel="Olumsuz yorum payı (%)", title="Hedefi yok sayınca olumsuz payı düşük görünüyor")
    ax.invert_yaxis(); ax.legend(frameon=False, loc="lower right")
    plt.tight_layout(); fig.savefig(OUT / "2_saf_hedefli.png"); plt.close(fig)

    # 3) Marka x video türü: markaya yönelik olumsuz payı, video bootstrap aralığıyla
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for ax, t in zip(axes, TURLER):
        for i, b in enumerate(MARKALAR):
            r = S["marka_tur"][b][t]
            if "oran" not in r:
                ax.text(50, i, "az veri", ha="center", va="center", fontsize=8, color=GRAY); continue
            ax.plot([r["alt"], r["ust"]], [i, i], color="#c9c5d6", linewidth=2)
            ax.scatter([r["oran"]], [i], color=ACCENT, s=45, zorder=3)
            ax.text(r["oran"], i - 0.28, f"%{r['oran']:.0f}", ha="center", fontsize=8)
        ax.set(title=TUR_ADI[t], xlim=(0, 100), yticks=range(3), yticklabels=MARKALAR, ylim=(2.6, -0.6))
        ax.title.set_fontsize(10)
    fig.supxlabel("Markaya yönelik yorumlarda olumsuz payı (%), %95 aralık", fontsize=9)
    plt.tight_layout(); fig.savefig(OUT / "3_marka_tur.png"); plt.close(fig)

    # 4) Olumsuz yorumlar hangi konuda?
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, b in zip(axes, MARKALAR):
        v = [S["konu"]["marka"][b][kk]["olumsuz_pay"] for kk in KONULAR]
        ax.barh([KONU_ADI[kk] for kk in KONULAR], v, color=ACCENT, height=0.6)
        for i, x in enumerate(v):
            ax.text(x + 1, i, f"%{x:.0f}", va="center", fontsize=8)
        ax.set(title=b, xlim=(0, 60)); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.supxlabel("Markaya yönelik olumsuz yorumlar içindeki pay (%)", fontsize=9)
    plt.tight_layout(); fig.savefig(OUT / "4_konu.png"); plt.close(fig)

    # 5) Öğrenme eğrisi
    fig, ax = plt.subplots(figsize=(6, 3))
    e = S["model"]["egri"]
    ax.plot([r["n"] for r in e], [r["duygu"] for r in e], marker="o", color=ACCENT, linewidth=2, label="Duygu")
    ax.plot([r["n"] for r in e], [r["hedef"] for r in e], marker="o", color=WARM, linewidth=2, label="Hedef")
    ax.set(xscale="log", xlabel="Eğitimde kullanılan etiketli yorum", ylabel="Makro F1", ylim=(0.3, 0.9), title="Kaç etiket yetiyor?")
    ax.set_xticks([r["n"] for r in e]); ax.set_xticklabels([f"{r['n']:,}".replace(",", ".") for r in e]); ax.minorticks_off()
    ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "5_ogrenme_egrisi.png"); plt.close(fig)


if __name__ == "__main__":
    main()
