# Etiketleme kılavuzu

Her yorum için üç etiket verilir: **duygu**, **hedef**, **konu**.
Yorumlar YouTube'da havayolu şirketleriyle (THY, Pegasus, AJet) ilgili videoların altına yazılmış Türkçe yorumlardır.
Her yorum, altına yazıldığı videonun başlığıyla birlikte verilir; yorumu o bağlamda oku.

## 1. Hedef: yorum kimin ya da neyin hakkında?

- `marka`: Havayolu şirketinin kendisi. Hizmeti, fiyatı, personeli, uçakları, güvenliği, yönetimi,
  kararları ya da genel olarak marka ("gururumuz", "bir daha uçmam").
- `video`: Videonun kendisi. Reklam filmi, müziği, oyuncuları, çekimi, videoyu yapan kanal ya da
  anlatıcı ("reklam çok güzel olmuş", "emeğinize sağlık hocam", "bu müzik efsane").
- `diger`: İkisi de değil. Siyaset, ülke ya da millet, başka şirketler ve kişiler, kaza kurbanları
  için dua ve taziye, genel havacılık bilgisi, başka bir yorumcuyla sohbet, şirketi değerlendirmeyen
  kişisel anı, spam, anlamsız ya da Türkçe olmayan metin.

Kararsız kalınca: yorum şirketi değerlendiriyorsa (reklam üzerinden bile olsa) `marka`.
Yalnızca videoyu, reklamı ya da müziği değerlendiriyorsa `video`.
Bir kaza videosunda pilotu, bakımı ya da şirketin sorumluluğunu tartışan yorum `marka`;
yalnızca üzüntü ya da dua bildiren yorum `diger`.

## 2. Duygu: yazanın o hedefe karşı tutumu

- `olumlu`: övgü, beğeni, gurur, teşekkür, destek.
- `olumsuz`: şikâyet, eleştiri, öfke, alay, hayal kırıklığı.
- `notr`: soru, bilgi, öneri, dua ve taziye, açık bir tutum içermeyen ya da olumlu ile olumsuzu
  dengeli biçimde taşıyan yorum.

İroni ve alayda kelimelere değil kastedilen anlama bak ("harika, yine 3 saat rötar" olumsuzdur).

## 3. Konu: yalnızca hedef `marka` ise

- `fiyat`: bilet fiyatı, ek ücretler, pahalılık, kampanya.
- `operasyon`: rötar, iptal, bagaj, check-in, aktarma, fazla bilet satışı.
- `hizmet`: kabin ekibi, yer personeli, ikram, müşteri hizmetleri, iade.
- `konfor`: koltuk, kabin, uçak tipi, filo, uçak içi eğlence.
- `guvenlik`: kaza, pilotaj, bakım, emniyet.
- `yonetim`: strateji, yönetim, ticari kararlar, sponsorluk, isim değişikliği, çalışan hakları.
- `genel`: belirli bir konu söylemeden markayı öven ya da yeren yorum ("en iyi havayolu", "gururumuz").

Birden fazla konu varsa yorumda en çok yer tutanı seç. Hedef `marka` değilse konu `yok`.

## Çıktı biçimi

Her yorum için tek satır, sekmeyle ayrılmış dört alan:

```
numara<TAB>duygu<TAB>hedef<TAB>konu
```
