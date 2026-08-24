# Otomatik Hedef Tanıma (ATR) Sınıflandırma Benchmark Raporu

FLIR termal kızılötesi veri seti üzerinde araç (`car`) ve yaya (`person`) hedeflerinin sınıflandırılması amacıyla eğitilen üç farklı derin öğrenme mimarisinin doğrulama başarımı, çıkarım hızı ve donanım kaynak tüketimi karşılaştırılmıştır.

## Karşılaştırma Sonuçları

| Model | Doğruluk (%) | F1-Score (%) | Gecikme (ms) | FPS | Parametre (M) | Boyut (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | %99.08 | %99.08 | 2.98 | 335.8 | 11.18M | 42.71 MB |
| **MobileNetV3-Large** | **%99.23** | **%99.23** | 7.23 | 138.3 | **3.96M** | **15.30 MB** |
| **DenseNet-121** | %99.14 | %99.14 | 23.94 | 41.8 | 6.96M | 27.12 MB |

## Mimari Analizi ve Bulgular

**Hız ve Donanım Verimliliği:**  
ResNet-18, basit ve doğrusal blok yapısı sayesinde GPU üzerinde tensör işlemlerini en optimize şekilde paralelleyerek 335.8 FPS ve 2.98 ms gecikmeyle hız testinin lideri olmuştur. Buna karşılık MobileNetV3-Large, Depthwise Separable Convolution katmanları sayesinde model boyutunu 15.30 MB'a, parametre sayısını ise 3.96M seviyesine indirerek kaynak kısıtlı donanımlar için en hafif profil sunmuştur. 138.3 FPS değeri, gerçek zamanlı video akışı (genellikle 30-60 FPS) için fazlasıyla yeterli bir pay bırakmaktadır.

**Doğruluk ve Ayrıştırma Hassasiyeti:**  
Termal silüetlerin düşük kontrastlı doğasında MobileNetV3-Large, kanal dikkat (Squeeze-and-Excitation) mekanizmasının katkısıyla %99.23 ile en yüksek doğruluk ve F1 skoruna ulaşmıştır. DenseNet-121 her ne kadar öznitelik haritalarını birleştirerek %99.14 doğruluk yakalasa da yoğun bellek trafiği sebebiyle çıkarım süresi 23.94 ms'ye çıkmış ve 41.8 FPS ile en yavaş model olmuştur.

## Nihai Karar ve Dağıtım Stratejisi

* **Uç Cihaz ve İHA Dağıtımı:** Düşük bellek ayak izi (15.3 MB), düşük güç tüketimi ve en yüksek doğruluk oranı (%99.23) nedeniyle **MobileNetV3-Large** birincil model olarak seçilmiştir.
* **Maksimum Çıkarım Hızı Gerektiren Sistemler:** Gecikmenin kritik olduğu yüksek frekanslı yer istasyonu veya sunucu tabanlı analiz hatlarında **ResNet-18** doğrudan alternatif olarak kullanılabilir.