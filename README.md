Bu proje, çok katmanlı (Multi-layer) sinir ağı kullanarak karmaşık spiral desenlerinin sınıflandırılmasını göstermektedir. 4 farklı sınıfa ait iç içe geçmiş spiral veriler üzerinde çalışan tam bağlantılı (fully connected) sinir ağı implementasyonu içerir.
Kod Ne Yapar?
1. Veri Üretimi

Spiral Veri Seti: 4 sınıfa ait iç içe geçmiş spiral desenler oluşturur
Sınıf Başına 100 Örnek: Toplam 400 veri noktası
Gürültülü Veriler: Gerçek dünya verilerini simüle etmek için rastgele gürültü ekler
One-Hot Encoding: Sınıf etiketlerini sinir ağına uygun formata dönüştürür

2. Sinir Ağı Mimarisi
Girdi Katmanı (2 nöron) → Gizli Katman (32 nöron) → Çıkış Katmanı (4 nöron)

Girdi Katmanı: 2D koordinatlar (x, y)
Gizli Katman: 32 nöron + Tanh aktivasyon fonksiyonu
Çıkış Katmanı: 4 nöron + Softmax aktivasyon fonksiyonu

3. Eğitim Süreci

İleri Yayılım (Forward Propagation):

Girdiler gizli katmana yayılır
Tanh aktivasyonu uygulanır
Çıkış katmanına geçer
Softmax ile olasılık dağılımı elde edilir


Kayıp Hesaplama:

Categorical Cross-Entropy kaybı kullanılır
Sayısal kararlılık için epsilon değeri eklenir


Geri Yayılım (Backpropagation):

Gradyanlar hesaplanır
Ağırlıklar ve bias değerleri güncellenir
Gradyan normalizasyonu uygulanır



4. Görselleştirme Özellikleri

Gerçek Zamanlı Eğitim: Her 500 epoch'ta grafik güncellenir
Karar Sınırları: Modelin öğrendiği sınıf bölgeleri görselleştirilir
Doğruluk Takibi: Eğitim sırasında doğruluk oranı izlenir
Kayıp Grafiği: Eğitim ilerlemesi görsel olarak takip edilir

5. Kullanılan Teknikler

Tanh Aktivasyonu: Gizli katmanda (-1, 1) aralığında değerler
Softmax Aktivasyonu: Çıkış katmanında olasılık dağılımı
Xavier/Glorot Başlatma: Küçük rastgele ağırlıklarla başlatma
Gradyan Normalizasyonu: Öğrenme stabilitesi için

Matematiksel Detaylar
İleri Yayılım
z1 = X · W1 + b1
a1 = tanh(z1)
z2 = a1 · W2 + b2
y_pred = softmax(z2)
Geri Yayılım
δ2 = y_pred - y_true
δ1 = (δ2 · W2^T) ⊙ (1 - a1²)
Ağırlık Güncellemesi
W2 = W2 - lr · (a1^T · δ2) / N
W1 = W1 - lr · (X^T · δ1) / N
Hiperparametreler

Öğrenme Oranı: 0.1
Epoch Sayısı: 30,000
Gizli Katman Boyutu: 32 nöron
Batch Boyutu: Tam veri seti (400 örnek)

Beklenen Sonuçlar
Eğitim tamamlandığında model:

Yüksek Doğruluk: %95+ doğruluk oranı
Net Karar Sınırları: 4 spiral sınıfı arasında temiz ayrım
Düşük Kayıp: Categorical Cross-Entropy kaybı minimum seviyede
Stabil Öğrenme: Kayıp fonksiyonu düzenli azalış gösterir

Öğrenme Çıktıları
Bu implementasyon ile öğrenilen konular:

Çok katmanlı sinir ağlarının çalışma prensibi
Geri yayılım algoritmasının matematiği
Aktivasyon fonksiyonlarının etkisi
Kayıp fonksiyonları ve optimizasyon
Karar sınırlarının görselleştirilmesi
Doğrusal olmayan problemlerin çözümü

Teknik Özellikler

Framework: Pure NumPy (TensorFlow/PyTorch kullanılmaz)
Görselleştirme: Matplotlib ile gerçek zamanlı grafikler
Veri Tipi: Sentetik spiral veri seti
Problem Tipi: Çok sınıflı sınıflandırma (Multi-class Classification)
Optimizasyon: Gradyan İnişi (Gradient Descent)
