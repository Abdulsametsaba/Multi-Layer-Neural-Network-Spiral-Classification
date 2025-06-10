import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def sigmoid(z):
    # Klip z'yi sayısal kararlılık için sınırlar
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    # Sayısal kararlılık için max değerini çıkar
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 1. Yeni Zorlayıcı Veri Seti (4 Sınıf - Spiraller)
def generate_spiral_data(points, classes):
    """İç içe geçmiş spiraller oluşturan fonksiyon."""
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) # Yarıçap
        # Theta: Başlangıç açısını sınıfa göre kaydırarak spiralleri iç içe geçir
        # Gürültü ekleyerek daha gerçekçi hale getir
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.3
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y

np.random.seed(1) # Tekrarlanabilirlik için
n_samples_per_class = 100 # Sınıf başına örnek sayısı
num_classes = 4
X, y_labels = generate_spiral_data(n_samples_per_class, num_classes)

# Etiketleri one-hot encoding formatına dönüştür
y = np.eye(num_classes)[y_labels]

# 2. Ağ Mimarisi Güncellemesi (Daha karmaşık veri için gizli katmanı büyütebiliriz)
input_size = 2
hidden_size = 32 # Gizli katman nöron sayısını artırıyoruz
output_size = num_classes # Çıkış boyutu 4

# Ağırlıkları ve biasları daha küçük değerlerle başlatmak genellikle daha iyidir
w1 = np.random.randn(input_size, hidden_size) * 0.01 # Daha küçük başlatma
b1 = np.zeros((1, hidden_size))

w2 = np.random.randn(hidden_size, output_size) * 0.01 # Daha küçük başlatma
b2 = np.zeros((1, output_size))

# 3. Hiperparametreler
lr = 0.1 # Öğrenme oranını biraz artırabiliriz veya dinamik hale getirebiliriz
epochs = 30000 # Epoch sayısını artırıyoruz
plot_update_frequency = 500 # Grafik güncelleme sıklığı

# 4. Eğitim ve Görselleştirme
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple']
markers = ['o', 'x', 's', '^']
labels = ['Sınıf 0', 'Sınıf 1', 'Sınıf 2', 'Sınıf 3']
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA', '#DDAAFF']) # Karar bölgeleri için renk haritası

losses = [] # Kayıp değerlerini takip etmek için

for epoch in range(epochs):
    # İleri Yayılım
    z1 = np.dot(X, w1) + b1
    a1 = np.tanh(z1) # Aktivasyon fonksiyonunu tanh olarak değiştirebiliriz (sigmoid yerine)
                     # Tanh genellikle gizli katmanlarda daha iyi performans gösterebilir

    z2 = np.dot(a1, w2) + b2
    y_pred = softmax(z2) # Çıkış katmanı aktivasyonu: Softmax

    # Kayıp Hesaplama (Categorical Cross-Entropy)
    # Küçük bir epsilon ekleyerek log(0) hatasını önleyelim
    epsilon = 1e-9
    loss = -np.sum(y * np.log(y_pred + epsilon)) / len(X)
    losses.append(loss)

    # Geri Yayılım
    delta2 = y_pred - y # Softmax ve Categorical Cross-Entropy için türev
    dw2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    # Tanh türevi: 1 - tanh(z)^2 = 1 - a1^2
    delta1 = np.dot(delta2, w2.T) * (1 - np.power(a1, 2))
    dw1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # Ağırlık Güncelleme (Gradyan normalizasyonu ile)
    w2 -= lr * dw2 / len(X)
    b2 -= lr * db2 / len(X)
    w1 -= lr * dw1 / len(X)
    b1 -= lr * db1 / len(X)

    # Grafik Güncelleme ve Kayıp Yazdırma
    if epoch % plot_update_frequency == 0 or epoch == epochs - 1:
        # Doğruluğu hesapla
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_labels)

        ax.cla()

        # Veri noktalarını çiz
        for i, label_index in enumerate(np.unique(y_labels)):
            class_indices = np.where(y_labels == label_index)[0]
            ax.scatter(X[class_indices, 0], X[class_indices, 1], color=colors[i], marker=markers[i], s=50, label=labels[i], edgecolors='k', alpha=0.8)

        # Karar Sınırlarını Çiz
        h = .02 # Meshgrid adımı
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5 # Sınırları spiral verisine göre ayarla
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5 # Sınırları spiral verisine göre ayarla
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Grid noktaları üzerinde tahmin yap
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        z1_grid = np.dot(grid_points, w1) + b1
        a1_grid = np.tanh(z1_grid) # Grid için de tanh kullan
        z2_grid = np.dot(a1_grid, w2) + b2
        pred_prob_grid = softmax(z2_grid)
        pred_class_grid = np.argmax(pred_prob_grid, axis=1) # Tahmin edilen sınıf indeksi
        pred_class_grid = pred_class_grid.reshape(xx.shape)

        # Karar bölgelerini çiz
        ax.contourf(xx, yy, pred_class_grid, cmap=cmap_light, alpha=0.5)

        ax.set_xlabel('Girdi 1 (x1)')
        ax.set_ylabel('Girdi 2 (x2)')
        ax.set_title(f'Spiral Veri Eğitimi - Epoch: {epoch} - Kayıp: {loss:.4f} - Doğruluk: {accuracy:.4f}')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        plt.draw()
        plt.pause(0.01)

print("Eğitim Tamamlandı.")

# Kayıp grafiğini çiz
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Eğitim Sırasındaki Kayıp Değeri")
plt.xlabel("Epoch")
plt.ylabel("Categorical Cross-Entropy Kaybı")
plt.grid(True)


# Son Tahminler ve Doğruluk
z1 = np.dot(X, w1) + b1
a1 = np.tanh(z1) # Son tahmin için de tanh kullan
z2 = np.dot(a1, w2) + b2
final_probabilities = softmax(z2)
final_predictions = np.argmax(final_probabilities, axis=1)

accuracy = np.mean(final_predictions == y_labels)
print(f"\nEğitim Sonrası Doğruluk: {accuracy:.4f}")

plt.ioff() # İnteraktif modu kapat
plt.show() # Tüm grafikleri göster