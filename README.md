# Gradient-Descent
Gradyan iniş, bir fonksiyonun en düşük değerlerini nerede çıkaracağını sayısal olarak tahmin eden bir algoritmadır.Bir maliyet fonksiyonunu en aza indirmek amacıyla bir modelin parametrelerini güncellemek için makine öğrenimi projelerinde en çok kullanılan optimizasyon tekniklerinden biridir.
Gradyan inişinde gradyan, fonksiyonun belirli bir noktada en dik artış yönünü gösteren bir vektördür.Gradyanın ters yönünde hareket etmek, algoritmanın yavaş yavaş fonksiyonun daha düşük değerlerine doğru inmesine ve sonunda fonksiyonun minimumuna ulaşmasına olanak tanır.

Kodu adım adım inceleyelim: 
### 1)Veri Seti :
   
   Bu algoritma için Boston' daki ev fiyatlarını çeşitli özelliklerine göre (kişi başına düşen suç oranı, konut başına düşen ortalama oda sayısı vs.) inceleyen bir veri seti kullanıldı.
   Veri setindeki konut fiyatlarının aralığını ve dağılımını anlamaya yardımcı olması için histogramları kullanarak Boston'daki farklı mahallelerdeki ortalama konut fiyatlarının dağılımı görselleştirildi.
   ![indir (3)](https://github.com/BilgeGoksel/Gradient-Descent/assets/163318769/5cc0732b-b9bb-4653-a672-f9b9850eca89)

   -Sağa çarpık bir dağılım, düşük ila orta fiyatlı mülklerin sayısına kıyasla çok yüksek fiyatlara sahip mülklerin sayısının daha az olduğunu gösterir.
   
   -Maksimum 50.000 (dolar cinsinden) gibi uç değerler potansiyel aykırı değerleri temsil edebilir.

### 2) Veriyi Ayırma (Split data):
   Veriyi eğitim ve test için random şekilde ayırıyoruz. Burada hedef değişkenimiz MEDV (ev fiaytlarının medyanı) kolonu olacak.
### 3) Parametreleri Tanımlama :
   Ağırlıkları (weights) ve bias değerini random tanımlıyoruz.

### 4) Fonksiyonlar:
- `compute_cost` fonksiyonunda gerçek ve tahmin edilen hedef değer arasındaki fark olan hatayı hesaplanır ardından cost değeri döndürülür.
  ![Ekran görüntüsü 2024-05-04 180356](https://github.com/BilgeGoksel/Gradient-Descent/assets/163318769/b70f85c1-152e-4d30-b827-588bd6028299)
  
- `compute_gradients` fonksiyonunda predict fonksiyonu kullanılarak,bir dögü içerisinde girdi ve mevcut ağırlıklar ile tahmin edilen çıktı elde edilir.Hata hesaplanır. 'dw' ve 'db' değerleri güncellenir. Hata ve girdinin çaarpımı gradyanlara eklenir.Döngü bittiğinde, dw ve db değerleri tüm örnekler için hesaplanan gradyanların ortalamasını almak için örnek sayısına (m) bölünür.
- `gradient_descent` her iterasyonda maliyet değerini tutmak için  'cost' listesi oluşturur. Belirlenen iterasyon sayısı kadar ( 100 tane seçildi.) compute_gradients fonksiyonu çağrılarak, mevcut ağırlıklar ve bias için gradyanlar hesaplanır. Bunlar learning_rate ile gradyanlar çarpılarak güncellenir. Güncellenmiş ağırlıklar ve bias ile eğitim verileri üzerindeki maliyet hesaplanır (compute_cost fonksiyonu kullanılarak).Her iterasyonun numarası ve hesaplanan maliyet değeri ekrana yazdırılır.
  
    
