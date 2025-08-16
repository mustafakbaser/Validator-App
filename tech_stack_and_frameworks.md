## 2. Teknoloji ve Kütüphane Seçimi

Bu bölümde, projenin gerçekleştirilmesinde kullanılacak teknolojileri, kütüphaneleri ve algoritmaları neden seçtiğimi açıklayacağım. Burada izlemeyi düşündüğüm yaklaşım sektörde kendini kanıtlamış, aktif olarak kullanılan ve ölçeklenebilir çözümler sunan teknolojileri tercih etmek olacaktır.

Ayrıca, seçilen her teknolojiyi, popüler olduğu gerekçesiyle değil, projeye değer katma, bakım kolaylığı, güvenilirlik gibi önemli faktörler üzerinden değerlendireceğim.

---

### Metin Tanıma (OCR - Optical Character Recognition)
Kimlik belgelerinden **Ad, Soyad ve TCKN**  gibi kritik bilgilerin çıkarılması için tercihim, yüksek esneklik ve doğruluk oranı sebebiyle Google Tesseract OCR olacak.

- **Avantajları:**  
  - Açık kaynak olması ve geniş topluluk desteği.  
  - Türkçeyi de içeren birden fazla dil desteği.
  - Ön işleme (preprocessing) adımları ile doğruluğun artırılması.

- **Dezavantajları:**  
  - Doğrudan ham görsel üzerinde düşük doğruluk verebilir, bu nedenle **OpenCV** tabanlı görüntü temizleme (gürültü azaltma, thresholding (eşikleme), dilate/erode (genişletme/erozyon) işlemleri) kritik rol oynar.
  - Eğitim gerektiren özel karakter setleri için ek model eğitimi gerekebilir.

Alternatif olarak, **AWS Textract** veya **Google Vision API** gibi bulut tabanlı çözümler de düşünülebilir. Bu çözümler, **yüksek doğruluk oranı** ve **daha az bakım yükü** sunsa da maliyet faktörü göz önünde bulundurulmalıdır.  

**RAG (Retrieval Augmented Generation)** ve **LLM (Large Language Models)** deneyimim sayesinde, OCR sonrası metinlerin güvenilirliğini arttırabilir ve tutarlılık kontrolünü yapabilirim. Örneğin, OCR tarafından çıkarılan “TCKN” alanının formatını **LLM destekli kurallar** ile doğrulamak, yanlış okumaların sisteme yansımasını engeller. Böylece saf OCR sonuçlarını, **LLM tabanlı metin düzeltme (post-processing)** ile desteklemek sistemi çok daha güvenilir hale getirir.

Bunlarla birlikte, OCR'ın düşük güvenli veya şüpheli okuduğu TCKN/isim alanlarını maskeli olarak RAG bağlamıyla LLM'e gönderip, şema kısıtlı (ör. JSON schema / regex-constrained) öneriler alarak sadece öneri düzeltmelerini sağlayabilirim. Burada, sadece düzeltme önerisi vermekle sınırlı kalarak, algoritmik biçimde kurallar, TCKN algoritması, regex, iş kuralları, deterministik süreçler ile karar vermek mümkün olur.

---

### İmza Karşılaştırma
İki imzanın benzerliğini ölçmek için **görüntü işleme ve özellik çıkarımı (feature extraction)** temelli bir yaklaşım tercih ederim. Burada iki yaygın yöntem öne çıkar:  

- **IFT (Ölçek Değişmez Özellik Dönüşümü) / ORB (Yönlendirilmiş Hızlı ve Döndürülmüş BRIEF) Yöntemleri:**  
  İmzaların karakteristik noktalarını çıkarıp eşleştirme yapar.  
  - Avantaj: Aydınlatma, döndürme gibi değişimlere dayanıklıdır.  
  - Dezavantaj: Hesaplama maliyeti yüksektir.  

- **Siamese Neural Network (Derin Öğrenme Yaklaşımı):**  
  İmzaları vektör uzayına indirger ve vektörler arasındaki mesafeyi hesaplayarak benzerlik oranı üretir.
  - Avantaj: Yüksek doğruluk ve farklı varyasyonlara dayanıklıdır.
  - Dezavantaj: Etkili sonuçlar için büyük miktarda etiketlenmiş imza verisine ihtiyaç duyar.

Hibrit çözümler şu anda sektörde oldukça fazla tercih ediliyor, bu sebeple benim seçimim bunlardan yana. İlk aşamada hızlı ve hafif ORB tabanlı bir karşılaştırma yapılarak sistem güvenilirliği artırılabilir. Kritik eşleşmelerde Siamese Ağı kullanılarak güvenilir sonuçlar elde edilebilir ve sistemin güvenliği arttırılabilir.

---

### API Geliştirme
Sistemin servis olarak çalışabilmesi için **RESTful API** mimarisini tercih ederim. Ancak, gelecekteki ölçeklenebilirlik ve farklı istemci ihtiyaçlarını da göz önünde bulundurarak **GraphQL** seçeneğini de değerlendirmek mantıklı olacaktır.  

- **Backend Teknolojisi:**  
  - **Python (FastAPI):** Hızlı geliştirme, async destek, otomatik dokümantasyon (Swagger/OpenAPI).  
  - Alternatif: **Node.js (NestJS)** - Benim de çok iyi seviyede olduğum, TypeScript tabanlı güçlü modüler framework'ler.

- **Veri Doğrulama Katmanı:**  
  - API katmanına gelen taleplerin (ör. TCKN, isim formatı) **Pydantic** (Python) ya da **Zod** (Node.js) gibi tip güvenliği sağlayan kütüphaneler ile doğrulanması.  

- **Kimlik Doğrulama & Güvenlik:**  
  - JWT tabanlı authentication.  
  - Rate limiting ve IP bazlı güvenlik kontrolleri.  
  - Belgelerin ve imza görsellerinin depolanmasında **şifreli (encrypted) dosya sistemi** veya **S3 + KMS** gibi çözümler.  

Docker tecrübem sayesinde, API bileşenlerini containerize ederek CI/CD sürecine entegre edebilirim. Bu, staging, test ve production ortamlari arasında tutarlılık ve taşınabilirlik saglayacaktir.

---

### Sonuç
Bu adımda seçtiğim tüm teknolojiler, çok kapsamlı araştırmalarım sonucunda bir araya getirilmiş ve sadece “çalışsın” diye değil, **endüstriyel standartlara uygun, güvenli, ölçeklenebilir ve bakımı kolay bir mimari** hedefiyle belirlenmiştir. Tesseract ve OpenCV kullanan OCR çözümü, imza karşılaştırmada hibrit metod, API geliştirmede FastAPI, Node.js veya NestJS gibi güçlü frameworkler, sistemin güvenilirliğini ve sürdürülebilirliğini garanti altına alacaktır.