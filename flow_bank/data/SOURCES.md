# مصادر البيانات

كل الملفات في هذا المجلد مأخوذة من UCI Machine Learning Repository دون أي تعديل على القيم،
باستثناء تحويل الصيغة إلى CSV وإضافة رقم تعريف للشركات. التحويل يتم بالملف `prepare_data.py`،
وبصمة الملفات الأصلية (SHA-256) محفوظة في `manifest.json`.

## credit_card_clients.csv
- Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C55S3H
- الرخصة: Creative Commons Attribution 4.0 (CC BY 4.0)
- 30 000 عميل بطاقة ائتمان في تايوان، أبريل–سبتمبر 2005، المبالغ بالدولار التايواني.
- الهدف `DEFAULT`: تعثّر في سداد الشهر التالي (1 = نعم).

## taiwan_companies.csv
- *Taiwanese Bankruptcy Prediction* [Dataset] (2020). UCI Machine Learning Repository.
  https://doi.org/10.24432/C5004D — https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
- الرخصة: CC BY 4.0. المصدر الأصلي: Taiwan Economic Journal، الفترة 1999–2009.
- 6 819 شركة، 95 مؤشرًا ماليًا، الهدف `BANKRUPT` (1 = أفلست حسب قواعد بورصة تايوان)، 220 حالة إفلاس.
- حذفنا المسافة في أول أسماء الأعمدة فقط. العمود `Net Income Flag` ثابت (1 دائمًا) ولا يستعمله النموذج.
- `COMPANY_ID` رقم أضفناه بترتيب الأسطر في الملف الأصلي.
