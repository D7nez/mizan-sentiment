# ربط المشروع بمتطلبات AI Engineer — Taibah Valley

هذا الربط دليل لما ينفّذه المشروع فعلًا، وليس إعلان استيفاء جميع شروط الوظيفة.

| متطلب | الدليل في المشروع | الحد |
|---|---|---|
| Python | وحدات preprocessing/predictor/data/train/api | يحتاج المرشح شرحها وتعديلها بنفسه |
| TensorFlow/Keras | BiLSTM + SelfAttention، تحميل حقيقي واختبار inference | لا توجد دقة جديدة على بيانات حقيقية |
| SQL | SQLite audit + feedback، معاملات واستعلامات parameterized | ليس Postgres موزعًا أو تصميم warehouse |
| REST APIs | predict/CSV/feedback/monitoring وOpenAPI | خدمة محلية بمفتاح مشترك اختياري |
| تقييم وتحسين موثوقية المخرجات | split بلا تداخل نصوص منظفة، macro F1، confusion matrix، log loss، OOV/truncation/review | منع exact duplicates فقط؛ near-duplicates وgroup/time تحتاج بيانات مناسبة |
| ML workflows / MLOps | تدريب قابل لإعادة التشغيل، manifest/hashes، MLflow مقارنة baseline والشبكة | لا automatic promotion/retraining أو data drift إحصائي |
| Docker | API CPU، volume SQL، training tools profile | لم يُنفذ Docker هنا |
| توثيق واختبارات وGit | README/Model card/CI و10 اختبارات ناجحة محليًا | نجاح GitHub Actions يحتاج تحقق بعد الرفع |
| LLMs / Generative AI | لا ينفّذها هذا المشروع | اعرض مشروع الأخبار والصياغة |
| LangChain / LangGraph | غير مستخدمين في هذا المصنّف | مثبتان في مشروع الأخبار |
| RAG / embeddings / vector databases | ليست هنا | اعرض RAG منفصلًا يعمل فعليًا؛ embedding layer داخل classifier ليست RAG |
| Hugging Face | غير مضاف | لا تضعه ضمن stack لهذا المشروع |
| Cloud | توجد حاوية قابلة للنشر | لا نشر سحابي منفّذ |
| سنة خبرة وبكالوريوس | متطلبات شخصية | المشروع لا يغيّر مدة الخبرة أو المؤهل |

## ما تستطيع قوله بالمقابلة

“I built an Arabic sentiment classification service using a BiLSTM with attention in TensorFlow. I separated preprocessing, inference, API and persistence, added batch and CSV prediction, and exposed model scores with review flags. While reviewing the original training notebook, I found oversampling and vocabulary fitting before the split, so I stopped treating the historical accuracy as a reliable held-out result. I added a split-first training pipeline and an MLflow comparison against a TF-IDF baseline. The original dataset is still needed for a credible new benchmark.”

اشرح مشكلة التسرب وسبب استخدام macro F1، الفرق بين low score وneutral، ولماذا لا تعتبر softmax أو attention تفسيرًا يقينيًا. هذا أدق وأقوى من عرض رقم كبير غير موثوق.
