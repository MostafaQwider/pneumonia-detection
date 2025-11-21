# استيراد مكتبة numpy للتعامل مع المصفوفات
import numpy as np

# استيراد Flask لإنشاء API وقراءة الطلبات وإرجاع النتائج بصيغة JSON
from flask import Flask, request, jsonify

# استيراد مكتبة PIL (Pillow) لقراءة الصور والتعديل عليها
from PIL import Image

# استيراد TensorFlow لتشغيل النموذج المدرب
import tensorflow as tf

# مكتبة io لقراءة البيانات الثنائية (bytes)
import io

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل النموذج المدرب المحفوظ مسبقاً بصيغة H5
model = tf.keras.models.load_model('pneumonia_model.h5')

# قائمة التصنيفات كما تم تدريب النموذج عليها
classes = ['سليم', 'مصاب بالالتهاب الرئوي']  # Normal = 0, Pneumonia = 1

# تحديد حجم الصورة الذي يتوقعه النموذج (حسب التدريب)
IMG_SIZE = (224, 224)

# دالة لتحويل الصورة إلى تنسيق مقبول من قبل النموذج
def preprocess_image(image_bytes):
    # فتح الصورة من البايتات وتحويلها إلى RGB (3 قنوات)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # إعادة تحجيم الصورة للحجم المطلوب
    img = img.resize(IMG_SIZE)

    # تحويل الصورة إلى مصفوفة NumPy بقيم بين 0 و 1 (تطبيع)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # إضافة بعد إضافي ليصبح شكل الصورة (1, 224, 224, 3) وهو ما يحتاجه النموذج
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# نقطة نهاية بسيطة لفحص أن السيرفر يعمل
@app.route('/')
def home():
    return "🩺 Pneumonia Detection API is running!"

# نقطة النهاية الرئيسية للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التحقق من وجود الصورة في الطلب
        if 'image' not in request.files:
            return jsonify({'error': 'No image found in request'}), 400

        # قراءة محتوى الصورة بصيغة bytes
        image_file = request.files['image'].read()

        # معالجة الصورة لتصبح جاهزة للتنبؤ
        processed_image = preprocess_image(image_file)

        # تنفيذ التنبؤ باستخدام النموذج، وإرجاع القيمة بين 0 و 1
        prediction = model.predict(processed_image)[0][0]

        # تصنيف النتيجة: إذا > 0.5 فالمريض مصاب، غير ذلك طبيعي
        predicted_class = 1 if prediction > 0.5 else 0

        # حساب الثقة: إذا مصاب نستخدم القيمة كما هي، وإذا طبيعي نطرح من 1
        confidence = prediction * 100 if predicted_class == 1 else (1 - prediction) * 100

        # بناء نتيجة JSON لإرجاعها
        result = {
            'label': classes[predicted_class],      # اسم الفئة (Normal أو Pneumonia)
            'confidence': float(round(confidence, 2))  # الثقة بنسبة مئوية
        }

        return jsonify(result)

    # معالجة الأخطاء العامة داخل السيرفر
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

# تشغيل السيرفر على جميع الواجهات (0.0.0.0) على المنفذ 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
