import json
import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import preprocess_input

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@keras.saving.register_keras_serializable(package="skin", name="ArcFace")
class ArcFace(keras.layers.Layer):
    def __init__(self, n_classes, margin=0.3, scale=30, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = int(n_classes)
        self.margin = float(margin)
        self.scale = float(scale)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(d, self.n_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, x, labels=None, training=False):
        xn = tf.nn.l2_normalize(x, axis=1)
        wn = tf.nn.l2_normalize(self.W, axis=0)
        logits = tf.matmul(xn, wn)
        if labels is None:
            return logits * self.scale
        theta = tf.acos(tf.clip_by_value(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = tf.cos(theta + self.margin)
        one_hot = tf.one_hot(labels, self.n_classes)
        output = logits * (1 - one_hot) + target_logits * one_hot
        return output * self.scale

    def get_config(self):
        c = super().get_config()
        c.update({"n_classes": self.n_classes, "margin": self.margin, "scale": self.scale})
        return c


FOLDER_TO_LABELS = [
    (("dry", "جاف", "dry_skin"), ("بشرة جافة", "Dry Skin")),
    (("oily", "دهن", "oily_skin"), ("بشرة دهنية", "Oily Skin")),
    (("combination", "مختلط", "combo", "mixed"), ("بشرة مختلطة", "Combination Skin")),
    (("normal", "عادية", "neutral"), ("بشرة عادية", "Normal Skin")),
]


def folder_to_display(folder_name: str):
    low = folder_name.lower().replace("_", " ")
    for keys, pair in FOLDER_TO_LABELS:
        if any(k in low for k in keys):
            return pair
    return folder_name, folder_name


app = FastAPI(title="Skin Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BACKEND_DIR, "model", "skin_model.keras"))
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", os.path.join(BACKEND_DIR, "class_names.json"))

model = None
folder_class_names: list[str] = []


def load_class_names():
    global folder_class_names
    if os.path.isfile(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            folder_class_names = json.load(f)
        if not isinstance(folder_class_names, list):
            folder_class_names = []
    else:
        folder_class_names = []


def try_load_model():
    global model
    model = None
    load_class_names()
    if not os.path.isfile(MODEL_PATH):
        print("WARNING: Model file not found at", MODEL_PATH)
        return
    print("Loading model from", MODEL_PATH)
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"ArcFace": ArcFace},
            compile=False,
        )
        print("Model loaded. Input shape:", model.input_shape)
    except Exception as e:
        print(f"Model loading error: {e}")
        model = None


try_load_model()


def get_mobilenet_input_size():
    """
    ابحث عن طبقة MobileNetV2 داخل الموديل واقرأ منها الحجم الصحيح.
    لو ما لقيناها، استخدم 224×224 كـ fallback.
    """
    if model is None:
        return (224, 224)
    for layer in model.layers:
        name = layer.name.lower()
        if "mobilenet" in name:
            s = layer.input_shape
            if isinstance(s, list):
                s = s[0]
            if s and len(s) == 4 and s[1] and s[2]:
                print(f"MobileNet input size detected: {s[1]}x{s[2]}")
                return (int(s[1]), int(s[2]))
    # fallback
    return (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def num_classes_from_model():
    if model is None:
        return 0
    out = model.output_shape
    if out and out[-1] is not None:
        return int(out[-1])
    for ly in model.layers:
        if isinstance(ly, ArcFace):
            return int(ly.n_classes)
    return 0


def resolve_labels():
    if model is None:
        return [], []
    n = num_classes_from_model()
    if n <= 0:
        return [], []
    if len(folder_class_names) >= n:
        names = folder_class_names[:n]
    else:
        names = [f"class_{i}" for i in range(n)]
    ar, en = [], []
    for name in names:
        a, e = folder_to_display(str(name))
        ar.append(a)
        en.append(e)
    return ar, en


@app.get("/")
def root():
    return {"status": "API running 🚀"}


@app.get("/health")
def health():
    ar, en = resolve_labels()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(ar),
        "model_path": MODEL_PATH if os.path.isfile(MODEL_PATH) else None,
        "input_size": get_mobilenet_input_size(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        image_bytes = await file.read()
        img_batch = preprocess_image(image_bytes)

        logits = model.predict(img_batch, verbose=0)
        logits = np.asarray(logits)
        if logits.ndim == 1:
            logits = np.expand_dims(logits, 0)
        probs = tf.nn.softmax(logits[0]).numpy()

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        class_ar, class_en = resolve_labels()
        n = len(probs)
        if len(class_ar) < n:
            class_ar = [f"class_{i}" for i in range(n)]
            class_en = class_ar.copy()

        results = [
            {"class": class_ar[i], "class_en": class_en[i], "confidence": float(probs[i])}
            for i in range(n)
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return JSONResponse({
            "class": class_ar[idx],
            "class_en": class_en[idx],
            "confidence": confidence,
            "all_predictions": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))