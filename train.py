import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================

DATASET_ROOT = "/kaggle/input/datasets/batoulnaji/ddddddd/final_processed"
IMG_SIZE = (380, 380)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# =========================
# FIND PATHS (SAFE)
# =========================

def find_paths(root):
    train_path = None
    val_path = None

    for r, d, _ in os.walk(root):
        for folder in d:
            if folder.lower() == "train":
                train_path = os.path.join(r, folder)
            if folder.lower() in ["valid", "val", "test"]:
                val_path = os.path.join(r, folder)

    if train_path is None or val_path is None:
        raise ValueError("Train/Val paths not found")

    return train_path, val_path


train_path, val_path = find_paths(DATASET_ROOT)

# =========================
# DATA LOADING
# =========================

train_ds_raw = keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds_raw = keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)

# normalize
preprocess = tf.keras.applications.efficientnet.preprocess_input

train_ds = train_ds_raw.map(lambda x,y: (preprocess(x), y)).prefetch(AUTOTUNE)
val_ds = val_ds_raw.map(lambda x,y: (preprocess(x), y)).prefetch(AUTOTUNE)

# =========================
# CLASS WEIGHTS (IMBALANCE FIX)
# =========================

y_train = np.concatenate([y.numpy() for _, y in train_ds_raw])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

# =========================
# MODEL BACKBONE
# =========================

base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    weights="imagenet"
)
base.trainable = False

# =========================
# ARCFACE LAYER (must match backend/main.py for inference)
# =========================


@keras.saving.register_keras_serializable(package="skin", name="ArcFace")
class ArcFace(layers.Layer):
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
        c.update(
            {
                "n_classes": self.n_classes,
                "margin": self.margin,
                "scale": self.scale,
            }
        )
        return c

# =========================
# MODEL
# =========================

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

arcface = ArcFace(num_classes)

outputs = arcface(x)

model = keras.Model(inputs, outputs)

# =========================
# FOCAL LOSS + LABEL SMOOTHING
# =========================

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def focal_loss(y_true, y_pred):
    ce = loss_fn(y_true, y_pred)
    pt = tf.exp(-ce)
    return tf.reduce_mean((1 - pt) ** 2 * ce)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=focal_loss,
    metrics=["accuracy"]
)

# =========================
# TRAIN STAGE 1
# =========================

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

# =========================
# FINE TUNE
# =========================

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=focal_loss,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights
)

# =========================
# TTA PREDICTION
# =========================

def predict_tta(model, ds):
    preds, labels = [], []

    for x, y in ds:
        p1 = model.predict(x, verbose=0)
        p2 = model.predict(tf.image.flip_left_right(x), verbose=0)

        p = (p1 + p2) / 2

        preds.extend(np.argmax(p, axis=1))
        labels.extend(y.numpy())

    return np.array(labels), np.array(preds)

y_true, y_pred = predict_tta(model, val_ds)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print(cm)

# =========================
# THRESHOLD TUNING (per class)
# =========================

probs = model.predict(val_ds)
y_true = np.concatenate([y.numpy() for _, y in val_ds])

thresholds = []

for c in range(num_classes):
    best_t, best_f1 = 0.5, 0

    for t in np.arange(0.3, 0.8, 0.05):
        preds = (probs[:, c] > t).astype(int)
        f1 = np.mean(preds == (y_true == c))

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    thresholds.append(best_t)

print("Optimal thresholds:", thresholds)

# =========================
# SAVE FOR API / UI
# =========================

OUT_DIR = os.environ.get("MODEL_OUTPUT_DIR")
if not OUT_DIR:
    _root = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(_root, "backend")
os.makedirs(OUT_DIR, exist_ok=True)
_model_path = os.path.join(OUT_DIR, "best_model.keras")
_names_path = os.path.join(OUT_DIR, "class_names.json")
model.save(_model_path)
with open(_names_path, "w", encoding="utf-8") as f:
    json.dump(list(class_names), f, ensure_ascii=False, indent=2)
print("Saved model for API:", _model_path)
print("Saved class order (index = label id):", _names_path)