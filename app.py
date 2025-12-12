from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

@app.route("/")
def home():
    return {"status": "Rubi Fake News Detector API is LIVE!"}

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    logits = outputs.logits

    prediction = tf.math.argmax(logits, axis=1).numpy()[0]
    probs = tf.nn.softmax(logits, axis=1)
    confidence = float(tf.reduce_max(probs))

    result = "REAL" if prediction == 1 else "FAKE"

    return jsonify({
        "result": result,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
