from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# simple fake news logic (demo + earning purpose)
fake_keywords = ["fake", "rumor", "hoax", "false", "scam"]

@app.route("/")
def home():
    return {"status": "Rubi Fake News Detector API is LIVE"}

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "").lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = "FAKE" if any(word in text for word in fake_keywords) else "REAL"

    return jsonify({
        "result": result,
        "confidence": 0.85
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

