from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer
model = load_model("sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 200

# Flask app
app = Flask(__name__)

def encode_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    return padded

@app.route("/")
def home():
    return "✅ Sentiment Analysis API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    if not review:
        return jsonify({"error": "No review text provided"}), 400

    encoded = encode_text(review)
    prob = model.predict(encoded)[0][0]
    label = "Positive" if prob > 0.5 else "Negative"

    return jsonify({"review": review, "sentiment": label, "probability": float(prob)})

if __name__ == "__main__":
    app.run(debug=True)

