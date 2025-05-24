from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.get_json()
    text1, text2 = data["text1"], data["text2"]
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return jsonify({"similarity score": round(score.item(), 2)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
