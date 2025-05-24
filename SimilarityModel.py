from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/similarity", methods=["POST"])
def similarity():

    #Assign the vectorizer
    vectorizer = TfidfVectorizer()

    # Get the JSON data from the request
    data = request.get_json()

    # Validate the input data
    if not data or "text1" not in data or "text2" not in data:
        return jsonify({"error": "Invalid input data"}), 400
    if not isinstance(data["text1"], str) or not isinstance(data["text2"], str):
        return jsonify({"error": "Both text1 and text2 must be strings"}), 400
    if not data["text1"] or not data["text2"]:
        return jsonify({"error": "Both text1 and text2 must be non-empty"}), 400
    
    # Data comes in the form of {"text1": "text", "text2": "text"}
    text1, text2 = data["text1"], data["text2"]

    #Fit the vectorizer and calculate cosine similarity
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity between the two texts
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Return the similarity score as a JSON response
    return jsonify({"similarity score": round(score, 2)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
