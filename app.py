from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
tfidf_matrix = load('tfidf_matrix.joblib')
df = pd.concat([pd.read_csv(f"cleaned_file-{i}.csv", index_col="index") for i in range(1, 4)]) 

@app.route('/recommend_by_name', methods=['POST'])
def recommend_by_name():
    try:
        data = request.get_json()

        if 'recipe_name' not in data:
            return jsonify({'error': 'Missing recipe_name in request'}), 400

        recipe_name_query = data['recipe_name'].lower()

        # Filter by name (case-insensitive partial match)
        matching_recipes = df[df['name'].str.lower().str.contains(recipe_name_query)]

        if not matching_recipes.empty:
            # Return top 10 matches (or fewer if less than 10)
            top_matches = matching_recipes.head(8)
            return top_matches.to_json(orient='records')

        else:
            return jsonify({'error': 'No matching recipes found'}), 404

    except pd.errors.EmptyDataError:
        app.logger.error("No recipe data available")
        return jsonify({'error': 'No recipe data found'}), 500

    except Exception as e:
        app.logger.error(f"Error recommending recipes: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=False)
