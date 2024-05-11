from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
tfidf_matrix = load('tfidf_matrix.joblib')
df = pd.concat([pd.read_csv(f"cleaned_file-{i}.csv", index_col="index") for i in range(1, 4)])  # Load all DataFrames

@app.route('/', methods=['POST'])  # Single route
def recommend():
    try:
        data = request.get_json()

        if 'ingredients' in data:
            # Ingredient-based recommendation
            user_ingredients = data['ingredients'].lower().split(',')
            essentials = [ess.lower() for ess in data.get("essentials", [])]

            filtered_df = df[df['ingredients'].apply(lambda x: all(ess in x for ess in essentials))]

            if not essentials:
                tfidf_vectorizer_to_use = tfidf_vectorizer
                tfidf_matrix_to_use = tfidf_matrix
            else:
                tfidf_vectorizer_to_use = TfidfVectorizer(stop_words='english')
                tfidf_matrix_to_use = tfidf_vectorizer_to_use.fit_transform(filtered_df['ingredients'])

            user_tfidf = tfidf_vectorizer_to_use.transform([','.join(user_ingredients)])
            similarities = cosine_similarity(user_tfidf, tfidf_matrix_to_use)
            top_indices = similarities[0].argsort()[-8:][::-1]

            recommendations = filtered_df.iloc[top_indices].to_json(orient='records')
            return recommendations

        elif 'recipe_name' in data:
            # Recipe name-based recommendation
            recipe_name_query = data['recipe_name'].lower()
            matching_recipes = df[df['name'].str.lower().str.contains(recipe_name_query)]

            if not matching_recipes.empty:
                return matching_recipes.iloc[0].to_json() 
            else:
                return jsonify({'error': 'No matching recipes found'}), 404

        else:
            return jsonify({'error': 'Invalid request data'}), 400  # Bad Request

    except Exception as e:
        app.logger.error(f"Error recommending dishes/recipes: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
