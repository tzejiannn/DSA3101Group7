import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes in your app

# Sample data (Customer choices and product features)
data = {
    'apparel_type': ['T-shirt', 'Shirt', 'Jacket', 'T-shirt', 'T-shirt'],
    'color': ['Red', 'red', 'Green', 'Black', 'Black'],
    'design': ['Striped', 'Plain', 'Graphic', 'Striped', 'Striped']
}

customer_preference = pd.DataFrame(data)
customer_preference_encoded = pd.get_dummies(customer_preference[['apparel_type', 'color', 'design']])

# Calculate the cosine similarity matrix between items
cos_sim = cosine_similarity(customer_preference_encoded)

# Recommendation function
def recommend_similar_items(item_index, similarity_matrix, top_n=3):
    similarity_scores = similarity_matrix[item_index]
    similar_items = np.argsort(similarity_scores)[::-1][1:]  # Exclude the item itself
    top_items = similar_items[:top_n]
    return top_items

# Route for getting recommendations
@app.route('/recommendations/', methods=['POST'])
def get_recommendations():
    # Get the input data from the request (apparel_type, color, design)
    customer_data = request.get_json()

    # Convert input to DataFrame and encode it using the same method as customer_preference
    encoded_input = pd.get_dummies(pd.DataFrame([customer_data]))  # Convert to DataFrame
    encoded_input = encoded_input.reindex(columns=customer_preference_encoded.columns, fill_value=0)  # Align columns

    # Calculate similarity with the dataset
    input_similarity = cosine_similarity(encoded_input, customer_preference_encoded)

    # Get recommendations
    recommended_item_indices = recommend_similar_items(0, input_similarity, top_n=3)

    # Return recommended items as JSON response
    recommended_items = customer_preference.iloc[recommended_item_indices].reset_index(drop=True)
    
    return jsonify(recommended_items.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

