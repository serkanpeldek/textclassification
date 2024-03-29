
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def run_tfidfculster(df, data_col, target_col, num_categories, save_dir, filename):

    # Extract features from 'product_text' using TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[data_col])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_categories, random_state=42)
    df['category_id'] = kmeans.fit_predict(tfidf_matrix)

    # Print category titles and products in each category
    output_file_path = f'{save_dir}/{filename}.txt'

    # Print category titles and products in each category and save to a text file
    with open(output_file_path, 'w') as output_file:
        for category_id in range(num_categories):
            category_title = f'Category {category_id + 1}'
            category_products = df[df['category_id'] == category_id][target_col].tolist()
            text_products = df[df['category_id'] == category_id][data_col].tolist()
            output_file.write(f"\n{category_title}:\n")
            n = 20
            for i, (t, c) in enumerate(zip(text_products, category_products)):
                if i > n:
                    break
                output_file.write(f"{t} --- {c}\n")
      
