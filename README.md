# Content-Based Movie Recommendation System

A Machine Learning recommendation engine built with Python, Pandas, and Scikit-learn that suggests similar movies based on genre, plot keywords, director, and cast metadata using TF-IDF vectorization and Cosine Similarity.

## Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)

## Key Features
- Feature extraction from movie metadata tags using TF-IDF
- Pairwise similarity scoring using Cosine Similarity matrices
- Ranked Top-N recommendation engine with percentage similarity scores

## How to Run
1. Clone this repository or download `movie_recommender.py`.
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn
