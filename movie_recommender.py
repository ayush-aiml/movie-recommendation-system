import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Sample Movie Metadata Dataset
movies_data = {
    'title': [
        'Inception', 'Interstellar', 'The Dark Knight', 'The Prestige',
        '3 Idiots', 'Taare Zameen Par', 'Dangal', 'Chhichhore',
        'Avengers: Endgame', 'Iron Man', 'Doctor Strange', 'Spider-Man: No Way Home',
        'The Matrix', 'Tenet', 'Shutter Island', 'Uri: The Surgical Strike'
    ],
    'tags': [
        'sci-fi thriller dream heist mind-bending christopher nolan leonardo dicaprio',
        'sci-fi space time-travel adventure drama christopher nolan matthew mcconaughey',
        'action crime superhero thriller gotham batman joker christopher nolan',
        'mystery thriller magic rivalry illusion drama christopher nolan hugh jackman',
        'comedy drama college friendship engineering inspiration bollywood aamir khan',
        'drama emotional school education dyslexia inspiration bollywood aamir khan',
        'sports biography wrestling family motivation bollywood aamir khan',
        'comedy drama college friendship hostel life inspiration bollywood sushant singh',
        'action sci-fi superhero marvel team-up time-travel tony stark',
        'action sci-fi superhero marvel technology suit genius tony stark',
        'action fantasy sci-fi superhero marvel magic multiverse',
        'action sci-fi superhero marvel multiverse spiderman',
        'sci-fi action cyberpunk simulation artificial intelligence hacker',
        'sci-fi action thriller time-inversion espionage christopher nolan',
        'mystery psychological thriller detective island twist leonardo dicaprio',
        'action military war patriotism army special-forces bollywood vicky kaushal'
    ]
}

df = pd.DataFrame(movies_data)

# 2. Convert Movie Tags into Numerical Vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

# 3. Compute Cosine Similarity Matrix Across All Movies
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function
def recommend_movies(movie_title, top_n=5):
    if movie_title not in df['title'].values:
        return f"Movie '{movie_title}' not found in database."

    # Get index of the selected movie
    idx = df[df['title'] == movie_title].index[0]

    # Pair similarity scores with movie indices and sort descending
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Skip index 0 (the movie itself) and fetch top N matches
    recommendations = []
    for i, score in sorted_movies[1:top_n + 1]:
        recommendations.append((df.iloc[i]['title'], round(score * 100, 1)))

    return recommendations

# 5. Test Recommendations
test_movie = 'Inception'
print(f"Top 5 Recommendations for '{test_movie}':\n")
for rank, (title, match_score) in enumerate(recommend_movies(test_movie), start=1):
    print(f"{rank}. {title} (Similarity: {match_score}%)")
