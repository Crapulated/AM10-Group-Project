#%%
import pandas as pd

df = pd.read_csv('movie_metadata.csv')

# Drop rows without year or country
df = df.dropna(subset=['title_year', 'country', 'genres'])

df['title_year'] = df['title_year'].astype(int)

# ---- Multi-genre explode ----
df['genre_list'] = df['genres'].str.split('|')
df_exploded = df.explode('genre_list')

# Export to CSV for Tableau
df_exploded[['movie_title', 'country', 'title_year', 'genre_list','imdb_score']].to_csv(
    'movies_ratings_by_country.csv',
    index=False
)