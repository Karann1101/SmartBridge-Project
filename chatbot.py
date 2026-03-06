import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("quotes.csv")

# Remove rows with empty quotes
data = data.dropna(subset=["quote"])

quotes = data["quote"].astype(str).tolist()
authors = data["author"].astype(str).tolist()

# Convert quotes to vectors
vectorizer = TfidfVectorizer(stop_words="english")
quote_vectors = vectorizer.fit_transform(quotes)

def get_quote(user_input):

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, quote_vectors)

    index = similarity.argmax()

    quote = quotes[index]
    author = authors[index]

    return f'"{quote}" — {author}'