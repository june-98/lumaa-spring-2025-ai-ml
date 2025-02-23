import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_process import finalize_movie_dataset, text_preprocessing

RECOMMENDATION_COLUMNS = ['title', 'tagline', 'genres', 'overview', 'similarity_score']

def load_tfidf(corpus: List[str]) -> np.ndarray:
    """
    tf-idf vectorization given a list of documents in the corpus
    :param corpus: document list
    :return: tf-idf vectors
    """
    model = TfidfVectorizer(ngram_range=(1,2), norm='l2', smooth_idf=True, max_features=500)
    output = model.fit_transform(corpus)
    return output.toarray()

def context_recommender(user_input: str, movies_df: pd.DataFrame, encoder) -> pd.DataFrame:
    """
    Given a user preference string and a dataset of movies, compute the cosine similarity scores based on encoded vectors.
    The similarity scores are ranked in descending order.
    :param user_input: user preference string
    :param movies_df: movies dataset
    :param encoder: encoder function such that encoder(corpus) returns the normalized vectors for each document in the corpus
    :return: movies df in descending order of the cosine similarity scores
    """
    user_input = text_preprocessing(user_input)
    embeddings = encoder([user_input.lower()] + movies_df.aggregate_description.tolist())
    user_embedding, movies_embeddings = embeddings[0], embeddings[1:]
    # cosine similarity is the dot product since the vectors are already normalized
    similarity = (movies_embeddings @ user_embedding.T).squeeze()
    movies_df['similarity_score'] = similarity
    return movies_df

def get_top_recommendations(movie_df: pd.DataFrame, top_k:int) -> pd.DataFrame:
    """
    helper_function to filter for top k movies in the dataframe based on similarity score
    :param movie_df: movies dataframe
    :param top_k: top k integer
    :return: top k movie recommendations
    """
    movie_df = movie_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    top_rec = movie_df.head(top_k)
    return  top_rec[RECOMMENDATION_COLUMNS]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_path",
                        type=str,
                        help="Enter the path to save dataset file",
                        default = r"Data/tmdb_5000_movies.csv")
    parser.add_argument("--user_input",
                        type=str,
                        required=True,
                        help="Enter user movie preference")
    parser.add_argument("--top_k",
                        type=int,
                        help="enter top k recommendations wanted",
                        # TODO: remove this after testing
                        default=5)
    args = parser.parse_args()
    return args

def main(file_path: str, user_input: str, top_k: int) -> pd.DataFrame:
    """Runs movie recommender given movie dataset path, user preference input, and top k integer"""
    data = finalize_movie_dataset(file_path)
    recommendations = context_recommender(user_input=user_input,
                                          movies_df=data,
                                          encoder=load_tfidf)
    top_recs = get_top_recommendations(recommendations, top_k)
    return top_recs

if __name__ == "__main__":
    args = get_args()
    recs = main(args.read_path, args.user_input, args.top_k)
    print(recs)