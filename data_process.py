import pandas as pd
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('stopwords')
nltk.download('punkt')

"""Using Movie Dataset From https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data"""
MOVIE_COLUMNS = ['genres', 'keywords', 'production_companies', 'overview', 'status', 'title', 'tagline']
JSOIN_COLUMNS = ['genres', 'keywords', 'production_companies']

def get_movie_data(file_path: str) -> pd.DataFrame:
    """
    get pandas df from a given file path with specified column names in MOVIE_COLUMNS
    :param file_path: file path
    :return: raw movies dataframe
    """
    df = pd.read_csv(file_path)
    return df[MOVIE_COLUMNS].dropna(axis=0)

def data_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw movies dataframe, filter for only released movies and turn json-format string columns into lists
    :param data: movies dataframe
    :return: dataframe with json columns processed
    """
    data = data[data.status == "Released"]
    for col in JSOIN_COLUMNS:
        data.loc[:,col] = data[col].apply(json.loads).apply(lambda row: [x['name'] for x in row])
    return data

def text_preprocessing(description: str) -> str:
    """
    Preprocessing movie description for NLP tasks
    1) normalization: lowercasing & remove punctuations
    2) stop words removal
    3) lemmatization
    :param description: raw movie description
    :return: description after text pre-processing
    """
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    tokens = word_tokenize(description)
    # removes any character that is not a letter, number, or space
    pattern = r'[^a-zA-Z0-9\s]'
    stop_words = set(stopwords.words('english'))
    for token in tokens:
        new_token = re.sub(pattern, '', token).lower()
        if new_token and new_token not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(new_token))
    return " ".join(cleaned_tokens)

def add_movie_description(movie_attributes: pd.Series) -> str:
    """
    Generate an aggregate movie description based on all the attributes of the movie
    :param movie_attributes: genres, keywords, production companies, and overview
    :return: aggregate movie description
    """
    movie_attributes = dict(movie_attributes)

    genres = ' '.join(movie_attributes['genres']) if movie_attributes['genres'] else ''
    keywords = ' '.join(movie_attributes['keywords']) if movie_attributes['keywords'] else ''
    production_companies = ' '.join(movie_attributes['production_companies']) if movie_attributes['production_companies'] else ''
    description = " ".join([genres, keywords, movie_attributes['overview'], production_companies, movie_attributes['tagline']])
    return text_preprocessing(description)

def finalize_movie_dataset(file_name: str)-> pd.DataFrame:
    """
    Prepare the movies dataset for vectorization
    :param file_name: movie dataset file path
    :return: movies_df
    """
    movies_df = get_movie_data(file_name)
    movies_df = data_processing(movies_df)
    movies_df['aggregate_description'] = movies_df.apply(add_movie_description, axis=1)
    return movies_df