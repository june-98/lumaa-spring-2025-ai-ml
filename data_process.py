import pandas as pd
import json

"""Using Movie Dataset From https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data"""
MOVIE_COLUMNS = ['genres', 'keywords', 'production_companies', 'overview', 'release_date', 'status', 'title']
JSOIN_COLUMNS = ['genres', 'keywords', 'production_companies']

def get_movie_data(file_path: str) -> pd.DataFrame:
    """
    get pandas df from a given file path with specified colum names in MOVIE_COLUMNS
    :param file_path: file path
    :return: raw movies dataframe
    """
    df = pd.read_csv(file_path)
    return df[MOVIE_COLUMNS]

def data_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw movies dataframe, filter for only released movies and turn json-format string columns into lists
    :param data: movies dataframe
    :return: dataframe with processed columns
    """
    data = data[data.status == "Released"]
    for col in JSOIN_COLUMNS:
        data.loc[:,col] = data[col].apply(json.loads).apply(lambda row: [x['name'] for x in row])
    return data

def add_movie_description(movie_attributes: pd.Series) -> str:
    """
    Generate an aggregate movie description based on all the attributes of the movie
    :param data: movies dataframe
    :return: df with added column "Aggregate Movie Description"
    """
    movie_attributes = dict(movie_attributes)
    genres = f"The genres are {', '.join(movie_attributes['genres'])}."
    keywords = f"The movie has {', '.join(movie_attributes['keywords'])}."
    production_companies = f"The movie was produced by {', '.join(movie_attributes['production_companies'])}."
    overview = f"The movie is about {movie_attributes['overview']}."
    release_date = f"The movie was released on {movie_attributes['release_date']}."
    title = f"The title of the movie is {movie_attributes['title']}."
    return " ".join([title, release_date, genres, production_companies, keywords, overview])

def finalize_movie_dataset(file_name: str)-> pd.DataFrame:
    """Prepare the movies dataset description for vectorization"""
    movies_df = get_movie_data(file_name)
    movies_df = data_processing(movies_df)
    movies_df['aggregate_description'] = movies_df.apply(add_movie_description, axis=1)
    return movies_df