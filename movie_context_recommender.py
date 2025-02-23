import argparse
from data_process import finalize_movie_dataset
import tensorflow_hub as hub

UNIVERSAL_ENCODER_LINK = r"https://tfhub.dev/google/universal-sentence-encoder/4"

def load_universal_sentence_encoder():
    model = hub.load(UNIVERSAL_ENCODER_LINK)
    print("module %s loaded" % UNIVERSAL_ENCODER_LINK)
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_path",
                        type=str,
                        help="Enter the path to save dataset file",
                        default = r"Data/tmdb_5000_movies.csv")
    parser.add_argument("--user_input",
                        type=str,
                        help="Enter user movie preference",
                        # TODO: remove this after testing
                        default = "I love thrilling action movies set in space, with a comedic twist.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data = finalize_movie_dataset(args.read_path)
    encoder = load_universal_sentence_encoder()
    print('hi')