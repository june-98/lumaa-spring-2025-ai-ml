import kagglehub
import argparse
from kagglehub import KaggleDatasetAdapter

def download_dataset(file_path):
    # Load the latest version
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "tmdb/tmdb-movie-metadata",
        file_path)
    return df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path",
                        type=str,
                        help="Enter the path to save dataset file",
                        default = r"Data/movies.csv")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data = download_dataset(args.save_path)
    # path = kagglehub.dataset_download(r"tmdb/tmdb-movie-metadata", path=args.save_path)
    print('hi')