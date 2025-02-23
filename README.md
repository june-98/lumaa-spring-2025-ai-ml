# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST

---

## Overview

This is a **content-based recommendation system** implemented with TF-IDF vectors and cosine similarity, 2

that, given a **short text description** of a user’s preferences, suggests **similar items** movies from a small dataset.

## Dataset

The dataset used is the TMDB 5000 Movie Dataset from Kaggle (https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data). For easier demonstration, only the top 500 movie entries in the final processed dataset is utilized.

The following movie attributes are used for the recommendation system:
1) title
2) tagline
3) production companies
4) status: Post Production, Released, Rumored
5) genres
6) keywords
7) overview: summary of the movie plot

The dataset used is in path: Data/tmdb_5000_movies.csv

## Setup
1) Create conda environment using the yml file to install all the dependencies and packages
   conda env create --file env.yml
2) Activate Conda environment
   conda activate content_recommender

Here is a quick video on setting up the environent:

## Running
1) To run the recommendation system in command line, using the following format:
   python -i movie_context_recommender.py --user_input [USER_INPUT] --read_path [DATA_PATH] -- top_k [top_k_recommendation]

   DATA_PATH is defaulted to Data/tmdb_5000_movies.csv
   top_k_recommendation is defaulted to 5

   EX: python -i movie_context_recommender.py --user_input "I love thrilling action movies set in space, with a comedic twist."

## Results
Below is an example query & output from the recommendation system: https://drive.google.com/file/d/13ZnHdpbzGsis2acuSEPuYcCHm3fDrMaZ/view?usp=sharing

User preference input is: I love thrilling action movies set in space, with a comedic twist.
Top 5 recommendations are:
1) 'Last Action Hero' with similarity score 0.4285
2) 'Tropic Thunder' with similarity score 0.3546
3) 'Armageddon' with similarity score 0.3464
4) 'Deep Impact' with similarity score 0.3378
5) 'Déjà Vu' with similarity score 0.3302

## Video Decom
Here is a link to a short video demo: https://drive.google.com/file/d/1PTKN2MgLq-x4GRh_rNlSO2peDww5NUN6/view?usp=sharing

## Salary Expectation
Internship Expectation: $20-30/hr