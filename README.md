# Book Recommender System

This project implements a book recommendation system using collaborative filtering and popularity-based methods.

## Project Structure

- `books.csv`: Dataset containing book information.
- `users.csv`: Dataset containing user information.
- `ratings.csv`: Dataset containing user ratings for books.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/book-recommender-system.git
    cd book-recommender-system
    ```

2. Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn
    ```

## Usage

1. Load the datasets:
    ```python
    import numpy as np
    import pandas as pd

    books = pd.read_csv('books.csv')
    users = pd.read_csv('users.csv')
    ratings = pd.read_csv('ratings.csv')
    ```

2. Check for any warnings related to data types during the loading of CSV files:
    ```python
    books = pd.read_csv('books.csv', dtype={'Column3': 'your_dtype'})
    ```

3. View the first few rows of each dataset:
    ```python
    print(users.head())
    print(ratings.head())
    ```

4. Inspect the shape of the datasets:
    ```python
    print(books.shape)
    print(ratings.shape)
    print(users.shape)
    ```

5. Check for missing values:
    ```python
    print(books.isnull().sum())
    print(users.isnull().sum())
    print(ratings.isnull().sum())
    ```

6. Check for duplicates:
    ```python
    print(books.duplicated().sum())
    print(ratings.duplicated().sum())
    print(users.duplicated().sum())
    ```

## Popularity Based Recommender System

1. Merge ratings with books data:
    ```python
    ratings_with_name = ratings.merge(books, on='ISBN')
    ```

2. Calculate the number of ratings for each book:
    ```python
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
    ```

3. Calculate the average rating for each book:
    ```python
    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    ```

4. Merge the number of ratings and average rating dataframes:
    ```python
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    ```

5. Filter and sort the most popular books:
    ```python
    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]
    ```

## Collaborative Filtering Based Recommender System

1. Filter users who have rated more than 200 books:
    ```python
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    padhe_likhe_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name["User-ID"].isin(padhe_likhe_users)]
    ```

2. Filter books that have more than 50 ratings:
    ```python
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    ```

3. Create a pivot table of user ratings:
    ```python
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    ```

4. Calculate the cosine similarity between books:
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(pt)
    print(similarity_scores)
    ```


