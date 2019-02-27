import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_hist(data, title=None):
    ''' Make a histogram of the given data. '''

    # Make the bins centered at 1, 2, 3, 4, 5.
    bins = np.array(range(75, 526, 50)) / 100

    # Make the histogram.
    # plt.hist(data, bins=bins, align='mid', density=True)
    plt.hist(data, bins=bins, align='mid')

    if title:
        plt.title(title)

    plt.xlabel('Rating')
    plt.ylabel('Occurance')
    plt.show()

    return


def main():
    ''' Make some visualizations. '''

    data = np.loadtxt('data/data.txt')
    movies_dict = get_movies_dict(data)
    # Make a full histogram of the data.
    make_hist(data[:, 2])

    # Get histograms for specific genres.
    show_genres(['action', 'crime', 'war'])

    # Make historgram of 10 most popular
    pop_ids = get_popular_ids(movies_dict)
    vis_by_ids(data, pop_ids)

    # Make histogram of 10 best
    best_ids = get_best_ids(movies_dict)
    vis_by_ids(data, best_ids)


def show_genres(genres):
    '''
    Make a histogram a separate histogram for each of the given genres.

    genres: a list of genre names.
    '''

    # Convert each genre to its corresponding index.
    genre_index = {'unknown': 1, 'action': 2, 'adventure': 3, 'animation': 4,
                   'childrens': 5, 'comedy': 6, 'crime': 7, 'documentary': 8,
                   'drama': 9, 'fantasy': 10, 'film-noir': 11, 'horror': 12,
                   'musical': 13, 'mystery': 14, 'romance': 15, 'sci-fi': 16,
                   'thriller': 17, 'war': 18, 'western': 19}

    # Load the file that maps movie IDs to their genres and the ratings file.
    movie_genres = np.loadtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[0] + list(range(2, 21)))
    all_data = np.loadtxt('data/data.txt')

    for genre in genres:
        # Get all movies with the given genre.
        movie_ids = []

        for movie in movie_genres:
            if movie[genre_index[genre]] == 1:
                movie_ids.append(int(movie[0]))

        make_hist(all_data[movie_ids][:, 2], title=genre)

    return


def get_genre(genre):
    genre_index = {'unknown': 1, 'action': 2, 'adventure': 3, 'animation': 4,
                   'childrens': 5, 'comedy': 6, 'crime': 7, 'documentary': 8,
                   'drama': 9, 'fantasy': 10, 'film-noir': 11, 'horror': 12,
                   'musical': 13, 'mystery': 14, 'romance': 15, 'sci-fi': 16,
                   'thriller': 17, 'war': 18, 'western': 19}

    # Load the file that maps movie IDs to their genres and the ratings file.
    movie_genres = np.loadtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[0] + list(range(2, 21)))
    all_data = np.loadtxt('data/data.txt')
    movie_ids = []
    for movie in movie_genres:
        if movie[genre_index[genre]] == 1:
            movie_ids.append(int(movie[0]))
    return movie_ids


def get_under_year(year):
    movie_dates = np.loadtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[0,1], dtype=str)
    movie_ids = []
    for movie in movie_dates:
        for i in range(1900,year + 1, 1):
            if movie[1].find(str(i)) != -1:
                movie_ids.append(int(movie[0]))
    return movie_ids

def get_year(year):
    movie_dates = np.loadtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[0,1], dtype=str)
    movie_ids = []
    for movie in movie_dates:
        if movie[1].find(str(year)) != -1:
            movie_ids.append(int(movie[0]))
    return movie_ids

def get_movies_dict(data):
    '''
    Get dictionary of movies where key is id, number of ratings and sum of
    ratings is value
    '''

    ids =  data[:, 1]
    ratings = data[:, 2]

    dict = {}
    for i in range(len(ids)):
        id = ids[i]
        rating = ratings[i]
        if id in dict:
            (num_ratings, total) = dict[id]
            dict[id] = (num_ratings + 1, total + rating)
        else:
            dict[id] = (1, rating)
    return dict


def get_popular_ids(movies_dict, genre = None, n = 10, ids = None):
    if genre != None:
        movies_dict = dict(zip(get_genre(genre), map(movies_dict.get, get_genre(genre))))
    if ids != None:
        movies_dict = dict(zip(ids, map(movies_dict.get, ids)))
    sorted_by_popular = sorted(movies_dict.items(), key=lambda x: x[1][0], reverse = True)
    popular = sorted_by_popular[:n]

    popular_ids = []
    for m in popular:
        id, details = m
        popular_ids.append(id)

    return popular_ids


def get_best_ids(movies_dict, n = 10):
    sorted_by_best = sorted(movies_dict.items(), key=lambda x: (x[1][1])/(x[1][0]), reverse = True)
    best = sorted_by_best[:n]

    best_ids = []
    for m in best:
        id, details = m
        best_ids.append(id)

    return best_ids


def vis_by_ids(data, ids, title=None):
    sorted = data[np.logical_or.reduce([data[:,1] == id for id in ids])]
    make_hist(sorted[:,2], title)

def get_average_ratings(movies_dict, ids):
    total = 0
    for id in ids:
        (num_ratings, total_rating) = movies_dict[id]
        total += (total_rating/num_ratings)
    return total/len(ids)

if __name__ == '__main__':
    main()
