import sys
import numpy as np
import matplotlib.pyplot as plt
from grad_utils import train_model, get_err, get_projected, draw_projection
from basic_visualization import get_popular_ids, get_best_ids, get_movies_dict

def main():
    # Check the command line arguments if for bias should be used.
    bias = (len(sys.argv) == 2 and sys.argv[1] == '-bias')

    print('using bias:', bias)

    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20

    reg = 0.0
    eta = 0.03 # learning rate
    reg = 0.1
    eps = 0.0001

    # Train the model.
    if bias:
        U, V, a, b, mu, err = train_model(M, N, K, eta, reg, Y_train, bias=True)
    else:
        U, V, err = train_model(M, N, K, eta, reg, Y_train)

    eout = get_err(U, V, Y_test, a=a, b=b, mu=mu) if bias else get_err(U, V, Y_test)
    print(err, eout)
    movies = get_projected(V.transpose()).transpose()
    data = np.loadtxt('data/data.txt')
    movies_dict = get_movies_dict(data)
    pop_ids = np.array(get_popular_ids(movies_dict)).astype(int)
    best_ids = np.array(get_best_ids(movies_dict)).astype(int)
    names = np.genfromtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[1],dtype='str')

    draw_projection(movies, [pop_ids])

    draw_projection(movies, [best_ids])

    star_wars = [50, 172, 181]
    star_trek = [222, 227, 228, 229, 230, 380, 449, 450]
    other = [135, 204, 271, 343, 434, 176, 183, 665]
    draw_projection(movies, [star_wars, star_trek, other])

    action_ids = np.array(get_popular_ids(movies_dict, 'action')).astype(int)
    crime_ids = np.array(get_popular_ids(movies_dict, 'crime')).astype(int)
    war_ids = np.array(get_popular_ids(movies_dict, 'war')).astype(int)

    draw_projection(movies, [action_ids, crime_ids, war_ids])

    genre_index = {'unknown': 1, 'action': 2, 'adventure': 3, 'animation': 4,
                   'childrens': 5, 'comedy': 6, 'crime': 7, 'documentary': 8,
                   'drama': 9, 'fantasy': 10, 'film-noir': 11, 'horror': 12,
                   'musical': 13, 'mystery': 14, 'romance': 15, 'sci-fi': 16,
                   'thriller': 17, 'war': 18, 'western': 19}
    for genre in genre_index.keys():
        draw_projection(movies, [np.array(get_popular_ids(movies_dict, 'war')).astype(int)])

if __name__ == "__main__":
    main()
