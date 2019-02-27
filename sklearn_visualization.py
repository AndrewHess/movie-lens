import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from grad_utils import train_model, get_err, get_projected, draw_projection, get_average_predicted
from basic_visualization import get_popular_ids, get_best_ids, get_movies_dict, get_average_ratings
from scipy.signal import savgol_filter
#from sklearn.decomposition import NMF
import surprise as sp

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def main():
    reader = sp.Reader(line_format='user item rating', sep='\t')

    #folds_files = [('data/train.txt','data/test.txt')]
    folds_files = [('data/train_ghost.txt','data/test.txt')]

    data = sp.Dataset.load_from_folds(folds_files, reader=reader)
    pkf = sp.model_selection.PredefinedKFold()
    Y_train, Y_test = next(pkf.split(data))
    print(Y_train.n_users, Y_train.n_items, Y_train.n_ratings)

    Y_test2 = np.loadtxt('data/test.txt').astype(int)
    #M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    #N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies

    #print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20

    reg = 0.0
    eta = 0.03 # learning rate
    reg = 0.1
    eps = 0.0001

    #U,V, err = train_model(M, N, K, eta, reg, Y_train)
    model = sp.prediction_algorithms.matrix_factorization.SVD(n_factors = K, n_epochs = 100, reg_all = reg, verbose=True, biased = False, lr_all = eta, init_std_dev = 0.5)
    model.fit(Y_train)
    U = model.pu
    V = model.qi

    print(get_err(U, V, Y_test2))
    predictions = model.test(Y_test)
    print(sp.accuracy.rmse(predictions))
    #eout = get_err(U, V, Y_test)
    #print(eout)

    movies = get_projected(V.transpose()).transpose()
    data = np.loadtxt('data/data.txt')
    movies_dict = get_movies_dict(data)
    pop_ids = np.array(get_popular_ids(movies_dict)).astype(int)
    best_ids = np.array(get_best_ids(movies_dict)).astype(int)
    names = np.genfromtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[1],dtype='str')

    # best vs. error
    predicted = []
    actual = []
    for id in np.array(get_best_ids(movies_dict, -1)).astype(int):
        #print(id)
        actual.append(get_average_ratings(movies_dict, [id]))
        predicted.append(get_average_predicted(U, V, [id]))
    fig, ax = plt.subplots()
    diff = np.abs(np.array(predicted) - np.array(actual))
    s_diff = savgol_filter(diff, 101, 0)
    ax.plot(actual, s_diff)
    plt.show()

    # popular vs. error
    predicted = []
    actual = []
    for id in np.array(get_popular_ids(movies_dict, None, -1)).astype(int):
        #print(id)
        actual.append(get_average_ratings(movies_dict, [id]))
        predicted.append(get_average_predicted(U, V, [id]))
    fig, ax = plt.subplots()
    diff = np.abs(np.array(predicted) - np.array(actual))
    s_diff = savgol_filter(diff, 101, 0)
    ax.plot(range(len(predicted)), s_diff)
    plt.show()

    draw_projection(movies, [pop_ids], "Most popular films")

    draw_projection(movies, [best_ids], "Highest rated films")

    star_wars = [50, 172, 181]
    star_trek = [222, 227, 228, 229, 230, 380, 449, 450]
    other = [135, 204, 271, 343, 434, 176, 183, 665]

    draw_projection(movies, [star_wars, star_trek, other], "Star Wars vs. Star Trek")

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
        draw_projection(movies, [np.array(get_popular_ids(movies_dict, genre)).astype(int)], genre)

if __name__ == "__main__":
    main()
