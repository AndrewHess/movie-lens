import numpy as np
import matplotlib.pyplot as plt
from grad_utils import train_model, get_err, get_projected
from basic_visualization import get_popular_ids, get_best_ids, get_movies_dict

def main():
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

    U,V, err = train_model(M, N, K, eta, reg, Y_train)
    eout = get_err(U, V, Y_test)
    print(err, eout)
    #print(V.shape)
    projection = get_projected(V)
    movies = V.dot(projection.transpose())
    data = np.loadtxt('data/data.txt')
    movies_dict = get_movies_dict(data)
    pop_ids = np.array(get_popular_ids(movies_dict)).astype(int)
    best_ids = np.array(get_best_ids(movies_dict)).astype(int)
    names = np.genfromtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[1],dtype='str')


    '''
    genre_index = {'unknown': 1, 'action': 2, 'adventure': 3, 'animation': 4,
                   'childrens': 5, 'comedy': 6, 'crime': 7, 'documentary': 8,
                   'drama': 9, 'fantasy': 10, 'film-noir': 11, 'horror': 12,
                   'musical': 13, 'mystery': 14, 'romance': 15, 'sci-fi': 16,
                   'thriller': 17, 'war': 18, 'western': 19}

    colors = ['blue','red','yellow','orange','green','purple','pink','black']
    # Load the file that maps movie IDs to their genres and the ratings file.
    movie_genres = np.loadtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[0] + list(range(2, 21)))

    genres = ['musical', 'crime']
    for genre in genres:
        # Get all movies with the given genre.
        movie_ids = []

        for movie in movie_genres:
            if movie[1 + genre_index[genre]] == 1:
                movie_ids.append(int(movie[0]))

        to_display = data[movie_ids]
        for i,id in enumerate(movie_ids):
            x = to_display[:,0][i]
            y = to_display[:,1][i]
            plt.scatter(x, y, marker='o', color=colors[genre_index[genre] % len(colors)])
            #plt.text(x+0.3, y+0.3, id, fontsize=9)

    plt.show()
    '''
    to_display_pop = movies[pop_ids-1]
    to_display_best = movies[best_ids-1]
    for i,id in enumerate(pop_ids):
        x = to_display_pop[:,0][i]
        y = to_display_pop[:,1][i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x+0.3, y+0.3, names[id-1], fontsize=9)
    plt.show()

    for i,id in enumerate(best_ids):
        x = to_display_best[:,0][i]
        y = to_display_best[:,1][i]
        plt.scatter(x, y, marker='o', color='blue')
        plt.text(x+0.3, y+0.3, names[id], fontsize=9)
    plt.show()

if __name__ == "__main__":
    main()
