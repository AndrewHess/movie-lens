import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from grad_utils import train_model, get_err, get_projected, draw_projection, get_average_predicted, get_average_location, get_RMSE
from basic_visualization import get_popular_ids, get_best_ids, get_movies_dict, get_average_ratings, get_year, get_under_year
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation

def make_years_plot(movies, movies_dict, names):

    x = []
    y = []
    for max_year in range(1920,2000,1):
        best_ids = get_popular_ids(movies_dict, None, 3, get_year(max_year - 1))
        for id in best_ids:
            x.append(movies[id - 1][0])
            y.append(movies[id - 1][1])

    max_x = max(x)
    min_x = min(x)

    max_y = max(y)
    min_y = min(y)

    for max_year in range(1920,2000,1):
        points = []
        all_ids = get_year(max_year - 1)
        best_ids = get_popular_ids(movies_dict, None, 3, all_ids)
        print(max_year, best_ids)

        for year in range(1900,max_year,1):
            ids = get_under_year(year)
            if ids != []:
                points.append(get_average_location(movies, ids) + [year])

        for i,id in enumerate(all_ids):
            if id not in best_ids:
                x = points[-1][0]
                y = points[-1][1]
                dx = movies[id - 1][0] - x
                dy = movies[id - 1][1] - y
                plt.arrow(x, y, dx, dy, color="#ddeaff", width = 0.00002, zorder = 1)

        for i,id in enumerate(best_ids):
            x = points[-1][0]
            y = points[-1][1]
            dx = movies[id - 1][0] - x
            dy = movies[id - 1][1] - y
            plt.arrow(x, y, dx, dy, color="#96beff", width = 0.0002, zorder = 3)
            plt.text(x + dx, y + dy, str(i + 1), fontsize=9,  zorder = 5)
            plt.text(max_x - 0.8, min_y + 0.25 - 0.06*(i), str(i + 1) + ": " + names[id - 1], fontsize=9, zorder = 5)

        for i,pt in enumerate(points):
            x = pt[0]
            y = pt[1]
            year = pt[2]
            if year == max_year - 1:
                plt.scatter(x, y, marker='x', color='red', s = 40, zorder = 4)
                #plt.text(x, y, str(year), fontsize=9, color='blue')
            else:
                black_level = min(1, 0.1 * (max_year - year - 1))

                plt.scatter(x, y, marker='o', c = [black_level, black_level, black_level], s = 2, zorder = 2)

        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title("Average movie: <" + str(max_year))
        plt.savefig("years_plot/f" + str(max_year), dpi=96)
        plt.clf()
        plt.gca()

def main():
    # Check the command line arguments if for bias should be used.
    bias = (len(sys.argv) >= 2 and '-bias' in sys.argv)
    rmse = (len(sys.argv) >= 2 and '-rmse' in sys.argv)
    if (len(sys.argv) >= 2 and '-h' in sys.argv or '-help' in sys.argv):
        print("python3 p5_visualization.py -bias -rmse -help")
        sys.exit(0)


    print('using bias:', bias)
    print('using rmse:', rmse)

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

    if (rmse):
        print("rmse error: ")
        if(bias):
            print("ein: " + str(get_RMSE(U, V, Y_train, a=a, b=b, mu=mu)))
            print("eout: " + str(get_RMSE(U, V, Y_test, a=a, b=b, mu=mu)))
        else:
            print("ein: " + str(get_RMSE(U, V, Y_train)))
            print("eout: " + str(get_RMSE(U, V, Y_test)))


    movies = get_projected(V.transpose()).transpose()
    data = np.loadtxt('data/data.txt')
    movies_dict = get_movies_dict(data)
    pop_ids = np.array(get_popular_ids(movies_dict)).astype(int)
    best_ids = np.array(get_best_ids(movies_dict)).astype(int)
    names = np.genfromtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[1],dtype='str')

    make_years_plot(movies, movies_dict, names)
    # best vs. error
    predicted = []
    actual = []
    for id in np.array(get_best_ids(movies_dict, -1)).astype(int):
        #print(id)
        actual.append(get_average_ratings(movies_dict, [id]))
        if bias:
            predicted.append(get_average_predicted(U, V, [id], mu, a, b))
        else:
            predicted.append(get_average_predicted(U, V, [id]))
    fig, ax = plt.subplots()
    diff = np.array(predicted) - np.array(actual)
    s_diff = savgol_filter(diff, 101, 0)
    ax.plot(actual, diff)
    plt.show()

    # popular vs. error
    predicted = []
    actual = []
    for id in np.array(get_popular_ids(movies_dict, None, -1)).astype(int):
        #print(id)
        actual.append(get_average_ratings(movies_dict, [id]))
        if bias:
            predicted.append(get_average_predicted(U, V, [id], mu, a, b))
        else:
            predicted.append(get_average_predicted(U, V, [id]))
    fig, ax = plt.subplots()
    diff = np.array(predicted) - np.array(actual)
    s_diff = savgol_filter(diff, 101, 0)
    ax.plot(range(len(predicted)), diff)
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
