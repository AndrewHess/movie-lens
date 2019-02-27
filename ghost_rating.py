import numpy as np
from basic_visualization import get_movies_dict

all_data = np.loadtxt('data/data.txt')
all_train = np.loadtxt('data/train.txt')
all_test = np.loadtxt('data/test.txt')


def make_ghost():
    user_set = set([i[0] for i in all_data])
    movie_set = set([i[1] for i in all_data])
    rating_set = set([i[2] for i in all_data])


    train_movie_set = set([i[1] for i in all_train])
    ghost = int(max(user_set) + 1)

    # a = get_movies_dict(all_data)
    # all_movie_ids = set(np.genfromtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[0],dtype='int'))

    missing = [i for i in movie_set if i not in train_movie_set]
    print(missing)


    copy_ratings = [str(int(i[0])) + "\t" + str(int(i[1])) + "\t" + str(int(i[2])) for i in all_train]
    ghost_ratings = [str(ghost) + "\t" + str(int(i)) + "\t1" for i in missing]
    writeback = []
    writeback.extend(copy_ratings)
    writeback.extend(ghost_ratings)

    with open('data/train_ghost.txt', 'w+') as f:
        for i in writeback:
            f.write(i + "\n")



def sparse_analysis():
    # gets frequency of movies
    user_freq = [i[1] for i in all_data]
    from collections import Counter
    cnt = Counter(user_freq)
    print(cnt)
    freqlst = [cnt[i] for i in cnt.keys()]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle('number of ratings per movie', fontsize=18)
    plt.hist(freqlst, bins=120)
    plt.xlabel("ratings per single movie")
    plt.ylabel("amount of movies")
    plt.show()



sparse_analysis()