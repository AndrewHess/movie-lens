import numpy as np
from basic_visualization import get_movies_dict

all_data = np.loadtxt('data/data.txt')
all_train = np.loadtxt('data/train.txt')
all_test = np.loadtxt('data/test.txt')
print(all_data)

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

