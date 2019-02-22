import numpy as np
import matplotlib.pyplot as plt


# def get_data(filename):
#     ''' Get the data from a tab deliminated file. '''
#

def main():
    ''' Make some visualizations. '''

    data = np.loadtxt('data/data.txt')
    print('ratings:', data[:, 2][:10])

    # Make a full histogram of the data.
    plt.hist(data[:, 2], bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25], align='mid')
    plt.show()


if __name__ == '__main__':
    main()
