import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - Vj.dot((Yij - Ui.transpose().dot(Vj)).transpose()))

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - Ui.dot((Yij - Ui.transpose().dot(Vj)).transpose()))

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    r_term = reg * (np.trace(U.transpose().dot(U)) + np.trace(V.transpose().dot(V)))
    l_term = 0
    for i, j, Yij in Y:
        l_term += (Yij - U[i-1].transpose().dot(V[j-1])) ** 2

    return (r_term + l_term) / 2 / len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    print(Y)
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.rand(M, K) - 0.5
    V = np.random.rand(N, K) - 0.5

    #print(U.shape)
    #print(V.shape)

    for epoch in range(max_epochs):
        init_loss = get_err(U, V, Y)
        print(epoch, "|", init_loss)

        np.random.shuffle(Y)

        for i, j, Yij in Y:
            U[i-1] -= grad_U(U[i-1], Yij, V[j-1], reg, eta)
            V[j-1] -= grad_V(V[j-1], Yij, U[i-1], reg, eta)

        if init_loss - get_err(U, V, Y) < eps:
            return (U, V, get_err(U, V, Y))

    return (U, V, get_err(U, V, Y))

def get_projected(V):
    A, S, B = np.linalg.svd(V,full_matrices=False)
    print(A.shape,V.shape)
    return (A[:,:2].transpose()).dot(V)

def draw_projection(projected, ids):
    colors = ['red','blue','green','yellow','orange','purple','black']
    data = np.loadtxt('data/data.txt')
    names = np.genfromtxt('data/movies.txt', delimiter='\t', encoding='latin1', usecols=[1],dtype='str')
    for ic,c in enumerate(ids):
        for i,id in enumerate(c):
            x = projected[id-1][0]
            y = projected[id-1][1]
            plt.scatter(x, y, marker='x', color=colors[ic % len(colors)])
            plt.text(x, y, names[id-1], fontsize=9)
    plt.show()
