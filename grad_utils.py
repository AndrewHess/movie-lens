import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def grad_U(Ui, Yij, Vj, reg, eta, ai=None, bj=None, mu=None):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    eta (the learning rate), and optionally ai, bj, mu (the bias of the ith
    user, jth movie, and the average value in Y). If ai, bj, and mu are not
    given, they are not used in the loss function.

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    if ai is None or bj is None or mu is None:
        grad = reg * Ui - Vj.dot((Yij - Ui.transpose().dot(Vj)).transpose())
    else:
        grad = reg * Ui - Vj.dot((Yij - mu - Ui.transpose().dot(Vj) - ai - bj))

    return eta * grad

def grad_V(Vj, Yij, Ui, reg, eta, ai=None, bj=None, mu=None):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    eta (the learning rate), and optionally ai, bj, mu (the bias of the ith
    user, jth movie, and the average value in Y). If ai, bj, and mu are not
    given, they are not used in the loss function.

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    if ai is None or bj is None or mu is None:
        grad = reg * Vj - Ui.dot((Yij - Ui.transpose().dot(Vj)).transpose())
    else:
        grad = reg * Vj - Ui.dot((Yij - mu - Ui.transpose().dot(Vj) - ai - bj))

    return eta * grad

def grad_ai(Ui, Yij, Vj, ai, bj, mu, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), ai (the bias for the ith user), bj (the bias
    for the jth movie), mu (the average value in Y), reg (the regularization
    parameter lambda), and eta (the learning rate).

    Returns the gradient of the regularized loss function with respect to ai
    multiplied by eta.
    """
    return eta * (reg * ai + (Yij - mu - Ui.transpose().dot(Vj) - ai - bj))

def grad_bj(Vj, Yij, Ui, ai, bj, mu, reg, eta):
    """
    Takes as input Vj (the jth column of U), a training point Yij, the row
    vector Ui (ith row of U), ai (the bias for the ith user), bj (the bias
    for the jth movie), mu (the average value in Y), reg (the regularization
    parameter lambda), and eta (the learning rate).

    Returns the gradient of the regularized loss function with respect to bj
    multiplied by eta.
    """
    return eta * (reg * bj + (Yij - mu - Ui.transpose().dot(Vj) - ai - bj))

def get_err(U, V, Y, reg=0.0, a=None, b=None, mu=None):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V. Also optionally takes a, b, and mu, the user
    and movie biases, and the average value in Y. If a, b, or mu is not given,
    the bias is not used to compute the error.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    r_term = reg * (np.trace(U.transpose().dot(U)) + np.trace(V.transpose().dot(V)))

    if a is not None and b is not None and mu is not None:
        r_term += reg * (np.sum(np.square(a)) + np.sum(np.square(b)))

    l_term = 0
    for i, j, Yij in Y:
        if a is None or b is None or mu is None:
            l_term += (Yij - U[i-1].transpose().dot(V[j-1])) ** 2
        else:
            l_term += (Yij - mu - U[i-1].transpose().dot(V[j-1]) - a[i-1] - b[j-1]) ** 2

    return (r_term + l_term) / 2 / len(Y)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300, bias=False):
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
    of the model if no bias is used. If bias is used, it returns (U, V, a, b, mu, err).
    """
    U = np.random.rand(M, K) - 0.5
    V = np.random.rand(N, K) - 0.5

    # Setup the bias parameters.
    if bias:
        a = np.random.random_sample(U.shape[0]) - 0.5
        b = np.random.random_sample(V.shape[0]) - 0.5
        mu = np.mean(Y[:, 2])

    for epoch in range(max_epochs):
        init_loss = get_err(U, V, Y, a=a, b=b, mu=mu) if bias else get_err(U, V, Y)
        print(epoch, "|", init_loss)

        np.random.shuffle(Y)

        for i, j, Yij in Y:
            if bias:
                U[i-1] -= grad_U(U[i-1], Yij, V[j-1], reg, eta, a[i-1], b[j-1], mu)
                V[j-1] -= grad_V(V[j-1], Yij, U[i-1], reg, eta, a[i-1], b[j-1], mu)
                a[i-1] -= grad_ai(U[i-1], Yij, V[j-1], a[i-1], b[j-1], mu, reg, eta)
                b[j-1] -= grad_bj(V[j-1], Yij, U[i-1], a[i-1], b[j-1], mu, reg, eta)
            else:
                U[i-1] -= grad_U(U[i-1], Yij, V[j-1], reg, eta)
                V[j-1] -= grad_V(V[j-1], Yij, U[i-1], reg, eta)


        err = get_err(U, V, Y, a=a, b=b, mu=mu) if bias else get_err(U, V, Y)

        if init_loss - err < eps:
            print(epoch, "|", err)
            break

    return (U, V, a, b, mu, err) if bias else (U, V, err)

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
