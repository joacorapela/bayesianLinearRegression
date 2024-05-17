import numpy as np


def batchWithSimplePrior(Phi, y, alpha, beta):
    """Performs batch linear regression with a simple prior.
    $P(w)=\mathcal{N}(0,\alpha^{-1}I)$

    :param Phi: matrix of basis functions transformations of indendent
    variables. $Phi\in\mathbb{R}^{NxP}$, where $N$ is the number of
    observations and $P$ the number of basis functions.
    $Phi[n,j]=\phi_j(\mathbf{x}_n)$.

    :type  Phi: numpy array

    :param y: dependent variable. $y\in\mathbb{R}^N$

    :type  y: numpy array

    :param alpha: prior precision

    :type  alpha: double

    :param beta: likelihood precision

    :type  beta: double
    """

    M = Phi.shape[1]
    SNinv = alpha*np.eye(M) + beta * Phi.T @ Phi
    mN = np.linalg.solve(a=SNinv, b=beta * Phi.T @ y)
    SN = np.linalg.inv(SNinv)
    return mN, SN


def onlineUpdate(mn, Sn, phi, y, alpha, beta):
    """Updates the posterior mean, mn, and posterior covariance, Sn, with the new
    sample, phi (basis functions expansion of the independent variables) and y
    (dependent variable.
    """

    aux1 = Sn @ phi
    aux2 = 1.0/(1.0 / beta + phi.T @ Sn @ phi)

    Snp1 = Sn - aux2 * np.outer(aux1, aux1)
    mnp1 = beta * y * Snp1 @ phi + mn - aux2 * np.inner(phi, mn) * Sn @ phi

    return mnp1, Snp1

def predict(phi, mn, Sn, alpha, beta):
    """Predicts the mean and variance of the dependent variable for a given
    new set of indepedent variables with basis function expansion in phi.

    :param phi: matrix of basis functions transformations of indendent
    variables. $Phi\in\mathbb{R}^{P}$, where $P$ the number of basis functions.
    $phi[j]=\phi_j(\mathbf{x})$.

    :type  phi: numpy array

    :param y: dependent variable. $y\in\mathbb{R}^N$

    :type  y: numpy array

    :param alpha: prior precision

    :type  alpha: double

    :param beta: likelihood precision

    :type  beta: double
    """

    predicted_mean = np.dot(mN, phi)
    predicted_var = 1.0/beta + np.dot(phi, np.dot(SN, phi))

    return predicted_mean, predicted_var
