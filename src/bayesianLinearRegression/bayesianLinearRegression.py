import numpy as np


def batchWithSimplePrior(Phi, y, alpha, beta):
    """Performs batch linear regression with a simple prior.
    :math:`P(w)=\\mathcal{N}(0,\\alpha^{-1}I)`

    :param Phi: matrix of basis functions transformations of indendent variables. :math:`\\text{Phi}\\in\\mathbb{R}^{NxP}`, where :math:`N` is the number of observations and :math:`P` the number of basis functions.  :math:`\\text{Phi}[n,j]=\\phi_j(\\mathbf{x}_n)`.

    :type  Phi: numpy array

    :param y: dependent variable. :math:`y\\in\\mathbb{R}^N`

    :type  y: numpy array

    :param alpha: prior precision

    :type  alpha: double

    :param beta: likelihood precision

    :type  beta: double

    :return: mean and covariance of the estimated coefficients
    :rtype: mean: numpy array of size P; covariance: numpy array of size PxP
    """

    M = Phi.shape[1]
    SNinv = alpha*np.eye(M) + beta * Phi.T @ Phi
    mN = np.linalg.solve(a=SNinv, b=beta * Phi.T @ y)
    SN = np.linalg.inv(SNinv)
    return mN, SN


def onlineUpdate(mn, Sn, phi, y, alpha, beta):
    """Updates the posterior mean, :math:`mn`, and posterior covariance, :math:`Sn`, with the new sample, :math:`phi` (basis functions expansion of the independent variables) and :math:`y` (dependent variable).

    :type  mn: numpy array

    :param mn: current posterior mean. :math:`mn\\in\\mathbb{R}^P`

    :type  Sn: numpy array

    :param Sn: current posterior covariance.  :math:`Sn\\in\\mathbb{R}^{P\\times P}`

    :param phi: bais function expansion of independent variables. :math:`phi\\in\\mathbb{R}^P`

    :type  phi: numpy array

    :param y: dependent variable. :math:`y\\in\\mathbb{R}`

    :type  y: double

    :param alpha: prior precision

    :type  alpha: double

    :param beta: likelihood precision

    :type  beta: double

    :return: updated mean and covariance
    :rtype: mean: numpy array of size P; covariance: numpy array of size PxP
    """

    aux1 = Sn @ phi
    aux2 = 1.0/(1.0 / beta + phi.T @ Sn @ phi)

    Snp1 = Sn - aux2 * np.outer(aux1, aux1)
    mnp1 = beta * y * Snp1 @ phi + mn - aux2 * np.inner(phi, mn) * Sn @ phi

    return mnp1, Snp1

def predict(phi, mn, Sn, alpha, beta):
    """Predicts the mean and variance of the dependent variable for a given new set of indepedent variables with basis function expansion in phi.

    :param phi: bais function expansion of independent variables. :math:`phi\\in\\mathbb{R}^P`

    :type  phi: numpy array

    :type  mn: numpy array

    :param mn: current posterior mean. :math:`mn\\in\\mathbb{R}^P`

    :type  Sn: numpy array

    :param Sn: current posterior covariance.  :math:`Sn\\in\\mathbb{R}^{P\\times P}`

    :param alpha: prior precision

    :type  alpha: double

    :param beta: likelihood precision

    :type  beta: double

    :return: mean and variance of prediction
    :rtype: mean: double; covariance: double
    """

    predicted_mean = np.dot(mN, phi)
    predicted_var = 1.0/beta + np.dot(phi, np.dot(SN, phi))

    return predicted_mean, predicted_var
