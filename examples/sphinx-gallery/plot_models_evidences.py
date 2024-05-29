

"""
Plot model evidence
===================

Plot the evidence of models of different polynomial order.

"""

#%%
# Import requirments
# ------------------

import numpy as np
import plotly.graph_objects as go

import bayesianLinearRegression

#%%
# Define functions to generate the design matrix sinusoidal regression data
# -------------------------------------------------------------------------

def getPolynomialBasisFunctions(M):
    basis_functions = [None for m in range(M+1)]
    for m in range(M+1):
        basis_functions[m] = lambda x, m=m: x**m
    return basis_functions


def buildDesignMatrixRow(x, basis_functions):
    M = len(basis_functions)
    design_matrix_row = np.empty(shape=M, dtype=np.double)
    for m in range(M):
        design_matrix_row[m] = basis_functions[m](x)
    return design_matrix_row


def buildDesignMatrix(x, basis_functions):
    M = len(basis_functions)
    N = len(x)
    design_matrix = np.empty(shape=(N, M), dtype=np.double)
    for n in range(N):
        design_matrix[n, :] = buildDesignMatrixRow(
            x=x[n], basis_functions=basis_functions)
    return design_matrix


#%%
# Define a function to generate polynomial data
# ---------------------------------------------

def generateData(x, sigma, coefs=np.array([-0.5, 0.5, -0.5, 0.5, -0.5])):
    basis_functions = getPolynomialBasisFunctions(M=len(coefs)-1)
    Phi = buildDesignMatrix(x=x, basis_functions=basis_functions)
    y = Phi @ coefs
    noise = np.random.normal(loc=0, scale=sigma, size=len(y))
    t = y + noise
    return y, t


#%%
# Set estimation parameters
# -------------------------

prior_precision = 10.0
likelihood_precision = 10.0

#%%
# Generate train and test data
# ----------------------------

N = 50
x = 1.0 + np.random.uniform(size=N)
_, y = generateData(x=x, sigma=1.0/likelihood_precision)

#%%
# Calculate model evindences
# --------------------------

Ms = np.arange(10)
log_evidences = [None for m in Ms]
for M in Ms:
    basis_functions = getPolynomialBasisFunctions(M=M)
    Phi = buildDesignMatrix(x=x, basis_functions=basis_functions)
    mN, SN = bayesianLinearRegression.batchWithSimplePrior(
        Phi=Phi, y=y, alpha=prior_precision, beta=likelihood_precision)
    log_evidences[M] = bayesianLinearRegression.computeLogEvidence(
        Phi=Phi, y=y, mN=mN, SN=SN,
        alpha=prior_precision, beta=likelihood_precision)

#%%
# Plot models' log evidences
# --------------------------

fig = go.Figure()
trace = go.Scatter(x=Ms, y=log_evidences, mode="lines+markers",
                   line=dict(color="blue"))
fig.add_trace(trace)
fig.update_layout(xaxis_title="M",
                  yaxis_title=r"$\log p(\mathbf{y}|\alpha,\beta)$")
fig.show()
