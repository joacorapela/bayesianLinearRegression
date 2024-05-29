"""
Predictive distribution
=======================

Calculation of the predictive istribution of Bayesian linear regression model.

"""

#%%
# Import requirments
# ------------------

import numpy as np
import scipy.stats
import plotly.graph_objects as go

import bayesianLinearRegression

#%%
# Define a function to generate sinusoidal regression data
# --------------------------------------------------------

def generateData(x, sigma=0.3):
    y = np.sin(2*np.pi*x)
    t = y + np.random.normal(loc=0, scale=sigma, size=len(y))
    return y, t


#%%
# Define functions to generate the design matrix sinusoidal regression data
# -------------------------------------------------------------------------

def getGaussianBasisFunctions(mus, sigma):
    M = len(mus)
    basis_functions = [None for m in range(M)]
    for m in range(M):
        basis_functions[m] = lambda x, mu=mus[m], sigma=sigma: \
            np.exp(-(x-mu)**2/(2.0*sigma**2))
    return basis_functions


def buildGaussianDesignMatrixRow(x, basis_functions):
    M = len(basis_functions)
    design_matrix_row = np.empty(shape=M, dtype=np.double)
    for m in range(M):
        design_matrix_row[m] = basis_functions[m](x)
    return design_matrix_row


def buildGaussianDesignMatrix(x, basis_functions):
    M = len(basis_functions)
    N = len(x)
    design_matrix = np.empty(shape=(N, M), dtype=np.double)
    for n in range(N):
        design_matrix[n,:] = buildGaussianDesignMatrixRow(x=x[n],
                                                          basis_functions=basis_functions)
    return design_matrix


#%%
# Generate train data
# -------------------

N = 10
# N = 25
# N = 4
x = np.sort(np.random.uniform(size=N))
_, t = generateData(x=x)
x_dense = np.linspace(0, 1, 1000)
y_dense, _ = generateData(x=x_dense)

#%%
# Plot train data
# ---------------

fig = go.Figure()
trace_true = go.Scatter(x=x_dense, y=y_dense, mode="lines", line_color="green")
trace_data = go.Scatter(x=x, y=t, mode="markers", marker_color="blue")
fig.add_trace(trace_true)
fig.add_trace(trace_data)
fig.update_layout(xaxis_title="independent variable",
                  yaxis_title="dependent variable",
                  showlegend=False)
fig

#%%
# Set estimation parameters
# -------------------------

bf_mus = np.arange(0.1, 1.0, 0.1)
bf_sigma = 1.0/(N-1)
prior_precision = 2.0
likelihood_precision = 25.0
N_new = 100

#%%
# Get and plot the basis functions
# --------------------------------

basis_functions = getGaussianBasisFunctions(mus=bf_mus, sigma=bf_sigma)

fig = go.Figure()
for i in range(len(basis_functions)):
    basis_function_values = basis_functions[i](x_dense)
    trace = go.Scatter(x=x_dense, y=basis_function_values, mode="lines")
    fig.add_trace(trace)
fig.update_layout(xaxis_title="x",
                  yaxis_title=r"$\phi_i(x)$",
                  showlegend=False)
fig

#%%
# Build design matrix
# -------------------

Phi = buildGaussianDesignMatrix(x=x, basis_functions=basis_functions)

#%%
# Estimate posterior distribution
# -------------------------------

mN, SN = bayesianLinearRegression.batchWithSimplePrior(
    Phi=Phi, y=t, alpha=prior_precision, beta=likelihood_precision)

#%%
# Estimate predictive distribution
# --------------------------------

new_x = np.sort(np.random.uniform(size=N_new))
true_mean = np.empty(shape=N_new, dtype=np.double)
new_mean = np.empty(shape=N_new, dtype=np.double)
new_var = np.empty(shape=N_new, dtype=np.double)
for n in range(N_new):
    true_mean[n] = np.sin(2*np.pi*new_x[n])
    phi = buildGaussianDesignMatrixRow(x=new_x[n],
                                       basis_functions=basis_functions)
    new_mean[n], new_var[n] = bayesianLinearRegression.predict(
        phi=phi, mn=mN, Sn=SN, beta=likelihood_precision)

#%%
# Plot predictive distribution
# ----------------------------

new_mean_upper = new_mean + 1.96*np.sqrt(new_var)
new_mean_lower = new_mean - 1.96*np.sqrt(new_var)
fig = go.Figure()
trace_true = go.Scatter(x=new_x, y=true_mean, mode="lines", line_color="green")
trace_mean = go.Scatter(x=new_x, y=new_mean, mode="lines", line_color="red")
trace_mean_cb = go.Scatter(x=np.concatenate((new_x, new_x[::-1])),
                           y=np.concatenate((new_mean_upper,
                                             new_mean_lower[::-1])),
                           fill="toself",
                           fillcolor="rgba(255,0,0,0.3)",
                           line=dict(color="rgba(255,255,255,0)"),
                           hoverinfo="skip",
                           showlegend=False,
                          )
trace_data = go.Scatter(x=x, y=t, mode="markers", marker_color="blue",
                        marker_symbol="circle-open", marker_size=10)
fig.add_trace(trace_true)
fig.add_trace(trace_mean)
fig.add_trace(trace_mean_cb)
fig.add_trace(trace_data)
fig.update_layout(xaxis_title="independent variable",
                  yaxis_title="dependent variable",
                  showlegend=False)
fig

# sphinx_gallery_thumbnail_path = '_static/predictive_distribution.png'
