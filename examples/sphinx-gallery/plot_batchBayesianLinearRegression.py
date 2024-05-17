import numpy as np
import scipy.stats
import plotly.graph_objects as go

import bayesianLinearRegression


n_samples = 20
a0 = -0.3
a1 = 0.5
likelihood_precision_coef = (1/0.2)**2

x = np.random.uniform(low=-1, high=1, size=n_samples)
y = a0 + a1 * x
t = y + np.random.standard_normal(size=y.shape) * 1.0/likelihood_precision_coef

n_post_samples = 6
marker_true = "cross"
size_true = 10
color_true = "red"
marker_data = "circle-open"
size_data = 10
color_data = "blue"
line_width_data = 5
x_dense = np.arange(-1.0, 1.0, 0.1)

y_true = a0 + a1 * x_dense
fig = go.Figure()
trace_true = go.Scatter(x=x_dense, y=y_true, mode="lines", line_color="green",
   showlegend=False)
fig.add_trace(trace_true)
trace_data_points = go.Scatter(x=x, y=t,
   mode="markers",
   marker_symbol=marker_data,
   marker_size=size_data,
   marker_color=color_data,
   marker_line_width=line_width_data,
   showlegend=False,
  )
fig.add_trace(trace_data_points)
fig.update_xaxes(title_text="x")
fig.update_yaxes(title_text="y")

prior_precision_coef = 2.0

Phi = np.column_stack((np.ones(len(x)), x))
mN, SN = bayesianLinearRegression.batchWithSimplePrior(
Phi=Phi, y=y, alpha=prior_precision_coef,
beta=likelihood_precision_coef)


x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_grid, Y_grid))

rv = scipy.stats.multivariate_normal(mN, SN)
Z = rv.pdf(pos)

fig = go.Figure()

trace_post = go.Contour(x=x_grid, y=y_grid, z=Z, showscale=False)
fig.add_trace(trace_post)

trace_true_coef = go.Scatter(x=[a0], y=[a1], mode="markers",
 marker_symbol=marker_true,
 marker_size=size_true,
 marker_color=color_true,
 name="true mean")
fig.add_trace(trace_true_coef)
fig.add_vline(x=0, line_color="white")
fig.add_hline(y=0, line_color="white")
fig.update_layout(xaxis_title="Intercept",
  yaxis_title="Slope")
fig

fig = go.Figure()

samples = rv.rvs(size=n_post_samples)
for a_sample in samples:
sample_intercept, sample_slope = a_sample
sample_y = sample_intercept + sample_slope * x_dense
trace = go.Scatter(x=x_dense, y=sample_y, mode="lines",
   line_color="red", showlegend=False)
fig.add_trace(trace)
fig.update_xaxes(title_text="x")
fig.update_yaxes(title_text="y")

fig.add_trace(trace_data_points)
fig
