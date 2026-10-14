# Incremental Gaussian Process Regression for MATLAB

`incrementalRegressionGP` is a fixed-basis recursive Gaussian process (RGP)
regression model for streaming numeric data. It updates a Gaussian posterior
without retaining the full data stream. With fixed hyperparameters, its state
uses O(M^2 + MP) storage for `M` basis points and `P` predictors. Online ARD
hyperparameter learning adds O(MP + P^2) state storage.

## Quick start

```matlab
x = linspace(0,1,40).';
y = sin(6*x) + 0.05*randn(size(x));

options = struct( ...
    'BasisVectors',x(1:15), ...
    'LengthScale',0.3, ...
    'SignalVariance',1, ...
    'NoiseVariance',0.05^2);
Mdl = incrementalRegressionGP(x(1:15),y(1:15),options);

% Assimilate each new batch; assign the returned value-class model.
Mdl = update(Mdl,x(16:20),y(16:20));

xTest = linspace(0,1,100).';
[yPred,ySD,yInterval] = predict(Mdl,xTest);
plot(xTest,yPred,'b-',xTest,yInterval,'b:')
```

The inputs are numeric matrices with observations in rows. `NoiseVariance` is
a variance, not a standard deviation. `predict` returns response predictions,
standard deviations, and pointwise prediction intervals; use
`struct('Alpha',0.01)` to request 99% intervals, or set
`IncludeObservationNoise` to `false` for latent-function uncertainty.

## Features and scope

- Fixed-basis recursive updates; memory does not grow with the stream.
- Squared-exponential, Matern 3/2, and Matern 5/2 kernels, including ARD
  length scales, with optional fixed mean functions.
- Batch and single-observation updates, reset, and prequential MSE metrics.
- Import from exact `fitrgp` models with any of the supported kernels.
- MATLAB Coder support for numeric construction, fitting, prediction, and metrics.

Select a kernel with `options.KernelFunction`: `'squaredexponential'` (default),
`'matern32'`, `'matern52'`, `'ardsquaredexponential'`, `'ardmatern32'`, or
`'ardmatern52'`. `LengthScale` accepts a positive scalar or one value per
predictor, preserving the existing vector-length-scale behavior for every
kernel. Explicit ARD names expand a scalar to one independently learnable
scale per predictor. All kernels use `SignalVariance` as their diagonal
covariance and support prediction intervals and online hyperparameter learning.

```matlab
options = struct('KernelFunction','ardmatern52', ...
    'LengthScale',[0.3 1.2],'SignalVariance',1,'NoiseVariance',0.01);
Mdl = incrementalRegressionGP(X,Y,options); % X has two predictor columns.
```

Kernel hyperparameters remain fixed by default. Set
`options.LearnHyperparameters = true` to learn length scales, signal variance,
and noise variance on subsequent `fit`, `update`, or `partialFit` calls.
The initial construction batch uses the supplied parameters. Online learning
uses a joint Gaussian state for basis values and log length scales and log
signal/noise standard deviations. Sigma-point predictions are conditioned on
each batch and weighted by their predictive likelihood before matching joint
posterior moments. Learning starts with independent log-parameter variances
of 0.1; log space ensures positive parameters. `reset` retains the current
parameter values and restarts that uncertainty. Mean coefficients remain fixed,
and no training history is retained. Predictions use the current parameter
estimates and moment-matched basis posterior, rather than integrating a new
sigma-point mixture at query points.

FITRGP import is MATLAB-only and supports numeric,
noncategorical predictors, uniform observation weights, and no response
transformation. See the class help (`help incrementalRegressionGP`) for
options and restrictions.

## Examples and tests

- [`demo_incrementalRegressionGP.m`](demo_incrementalRegressionGP.m) compares
  squared-exponential, Matern 3/2, and Matern 5/2 kernels on a noisy streaming
  signal with a sharp corner and localized peak. Its six-panel dashboard
  shows predictions, 95% latent intervals, grid RMSE, uncertainty reduction,
  and pre-update batch error. An unsampled gap highlights uncertainty away
  from observations. Set `animateUpdates = false` to skip intermediate
  redraws; the final plots and summary table are still produced.

Run the numerical and API tests with:

```matlab
results = runtests('test_incrementalRegressionGP.m');
assertSuccess(results)
```

The optional generated-C/MEX test in `test_incrementalRegressionGP_codegen.m`
requires MATLAB Coder and a configured C compiler. It checks both fixed-
parameter updates and hyperparameter learning for all supported kernels,
with one and two predictor columns.

## Reference

M. F. Huber, "Recursive Gaussian process: On-line regression and learning,"
*Pattern Recognition Letters*, vol. 45, pp. 85-91, 2014.
