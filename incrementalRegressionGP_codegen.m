function [ypred,ysd,yint,meanSquaredError] = incrementalRegressionGP_codegen(X,Y,Xtest,learnHyperparameters,kernelFunction)
%INCREMENTALREGRESSIONGP_CODEGEN Numeric entry point for MATLAB Coder.
%   Use one or two predictor columns. X and Y are a new training batch. Xtest contains
%   prediction locations. The model is reconstructed on each call, using a
%   fixed basis and an initial batch followed by a recursive update.
%   An optional fourth logical input enables hyperparameter learning
%   (default: false).
%   An optional fifth character-vector input selects KernelFunction
%   (default: 'squaredexponential'), including Matern and ARD kernels.
%
%   Example (requires MATLAB Coder and a configured C compiler):
%     codegen incrementalRegressionGP_codegen -args ...
%         {zeros(3,1),zeros(3,1),zeros(5,1)}
%
%   FITRGP conversion is a MATLAB-only operation. This entry point demonstrates
%   numeric construction, incremental fitting, metrics, and prediction in C.
%#codegen
if nargin < 4
    learnHyperparameters = false;
end
if nargin < 5
    kernelFunction = 'squaredexponential';
end
locations = [0;0.25;0.5;0.75;1];
if size(Xtest,2) == 2
    basis = [locations,1-locations];
    lengthScale = [0.3 0.7];
elseif size(Xtest,2) == 1
    basis = locations;
    lengthScale = 0.3;
else
    error('incrementalRegressionGP_codegen:PredictorSize', ...
        'Xtest must have one or two predictor columns.');
end
options = struct('LengthScale',lengthScale,'NoiseVariance',0.0025, ...
    'KernelFunction',kernelFunction, ...
    'LearnHyperparameters',learnHyperparameters);
model = incrementalRegressionGP(basis,sin(6*basis(:,1)),options);
model = updateMetricsAndFit(model,X,Y);
[ypred,ysd,yint] = predict(model,Xtest,struct('Alpha',0.05));
meanSquaredError = model.Metrics.MeanSquaredError;
end
