classdef incrementalRegressionGP
    %incrementalRegressionGP Incremental Gaussian process regression.
    %   Mdl = incrementalRegressionGP creates an untrained model. Use
    %   Mdl = fit(Mdl,X,Y) to select basis vectors and fit the first batch.
    %
    %   Mdl = incrementalRegressionGP(X,Y) selects the rows of X as fixed
    %   basis vectors and conditions their latent Gaussian distribution on Y.
    %   X is an N-by-P real numeric predictor matrix. Y is an N-element real
    %   numeric response vector. Rows are observations; columns are predictors.
    %
    %   Mdl = incrementalRegressionGP(X,Y,Options) specifies a scalar structure:
    %     BasisVectors    - M-by-P fixed basis matrix (default: X).
    %     KernelFunction  - 'squaredexponential' (default), 'matern32',
    %                       'matern52', or the corresponding 'ard' prefix.
    %     LengthScale     - Positive scalar or P-element vector (default: 1).
    %                       A vector specifies one scale per predictor. For
    %                       ARD kernels, a scalar is expanded to P scales.
    %     SignalVariance  - Positive kernel amplitude variance (default: 1).
    %     NoiseVariance   - Positive observation noise variance (default: 1e-3).
    %     Jitter          - Positive relative basis-kernel nugget (default: 1e-10).
    %     BasisFunction   - 'none', 'constant', 'linear', or 'purequadratic'.
    %                       Default: 'none'.
    %     Beta            - Fixed mean-function coefficients (default: []).
    %     LearnHyperparameters - Logical scalar enabling online sigma-point
    %                       learning during FIT (default: false).
    %
    %   Mdl = incrementalRegressionGP(InitialMdl) imports a full RegressionGP
    %   model from FITRGP. Supported models use exact fitting and prediction,
    %   squared-exponential, Matern 3/2, or Matern 5/2 kernels (including ARD),
    %   numeric noncategorical predictors, and no response transformation. Fitted
    %   kernel parameters, mean coefficients, and standardization are retained.
    %   Conversion is performed in MATLAB, not in generated code.
    %
    %   Mdl = fit(Mdl,Xnew,Ynew) updates the posterior on the fixed basis.
    %   It does not append training data. Hyperparameters remain fixed unless
    %   LearnHyperparameters is true. Construction conditions the initial
    %   batch using the supplied parameters; subsequent FIT calls learn them.
    %   Learning conditions log-space sigma points on each batch and weights
    %   them by predictive likelihood. Initial log-parameter variances are
    %   0.1. Mean coefficients and basis locations remain fixed.
    %   Mdl = initialize(Mdl,X,Y,Options) replaces the model with a new fit.
    %   Mdl = reset(Mdl) restores the prior while keeping the configuration.
    %
    %   [YPred,YSD,YInt] = predict(Mdl,Xnew) returns predicted responses,
    %   response standard deviations, and 95% prediction intervals. Specify
    %   struct('Alpha',A,'IncludeObservationNoise',false) for (1-A)*100%
    %   pointwise intervals on the latent function instead of noisy responses.
    %
    %   This is a value class. Assign the output of FIT, UPDATE, PARTIALFIT,
    %   RESET, INITIALIZE, and metric-update methods to retain changes.
    %   State storage is O(M^2 + M*P), independent of the number of updates.
    %   Batch updates use the joint conditional covariance within the batch.
    %   Outside the basis, splitting a batch can change the RGP approximation.
    %
    %   Numeric construction, fitting, prediction, and metric methods support
    %   MATLAB Coder. Create the model inside a code-generation entry-point
    %   function, with fixed configuration and bounded input dimensions.
    %
    %   Example:
    %     x = linspace(0,1,20).';
    %     Mdl = incrementalRegressionGP(x,sin(6*x), ...
    %         struct('LengthScale',0.3,'NoiseVariance',0.0025));
    %     Mdl = fit(Mdl,0.25,sin(1.5));
    %     [y,sd,interval] = predict(Mdl,[0.2;0.4]);
    %
    %   Reference:
    %   Huber, M. F. (2014). Recursive Gaussian process: On-line regression
    %   and learning. Pattern Recognition Letters, 45, 85-91. This class
    %   implements fixed-hyperparameter RGP (Section 3), with optional
    %   sigma-point Bayesian hyperparameter learning inspired by Section 4.
    %
    %   See also fitrgp, incrementalRegressionKernel, incrementalRegressionLinear.

    properties (SetAccess = private)
        %X Fixed basis vectors, in original predictor coordinates.
        X = zeros(0,0);
        %LengthScale Kernel length scales, in standardized coordinates.
        LengthScale = 1;
        %KernelFunction Covariance kernel name, including an optional ARD prefix.
        KernelFunction = 'squaredexponential';
        %SignalVariance Variance of the latent covariance kernel.
        SignalVariance = 1;
        %NoiseVariance Variance of independent observation noise.
        NoiseVariance = 1e-3;
        %LearnHyperparameters True to learn kernel and noise parameters online.
        LearnHyperparameters = false;
        %Jitter Relative diagonal regularization of the basis kernel.
        Jitter = 1e-10;
        %BasisFunction Fixed parametric mean function.
        BasisFunction = 'none';
        %Beta Coefficients of the fixed parametric mean.
        Beta = zeros(0,1);
        %PredictorLocation Predictor centering offsets.
        PredictorLocation = zeros(1,0);
        %PredictorScale Predictor scaling factors.
        PredictorScale = zeros(1,0);
        %NumPredictors Number of predictor columns.
        NumPredictors = 0;
        %NumTrainingObservations Number of responses assimilated since reset.
        NumTrainingObservations = 0;
        %IsInitialized True when the fixed basis and prior are available.
        IsInitialized = false;
        %mu_g Posterior mean of the latent function at the basis vectors.
        mu_g = zeros(0,1);
        %C_g Posterior covariance of the latent function at the basis vectors.
        C_g = zeros(0,0);
        %Metrics Pre-update prediction error statistics.
        Metrics = struct('MeanSquaredError',NaN,'NumObservations',0);
    end

    properties (Access = private)
        %BasisInputs Fixed basis in standardized kernel coordinates.
        BasisInputs = zeros(0,0);
        %BasisMean Fixed parametric mean evaluated at the basis.
        BasisMean = zeros(0,1);
        %Kx Prior basis covariance, including relative diagonal jitter.
        Kx = zeros(0,0);
        %KernelFactor Lower Cholesky factor of Kx for covariance solves.
        KernelFactor = zeros(0,0);
        %HyperparameterMean Mean of log length scales and log standard deviations.
        HyperparameterMean = zeros(0,1);
        %HyperparameterCovariance Posterior covariance in log-parameter space.
        HyperparameterCovariance = zeros(0,0);
        %BasisHyperparameterCovariance Cross-covariance of basis and log parameters.
        BasisHyperparameterCovariance = zeros(0,0);
        %MetricSquaredError Sum of squared errors used for cumulative MSE.
        MetricSquaredError = 0;
    end

    methods
        function obj = incrementalRegressionGP(X,Y,options)
            % The fixed basis is selected at construction, not at class loading.
            matrix = zeros(0,0);
            column = zeros(0,1);
            row = zeros(1,0);
            scales = 1;
            basisName = 'none';
            kernelName = 'squaredexponential';
            % Variable-size seeds let generated code accommodate the selected
            % basis size, predictor count, and configuration-name lengths.
            coder.varsize('matrix',[Inf Inf],[true true]);
            coder.varsize('column',[Inf 1],[true false]);
            coder.varsize('row','scales','basisName','kernelName',[1 Inf],[false true]);
            obj.X = matrix;
            obj.BasisInputs = matrix;
            obj.C_g = matrix;
            obj.Kx = matrix;
            obj.KernelFactor = matrix;
            obj.HyperparameterCovariance = matrix;
            obj.BasisHyperparameterCovariance = matrix;
            obj.mu_g = column;
            obj.BasisMean = column;
            obj.Beta = column;
            obj.HyperparameterMean = column;
            obj.LengthScale = scales;
            obj.PredictorLocation = row;
            obj.PredictorScale = row;
            obj.BasisFunction = basisName;
            obj.KernelFunction = kernelName;
            if nargin == 0
                return
            end
            if nargin == 1
                obj = obj.importFitrgp(X);
                return
            end
            if nargin < 3
                options = struct();
            end
            obj = initialize(obj,X,Y,options);
        end

        function obj = initialize(~,X,Y,options)
            %INITIALIZE Replace the model with a fit to a new training batch.
            %   Mdl = initialize(Mdl,X,Y) uses default parameters and X as the
            %   basis. Options has the same fields as the constructor.
            if nargin < 4
                options = struct();
            end
            obj = incrementalRegressionGP();
            obj = obj.configure(options);
            X = obj.validateX(X,0,false);
            Y = obj.validateY(Y,size(X,1));
            obj.NumPredictors = size(X,2);
            % Numeric construction uses original coordinates. Imported FITRGP
            % models can instead supply learned centering and scaling.
            obj.PredictorLocation = zeros(1,obj.NumPredictors);
            obj.PredictorScale = ones(1,obj.NumPredictors);
            basis = X;
            if isfield(options,'BasisVectors')
                basis = obj.validateX(options.BasisVectors,obj.NumPredictors,false);
            end
            obj = obj.prepareBasis(basis);
            % Condition the first batch at the supplied hyperparameters;
            % optional hyperparameter learning begins on subsequent FIT calls.
            obj = obj.assimilate(X,Y);
        end

        function obj = fit(obj,X,Y)
            %FIT Assimilate observations using fixed-basis recursive GP.
            %   Mdl = fit(Mdl,X,Y) updates the mean and covariance at the
            %   existing basis vectors. The first fit of an untrained model
            %   initializes the basis from X. Empty batches leave a trained
            %   model unchanged. Mean coefficients stay fixed. If
            %   LearnHyperparameters is true, learn length scales, signal
            %   variance, and noise variance jointly with the basis values.
            if ~obj.IsInitialized
                obj = initialize(obj,X,Y);
                return
            end
            X = obj.validateX(X,obj.NumPredictors,true);
            Y = obj.validateY(Y,size(X,1));
            if isempty(Y)
                return
            end
            if obj.LearnHyperparameters
                obj = obj.learn(X,Y);
            else
                obj = obj.assimilate(X,Y);
            end
        end

        function obj = update(obj,X,Y)
            %UPDATE Alias for FIT; assign the returned model to retain changes.
            obj = fit(obj,X,Y);
        end

        function obj = partialFit(obj,X,Y)
            %PARTIALFIT Alias for FIT for incremental training workflows.
            obj = fit(obj,X,Y);
        end

        function [ypred,ysd,yint] = predict(obj,X,options)
            %PREDICT Predict responses, standard deviations, and intervals.
            %   YPred = predict(Mdl,X) returns an N-by-1 vector.
            %   [YPred,YSD,YInt] = predict(Mdl,X,Options) also returns N-by-1
            %   standard deviations and N-by-2 pointwise prediction intervals.
            %   Options.Alpha is in (0,1), with default 0.05.
            %   Options.IncludeObservationNoise defaults to true. Set it to
            %   false to obtain latent-function uncertainty instead.
            %   With hyperparameter learning, use the current parameter
            %   estimates and the moment-matched basis posterior.
            if nargin < 3
                options = struct();
            end
            obj.requireInitialized();
            obj.validateOptions(options,{'Alpha','IncludeObservationNoise'});
            alpha = 0.05;
            includeNoise = true;
            if isfield(options,'Alpha')
                alpha = options.Alpha;
            end
            if isfield(options,'IncludeObservationNoise')
                includeNoise = options.IncludeObservationNoise;
            end
            validateattributes(alpha,{'numeric'},{'real','finite','scalar','>',0,'<',1});
            validateattributes(includeNoise,{'logical'},{'scalar'});
            X = obj.validateX(X,obj.NumPredictors,true);
            inputs = obj.standardize(X);
            cross = obj.kernel(inputs,obj.BasisInputs);
            % J = K(query,basis)/Kx. Two triangular solves avoid explicitly
            % forming the inverse. Project deviations from the fixed mean.
            J = (cross/obj.KernelFactor.')/obj.KernelFactor;
            ypred = obj.meanFunction(inputs) + J*(obj.mu_g-obj.BasisMean);
            ysd = zeros(size(X,1),1);
            yint = zeros(size(X,1),2);
            if nargout > 1
                % Only the diagonal is needed; avoid an N-by-N test covariance.
                % Total latent variance combines the conditional GP residual
                % with projected uncertainty in the posterior basis values.
                variance = obj.SignalVariance-sum(J.*cross,2) + sum((J*obj.C_g).*J,2);
                if any(variance < -1e-8*obj.SignalVariance)
                    error('incrementalRegressionGP:InvalidCovariance', ...
                        'The posterior produced a negative latent variance.');
                end
                % Clip roundoff-sized negative values only; larger negatives
                % indicate an invalid covariance and raise the error above.
                variance = max(variance,0);
                if includeNoise
                    variance = variance + obj.NoiseVariance;
                end
                ysd = sqrt(variance);
                % ERFCINV gives the two-sided standard-normal quantile without
                % requiring NORMINV in the generated-code prediction path.
                width = sqrt(2)*erfcinv(double(alpha))*ysd;
                yint = [ypred-width,ypred+width];
            end
        end

        function obj = reset(obj)
            %RESET Restore the prior and clear training and metric counts.
            %   Mdl = reset(Mdl) preserves basis vectors, hyperparameters,
            %   standardization, and the mean function. It does not refit data.
            %   Online learning restarts with independent log-parameter
            %   variances of 0.1 at the current parameter values.
            if obj.IsInitialized
                obj.mu_g = obj.BasisMean;
                obj.C_g = obj.Kx;
                if obj.LearnHyperparameters
                    % Amplitude and noise are stored as variances publicly,
                    % but their standard deviations are represented in log space.
                    obj.HyperparameterMean = log([obj.LengthScale(:); ...
                        sqrt(obj.SignalVariance);sqrt(obj.NoiseVariance)]);
                    count = numel(obj.HyperparameterMean);
                    obj.HyperparameterCovariance = 0.1*eye(count);
                    obj.BasisHyperparameterCovariance = zeros(size(obj.X,1),count);
                end
            end
            obj.NumTrainingObservations = 0;
            obj.MetricSquaredError = 0;
            obj.Metrics = struct('MeanSquaredError',NaN,'NumObservations',0);
        end

        function values = perObservationLoss(obj,X,Y)
            %PEROBSERVATIONLOSS Return squared prediction errors for each row.
            %   E = perObservationLoss(Mdl,X,Y) returns an N-by-1 vector.
            prediction = predict(obj,X);
            Y = obj.validateY(Y,numel(prediction));
            values = (Y-prediction).^2;
        end

        function value = loss(obj,X,Y)
            %LOSS Return mean squared prediction error for a nonempty batch.
            values = perObservationLoss(obj,X,Y);
            if isempty(values)
                error('incrementalRegressionGP:EmptyLoss', ...
                    'A mean squared error requires at least one observation.');
            end
            value = mean(values);
        end

        function obj = updateMetrics(obj,X,Y)
            %UPDATEMETRICS Accumulate MSE without fitting the observations.
            %   Metrics contains MeanSquaredError and NumObservations since
            %   the last RESET. There is no warmup period or sliding window.
            values = perObservationLoss(obj,X,Y);
            if isempty(values)
                return
            end
            % Accumulate sums and counts rather than averaging batch MSEs,
            % so observations are equally weighted across unequal batch sizes.
            obj.MetricSquaredError = obj.MetricSquaredError + sum(values);
            obj.Metrics.NumObservations = obj.Metrics.NumObservations + numel(values);
            obj.Metrics.MeanSquaredError = obj.MetricSquaredError/obj.Metrics.NumObservations;
        end

        function obj = updateMetricsAndFit(obj,X,Y)
            %UPDATEMETRICSANDFIT Evaluate predictions, then fit the same batch.
            %   An initialized model is required so the errors measure
            %   predictions made before assimilating the new observations.
            obj = updateMetrics(obj,X,Y);
            obj = fit(obj,X,Y);
        end
    end

    methods (Access = private)
        function obj = learn(obj,X,Y)
            %LEARN Condition a sigma-point mixture and match joint moments.
            %   Parameters are log length scales and log signal/noise standard
            %   deviations. Positive sigma-point weights are reweighted by
            %   the batch predictive likelihood, including observation noise.
            inputs = obj.standardize(X);
            count = numel(obj.HyperparameterMean);
            numBasis = size(obj.X,1);
            numPoints = 2*count+1;
            L = obj.factor(obj.HyperparameterCovariance);
            % Symmetric sigma points reproduce the log-parameter mean and
            % covariance with positive prior weights summing to one.
            offsets = sqrt(2*count)*L;
            points = [obj.HyperparameterMean, ...
                obj.HyperparameterMean+offsets,obj.HyperparameterMean-offsets];
            weights = [0.5;ones(2*count,1)/(4*count)];
            % Gaussian conditioning expresses basis values as a linear
            % function of log parameters plus a shared residual covariance.
            slope = (obj.BasisHyperparameterCovariance/L.')/L;
            conditional = obj.C_g-slope*obj.BasisHyperparameterCovariance.';
            conditional = (conditional+conditional.')/2;
            means = zeros(numBasis+count,numPoints);
            covariances = zeros(numBasis,numBasis,numPoints);
            logWeights = zeros(numPoints,1);
            for k = 1:numPoints
                % Each candidate has its own kernel and observation noise,
                % while retaining the same fixed mean and basis locations.
                candidate = obj.setHyperparameters(points(:,k));
                [J,B] = candidate.project(inputs);
                basisMean = obj.mu_g+slope*(points(:,k)-obj.HyperparameterMean);
                predicted = obj.meanFunction(inputs)+J*(basisMean-obj.BasisMean);
                cross = conditional*J.';
                residual = B+candidate.NoiseVariance*eye(numel(Y));
                innovation = residual+J*cross;
                factor = obj.factor(innovation);
                % Condition the basis distribution on the observed targets.
                % The candidate log parameters stay fixed within this component.
                gain = (cross/factor.')/factor;
                errorVector = Y-predicted;
                means(:,k) = [basisMean+gain*errorVector;points(:,k)];
                transition = eye(numBasis)-gain*J;
                covariance = transition*conditional*transition.'+gain*residual*gain.';
                covariances(:,:,k) = (covariance+covariance.')/2;
                standardizedError = factor\errorVector;
                % Log likelihood uses the Cholesky log determinant and
                % squared whitened residual. Common normalization terms cancel.
                logWeights(k) = log(weights(k))-sum(log(diag(factor))) ...
                    -0.5*sum(standardizedError.^2);
            end
            if any(~isfinite(logWeights))
                error('incrementalRegressionGP:InvalidLearningState', ...
                    'The hyperparameter predictive likelihood is not finite.');
            end
            % Subtract the largest log weight to avoid numerical underflow
            % when normalizing the candidate predictive likelihoods.
            weights = exp(logWeights-max(logWeights));
            weights = weights/sum(weights);
            jointMean = means*weights;
            jointCovariance = zeros(numBasis+count);
            for k = 1:numPoints
                % Total covariance includes between-component mean variation
                % and within-component basis uncertainty. This also recovers
                % the updated basis/log-parameter cross-covariance.
                difference = means(:,k)-jointMean;
                jointCovariance = jointCovariance+weights(k)*(difference*difference.');
                jointCovariance(1:numBasis,1:numBasis) = ...
                    jointCovariance(1:numBasis,1:numBasis)+weights(k)*covariances(:,:,k);
            end
            % A small log-space nugget prevents collapse of the sigma points.
            jointCovariance(numBasis+1:end,numBasis+1:end) = ...
                jointCovariance(numBasis+1:end,numBasis+1:end)+1e-10*eye(count);
            jointCovariance = (jointCovariance+jointCovariance.')/2;
            obj.mu_g = jointMean(1:numBasis);
            obj.C_g = jointCovariance(1:numBasis,1:numBasis);
            obj.HyperparameterMean = jointMean(numBasis+1:end);
            obj.HyperparameterCovariance = jointCovariance(numBasis+1:end,numBasis+1:end);
            obj.BasisHyperparameterCovariance = jointCovariance(1:numBasis,numBasis+1:end);
            obj = obj.setHyperparameters(obj.HyperparameterMean);
            obj.NumTrainingObservations = obj.NumTrainingObservations+numel(Y);
        end

        function obj = setHyperparameters(obj,parameters)
            %SETHYPERPARAMETERS Refresh the kernel without resetting the basis.
            %   PARAMETERS contains log length scales, log signal standard
            %   deviation, and log noise standard deviation, in that order.
            count = numel(obj.LengthScale);
            obj.LengthScale = reshape(exp(parameters(1:count)),1,[]);
            obj.SignalVariance = exp(2*parameters(count+1));
            obj.NoiseVariance = exp(2*parameters(count+2));
            if any(~isfinite(obj.LengthScale)) || any(obj.LengthScale <= 0) || ...
                    ~isfinite(obj.SignalVariance) || obj.SignalVariance <= 0 || ...
                    ~isfinite(obj.NoiseVariance) || obj.NoiseVariance <= 0
                error('incrementalRegressionGP:InvalidLearningState', ...
                    'Learned hyperparameters must be finite and strictly positive.');
            end
            obj.Kx = obj.kernel(obj.BasisInputs,obj.BasisInputs)+ ...
                obj.Jitter*obj.SignalVariance*eye(size(obj.X,1));
            obj.KernelFactor = obj.factor(obj.Kx);
        end

        function obj = assimilate(obj,X,Y)
            %ASSIMILATE Condition the basis posterior at fixed hyperparameters.
            %   J projects basis deviations onto the new observations. B is
            %   the remaining conditional function covariance within the batch.
            inputs = obj.standardize(X);
            [J,B] = obj.project(inputs);
            predicted = obj.meanFunction(inputs) + J*(obj.mu_g-obj.BasisMean);
            crossCovariance = obj.C_g*J.';
            % Include both conditional GP uncertainty and measurement noise;
            % off-diagonal entries retain dependence between batch observations.
            residual = B + obj.NoiseVariance*eye(size(X,1));
            innovation = residual + J*crossCovariance;
            L = obj.factor(innovation);
            gain = (crossCovariance/L.')/L;
            updatedMean = obj.mu_g + gain*(Y-predicted);

            % Joseph form avoids subtracting two nearly equal covariances.
            transition = eye(size(obj.X,1))-gain*J;
            covariance = transition*obj.C_g*transition.' + gain*residual*gain.';
            obj.mu_g = updatedMean;
            obj.C_g = (covariance+covariance.')/2;
            obj.NumTrainingObservations = obj.NumTrainingObservations + numel(Y);
        end

        function obj = configure(obj,options)
            %CONFIGURE Validate supplied options and retain unspecified defaults.
            obj.validateOptions(options,{'BasisVectors','LengthScale', ...
                'SignalVariance','NoiseVariance','Jitter','BasisFunction','Beta', ...
                'LearnHyperparameters','KernelFunction'});
            if isfield(options,'KernelFunction')
                name = options.KernelFunction;
                if ~(ischar(name) && isrow(name)) && ~(isstring(name) && isscalar(name))
                    error('incrementalRegressionGP:UnsupportedKernel', ...
                        'KernelFunction must be a character vector or string scalar.');
                end
                obj.KernelFunction = lower(char(name));
            end
            if ~any(strcmp(obj.KernelFunction,{'squaredexponential', ...
                    'ardsquaredexponential','matern32','ardmatern32', ...
                    'matern52','ardmatern52'}))
                error('incrementalRegressionGP:UnsupportedKernel', ...
                    ['KernelFunction must be squaredexponential, matern32, or ' ...
                    'matern52, optionally prefixed with ard.']);
            end
            if isfield(options,'LearnHyperparameters')
                validateattributes(options.LearnHyperparameters,{'logical'},{'scalar'});
                obj.LearnHyperparameters = options.LearnHyperparameters;
            end
            if isfield(options,'LengthScale')
                validateattributes(options.LengthScale,{'numeric'}, ...
                    {'real','finite','positive','vector','nonempty','nonsparse'});
                obj.LengthScale = reshape(double(options.LengthScale),1,[]);
            end
            if isfield(options,'SignalVariance')
                validateattributes(options.SignalVariance,{'numeric'}, ...
                    {'real','finite','positive','scalar'});
                obj.SignalVariance = double(options.SignalVariance);
            end
            if isfield(options,'NoiseVariance')
                validateattributes(options.NoiseVariance,{'numeric'}, ...
                    {'real','finite','positive','scalar'});
                obj.NoiseVariance = double(options.NoiseVariance);
            end
            if isfield(options,'Jitter')
                validateattributes(options.Jitter,{'numeric'}, ...
                    {'real','finite','positive','scalar'});
                obj.Jitter = double(options.Jitter);
            end
            if isfield(options,'BasisFunction')
                name = options.BasisFunction;
                if ~(ischar(name) && isrow(name)) && ~(isstring(name) && isscalar(name))
                    error('incrementalRegressionGP:UnsupportedBasis', ...
                        'BasisFunction must be a character vector or string scalar.');
                end
                obj.BasisFunction = lower(char(options.BasisFunction));
            end
            if isfield(options,'Beta')
                validateattributes(options.Beta,{'numeric'},{'real','finite','2d','nonsparse'});
                if ~isempty(options.Beta) && ~isvector(options.Beta)
                    error('incrementalRegressionGP:BetaSize', ...
                        'Beta must be a coefficient vector.');
                end
                obj.Beta = double(options.Beta(:));
            end
            validateattributes(obj.LengthScale,{'double'},{'real','finite','positive','vector','nonempty'});
            validateattributes(obj.SignalVariance,{'double'},{'real','finite','positive','scalar'});
            validateattributes(obj.NoiseVariance,{'double'},{'real','finite','positive','scalar'});
            validateattributes(obj.Jitter,{'double'},{'real','finite','positive','scalar'});
            if ~any(strcmp(obj.BasisFunction,{'none','constant','linear','purequadratic'}))
                error('incrementalRegressionGP:UnsupportedBasis', ...
                    'BasisFunction must be none, constant, linear, or purequadratic.');
            end
        end

        function obj = prepareBasis(obj,basis)
            %PREPAREBASIS Build the fixed prior and initialize posterior state.
            if numel(obj.LengthScale) ~= 1 && numel(obj.LengthScale) ~= obj.NumPredictors
                error('incrementalRegressionGP:LengthScaleSize', ...
                    'LengthScale must be scalar or have one element per predictor.');
            end
            obj.LengthScale = reshape(obj.LengthScale,1,[]);
            % Explicit ARD gives each predictor its own learnable parameter,
            % even when the user supplies a common initial length scale.
            if strncmp(obj.KernelFunction,'ard',3) && isscalar(obj.LengthScale)
                obj.LengthScale = repmat(obj.LengthScale,1,obj.NumPredictors);
            end
            expected = 0;
            if strcmp(obj.BasisFunction,'constant')
                expected = 1;
            elseif strcmp(obj.BasisFunction,'linear')
                expected = obj.NumPredictors+1;
            elseif strcmp(obj.BasisFunction,'purequadratic')
                % Pure quadratic uses intercept, linear, and squared terms;
                % it excludes interactions between different predictors.
                expected = 2*obj.NumPredictors+1;
            end
            if numel(obj.Beta) ~= expected
                error('incrementalRegressionGP:BetaSize', ...
                    'Beta must contain the coefficients required by BasisFunction.');
            end
            obj.X = basis;
            obj.BasisInputs = obj.standardize(basis);
            obj.BasisMean = obj.meanFunction(obj.BasisInputs);
            % Relative jitter stabilizes the basis factorization, including
            % duplicate basis points. It is not additional observation noise.
            obj.Kx = obj.kernel(obj.BasisInputs,obj.BasisInputs) + ...
                obj.Jitter*obj.SignalVariance*eye(size(basis,1));
            obj.KernelFactor = obj.factor(obj.Kx);
            obj.IsInitialized = true;
            obj = reset(obj);
        end

        function inputs = standardize(obj,X)
            %STANDARDIZE Apply the same coordinate transform to all inputs.
            inputs = (X-obj.PredictorLocation)./obj.PredictorScale;
        end

        function K = kernel(obj,A,B)
            %KERNEL Evaluate the covariance using scaled Euclidean distances.
            %   Scalar LengthScale is shared; a vector scales each predictor
            %   separately. All supported kernels have diagonal SignalVariance.
            %   For scaled distance d, Matern 3/2 uses r = sqrt(3)*d and
            %   Matern 5/2 uses r = sqrt(5)*d in their exponential polynomials.
            K = zeros(size(A,1),size(B,1));
            for i = 1:size(A,1)
                for j = 1:size(B,1)
                    distance = (A(i,:)-B(j,:))./obj.LengthScale;
                    squaredDistance = sum(distance.^2);
                    % Avoid Inf*0 when scaled distances overflow.
                    if squaredDistance > realmax/5
                        value = 0;
                    elseif any(strcmp(obj.KernelFunction,{'matern32','ardmatern32'}))
                        r = sqrt(3*squaredDistance);
                        value = (1+r)*exp(-r);
                    elseif any(strcmp(obj.KernelFunction,{'matern52','ardmatern52'}))
                        r = sqrt(5*squaredDistance);
                        value = (1+r+r^2/3)*exp(-r);
                    else
                        value = exp(-0.5*squaredDistance);
                    end
                    K(i,j) = obj.SignalVariance*value;
                end
            end
        end

        function value = meanFunction(obj,X)
            %MEANFUNCTION Evaluate the fixed trend in standardized coordinates.
            %   BasisFunction specifies a parametric mean, not the GP kernel
            %   or the fixed basis vectors used by the recursive posterior.
            value = zeros(size(X,1),1);
            if strcmp(obj.BasisFunction,'constant')
                value(:) = obj.Beta(1);
            elseif strcmp(obj.BasisFunction,'linear')
                value = [ones(size(X,1),1),X]*obj.Beta;
            elseif strcmp(obj.BasisFunction,'purequadratic')
                value = [ones(size(X,1),1),X,X.^2]*obj.Beta;
            end
        end

        function [J,B] = project(obj,inputs)
            %PROJECT Compute the conditional GP mapping, p(g_new | g_basis).
            %   Huber (2014): conditional mean deviations are J times basis
            %   deviations, and B is uncertainty not explained by the basis.
            %   B retains the full batch covariance, not just its diagonal.
            cross = obj.kernel(inputs,obj.BasisInputs);
            J = (cross/obj.KernelFactor.')/obj.KernelFactor;
            B = obj.kernel(inputs,inputs)-J*cross.';
            B = (B+B.')/2;
        end

        function obj = importFitrgp(obj,model)
            %IMPORTFITRGP Reconstruct an exact fitted GP on its training basis.
            %   Keep fitted kernel, noise, mean, and coordinate-transform
            %   parameters fixed; assimilate original targets under that prior.
            if ~isa(model,'RegressionGP')
                error('incrementalRegressionGP:InvalidInitialModel', ...
                    'Supply a full RegressionGP model returned by fitrgp.');
            end
            if ~strcmpi(model.FitMethod,'exact') || ~strcmpi(model.PredictMethod,'exact')
                error('incrementalRegressionGP:UnsupportedInitialModel', ...
                    'The initial model must use exact fitting and prediction.');
            end
            if ~isnumeric(model.X) || ~isempty(model.CategoricalPredictors) || ...
                    ~strcmpi(model.ResponseTransform,'none')
                error('incrementalRegressionGP:UnsupportedInitialModel', ...
                    'Use numeric noncategorical predictors and no response transformation.');
            end
            information = model.KernelInformation;
            if ~any(strcmpi(information.Name,{'SquaredExponential','ARDSquaredExponential', ...
                    'Matern32','ARDMatern32','Matern52','ARDMatern52'}))
                error('incrementalRegressionGP:UnsupportedKernel', ...
                    ['Only squared-exponential, Matern 3/2, and Matern 5/2 ' ...
                    'kernels, including ARD variants, are supported.']);
            end
            trainingInputs = obj.validateX(model.X,0,false);
            trainingResponses = obj.validateY(model.Y,size(trainingInputs,1));
            % Nonuniform observation weights require a different noise model.
            if any(abs(model.W-model.W(1)) > 1e-12)
                error('incrementalRegressionGP:UnsupportedWeights', ...
                    'The initial model must have uniform observation weights.');
            end
            parameters = information.KernelParameters;
            % FITRGP stores signal and noise standard deviations. Convert
            % them to the variance convention used by this class.
            options = struct('LengthScale',parameters(1:end-1), ...
                'KernelFunction',lower(information.Name), ...
                'SignalVariance',parameters(end)^2,'NoiseVariance',model.Sigma^2, ...
                'BasisFunction',lower(model.BasisFunction));
            if ~isempty(model.Beta)
                options.Beta = model.Beta;
            end
            obj = obj.configure(options);
            obj.NumPredictors = size(trainingInputs,2);
            obj.PredictorLocation = zeros(1,obj.NumPredictors);
            obj.PredictorScale = ones(1,obj.NumPredictors);
            if ~isempty(model.PredictorLocation)
                obj.PredictorLocation = reshape(model.PredictorLocation,1,[]);
                obj.PredictorScale = reshape(model.PredictorScale,1,[]);
                validateattributes(obj.PredictorLocation,{'numeric'},{'real','finite'});
                validateattributes(obj.PredictorScale,{'numeric'},{'real','finite','positive'});
            end
            % Using all training inputs as basis points reproduces the exact
            % posterior to numerical tolerance without retaining a separate
            % response history for future recursive updates.
            obj = obj.prepareBasis(trainingInputs);
            obj = obj.assimilate(trainingInputs,trainingResponses);
        end

        function requireInitialized(obj)
            if ~obj.IsInitialized
                error('incrementalRegressionGP:NotInitialized', ...
                    'Fit the model before predicting or computing metrics.');
            end
        end
    end

    methods (Static, Access = private)
        function X = validateX(X,p,allowEmpty)
            %VALIDATEX Normalize dense numeric predictors and enforce row width.
            validateattributes(X,{'numeric'},{'real','finite','2d','nonsparse'});
            X = double(X);
            if isempty(X)
                if ~allowEmpty
                    error('incrementalRegressionGP:EmptyData', ...
                        'The initial training data and basis vectors must be nonempty.');
                end
                if size(X,2) ~= p && size(X,2) ~= 0
                    error('incrementalRegressionGP:PredictorSize', ...
                        'X must have one column per predictor.');
                end
                if size(X,2) == 0
                    % Accept [] as an empty batch while preserving the known
                    % predictor dimension for subsequent matrix operations.
                    X = zeros(0,p);
                end
                return
            end
            if p > 0 && size(X,2) ~= p
                error('incrementalRegressionGP:PredictorSize', ...
                    'X must have one column per predictor.');
            end
        end

        function Y = validateY(Y,n)
            %VALIDATEY Accept row or column responses and return a double column.
            validateattributes(Y,{'numeric'},{'real','finite','2d','nonsparse'});
            if (~isempty(Y) && ~isvector(Y)) || numel(Y) ~= n
                error('incrementalRegressionGP:ResponseSize', ...
                    'Y must be a response vector with one element per row of X.');
            end
            Y = double(Y(:));
        end

        function validateOptions(options,allowed)
            %VALIDATEOPTIONS Reject unknown fields rather than ignoring typos.
            if ~isstruct(options) || ~isscalar(options)
                error('incrementalRegressionGP:InvalidOptions', ...
                    'Options must be a scalar structure.');
            end
            names = fieldnames(options);
            for i = 1:numel(names)
                if ~any(strcmp(names{i},allowed))
                    error('incrementalRegressionGP:UnknownOption', ...
                        'Options contains an unrecognized field.');
                end
            end
        end

        function L = factor(A)
            %FACTOR Obtain a lower Cholesky factor or report invalid covariance.
            %   Do not add adaptive jitter here: silently changing A would
            %   change the uncertainty model used by the caller.
            [L,status] = chol(A,'lower');
            if status ~= 0
                error('incrementalRegressionGP:NotPositiveDefinite', ...
                    'Covariance factorization failed. Check the basis, length scales, and noise variance.');
            end
        end
    end
end
