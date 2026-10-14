%% Compare Recursive Gaussian Process Kernels on a Streaming Signal
%DEMO_INCREMENTALREGRESSIONGP Visualize fixed-basis regression and uncertainty.
%   Compare squared-exponential, Matern 3/2, and Matern 5/2 kernels as noisy
%   observations arrive from left to right. The signal combines oscillations,
%   a sharp corner, and a localized peak. An unsampled gap illustrates why
%   posterior uncertainty depends on observation coverage.
%
%   The dashboard shows latent-function 95% pointwise intervals, prediction
%   accuracy on a noise-free grid, interval widths, and pre-update error on
%   incoming observations. All kernels use the same fixed hyperparameters
%   and basis locations. The demo retains data for plotting; the models do
%   not retain the stream. No figures or animations are written to disk.

%% Define a Reproducible Signal and Observation Stream
% Reuse the same random observations on every run so kernel comparisons are
% reproducible. Disable animation to redraw only the initial and final states.
rng(7);
animateUpdates = true;
% The absolute-value term creates a corner at 3.2; the Gaussian-shaped term
% creates a narrow peak at 7.2. These features challenge smoothness assumptions.
signal = @(x) 0.65*sin(1.4*x) + 0.25*sin(4.5*x) ...
    + 0.8*exp(-((x-7.2)/0.35).^2) + 0.35*abs(x-3.2) - 0.8;
noiseSD = 0.12;
gap = [4.3 5.7];
% Basis points span the entire domain, including the gap. They represent
% latent function values, not additional observations of the true signal.
% The dense test grid is used only for visualization and accuracy evaluation.
basis = linspace(0,10,45).';
xTest = linspace(0,10,501).';
yTrue = signal(xTest);

% Start with observations near the left boundary. Sorting the subsequent
% inputs simulates a sensor scanning toward the right, skipping the gap.
xInitial = linspace(0,1.8,12).';
yInitial = signal(xInitial) + noiseSD*randn(size(xInitial));
xStream = sort([1.9+(gap(1)-1.9)*rand(48,1); ...
    gap(2)+(10-gap(2))*rand(72,1)]);
yStream = signal(xStream) + noiseSD*randn(size(xStream));
batchSize = 6;
numBatches = ceil(numel(xStream)/batchSize);
% Include the initial fit at history index 1 and allow a shorter final batch.
numObservations = numel(xInitial) + min((0:numBatches)*batchSize,numel(xStream));

%% Initialize Models with Identical Fixed Hyperparameters
% Squared-exponential assumes a very smooth function. Matern 3/2 and 5/2
% permit rougher functions, with 5/2 smoother than 3/2. Equal parameters
% isolate kernel effects; they are not individually optimized for this signal.
kernelNames = {'squaredexponential','matern32','matern52'};
kernelLabels = {'Squared exponential','Matern 3/2','Matern 5/2'};
colors = [0.00 0.40 0.70;0.85 0.33 0.10;0.30 0.60 0.25];
numKernels = numel(kernelNames);
models = cell(1,numKernels);
% History rows represent successive model states; columns represent kernels.
% Pre-update errors have no initial-state entry because no stream batch has
% arrived yet. Keep the initial means separately as a visual reference.
initialMean = zeros(numel(xTest),numKernels);
gridRMSE = zeros(numBatches+1,numKernels);
meanIntervalWidth = zeros(numBatches+1,numKernels);
preUpdateRMSE = nan(numBatches,numKernels);
% Exclude observation noise to show uncertainty in the underlying function.
% These are pointwise intervals, not a simultaneous band over the whole grid.
predictionOptions = struct('Alpha',0.05,'IncludeObservationNoise',false);
for k = 1:numKernels
    % NoiseVariance takes the squared observation standard deviation.
    % Hyperparameter learning remains disabled for this controlled comparison.
    options = struct('KernelFunction',kernelNames{k}, ...
        'BasisVectors',basis,'LengthScale',0.55, ...
        'SignalVariance',1,'NoiseVariance',noiseSD^2);
    models{k} = incrementalRegressionGP(xInitial,yInitial,options);
    [initialMean(:,k),~,interval] = predict(models{k},xTest,predictionOptions);
    gridRMSE(1,k) = sqrt(mean((initialMean(:,k)-yTrue).^2));
    meanIntervalWidth(1,k) = mean(interval(:,2)-interval(:,1));
end

%% Create the Kernel Comparison Dashboard
% The upper row compares posterior functions. The lower row tracks accuracy,
% uncertainty, and out-of-sample streaming error using consistent colors.
demoFigure = figure('Color','w','Name','Streaming GP: kernel comparison', ...
    'Position',[100 100 1450 800]);
layout = tiledlayout(demoFigure,2,3,'TileSpacing','compact','Padding','compact');
posteriorAxes = gobjects(1,numKernels);
meanLines = gobjects(1,numKernels);
intervalPatches = gobjects(1,numKernels);
observationPoints = gobjects(1,numKernels);
% Retain graphics handles so each batch changes existing plot data rather
% than adding new objects and accumulating duplicate legend entries.
for k = 1:numKernels
    ax = nexttile(layout);
    posteriorAxes(k) = ax;
    hold(ax,'on');
    xregion(ax,gap(1),gap(2),'FaceColor',[0.5 0.5 0.5], ...
        'FaceAlpha',0.12,'HandleVisibility','off');
    [~,~,interval] = predict(models{k},xTest,predictionOptions);
    % Traverse the lower boundary forward and the upper boundary backward
    % to form a closed polygon for the shaded uncertainty band.
    intervalPatches(k) = fill(ax,[xTest;flipud(xTest)], ...
        [interval(:,1);flipud(interval(:,2))],colors(k,:), ...
        'FaceAlpha',0.18,'EdgeColor','none','DisplayName','95% latent interval');
    plot(ax,xTest,yTrue,'k-','LineWidth',1.3,'DisplayName','True signal');
    plot(ax,xTest,initialMean(:,k),'--','Color',[0.5 0.5 0.5], ...
        'DisplayName','Initial prediction');
    meanLines(k) = plot(ax,xTest,initialMean(:,k),'Color',colors(k,:), ...
        'LineWidth',1.8,'DisplayName','Current prediction');
    observationPoints(k) = scatter(ax,xInitial,yInitial,14, ...
        colors(k,:),'filled','DisplayName','Observed data');
    % The bottom rug marks basis locations; its height is purely decorative.
    plot(ax,basis,-2.3*ones(size(basis)),'|','Color',[0.4 0.4 0.4], ...
        'DisplayName','Fixed basis');
    xlim(ax,[0 10]);
    ylim(ax,[-2.5 3.5]);
    xlabel(ax,'Input');
    ylabel(ax,'Response');
    title(ax,kernelLabels{k});
    grid(ax,'on');
    if k == 1
        legend(ax,'Location','southoutside','NumColumns',2);
    end
end

errorAxes = nexttile(layout);
hold(errorAxes,'on');
errorLines = gobjects(1,numKernels);
widthAxes = nexttile(layout);
hold(widthAxes,'on');
widthLines = gobjects(1,numKernels);
streamAxes = nexttile(layout);
hold(streamAxes,'on');
streamLines = gobjects(1,numKernels);
for k = 1:numKernels
    errorLines(k) = plot(errorAxes,numObservations(1),gridRMSE(1,k), ...
        'Color',colors(k,:),'LineWidth',1.8,'DisplayName',kernelLabels{k});
    widthLines(k) = plot(widthAxes,numObservations(1),meanIntervalWidth(1,k), ...
        'Color',colors(k,:),'LineWidth',1.8);
    streamLines(k) = plot(streamAxes,nan,nan,'Color',colors(k,:), ...
        'LineWidth',1.3);
end
title(errorAxes,'Accuracy across the entire input range');
ylabel(errorAxes,'Noise-free grid RMSE');
legend(errorAxes,'Location','best');
title(widthAxes,'Uncertainty as observations arrive');
ylabel(widthAxes,'Mean 95% latent interval width');
title(streamAxes,'Incoming batch error before fitting');
ylabel(streamAxes,'Pre-update batch RMSE');
% Noise SD is a reference scale for noisy-target RMSE, not a strict lower
% bound on the error of any individual finite batch.
yline(streamAxes,noiseSD,'k:','Observation noise SD','HandleVisibility','off');
for ax = [errorAxes widthAxes streamAxes]
    xlabel(ax,'Observations assimilated');
    xlim(ax,[numObservations(1) numObservations(end)]);
    grid(ax,'on');
end
title(layout,sprintf('Streaming GP | %d observations | %d fixed basis points', ...
    numObservations(1),numel(basis)));
subtitle(layout,'Gray region: no observations. Dashed curves: initial predictions.');
drawnow;

%% Evaluate Each Batch Before Updating the Models
% Test each model on unseen batch targets before assimilating them. Measuring
% after fitting would instead report in-sample error and overstate accuracy.
for batch = 1:numBatches
    indices = (batch-1)*batchSize+1:min(batch*batchSize,numel(xStream));
    for k = 1:numKernels
        preUpdateRMSE(batch,k) = sqrt(loss(models{k},xStream(indices),yStream(indices)));
        % The value-class update must be assigned back. This call also
        % accumulates pre-update squared errors across all streamed responses.
        models{k} = updateMetricsAndFit(models{k},xStream(indices),yStream(indices));
        % Evaluate the updated posterior everywhere, including regions that
        % are still unobserved. The true grid values never enter model fitting.
        [mu,~,interval] = predict(models{k},xTest,predictionOptions);
        gridRMSE(batch+1,k) = sqrt(mean((mu-yTrue).^2));
        meanIntervalWidth(batch+1,k) = mean(interval(:,2)-interval(:,1));
        meanLines(k).YData = mu;
        intervalPatches(k).YData = [interval(:,1);flipud(interval(:,2))];
        % Retaining observed samples here is only for scatter-plot display;
        % each model's stored basis and posterior remain fixed in size.
        observationPoints(k).XData = [xInitial;xStream(1:indices(end))];
        observationPoints(k).YData = [yInitial;yStream(1:indices(end))];
        errorLines(k).XData = numObservations(1:batch+1);
        errorLines(k).YData = gridRMSE(1:batch+1,k);
        widthLines(k).XData = numObservations(1:batch+1);
        widthLines(k).YData = meanIntervalWidth(1:batch+1,k);
        % Locate each pre-update error at its batch's ending observation
        % count, even though that error was computed before the batch update.
        streamLines(k).XData = numObservations(2:batch+1);
        streamLines(k).YData = preUpdateRMSE(1:batch,k);
        title(posteriorAxes(k),sprintf('%s | grid RMSE %.3f', ...
            kernelLabels{k},gridRMSE(batch+1,k)));
    end
    title(layout,sprintf('Streaming GP | %d observations | %d fixed basis points', ...
        numObservations(batch+1),numel(basis)));
    if animateUpdates
        drawnow;
    end
end
drawnow;

%% Summarize Prediction Accuracy and Streaming Metrics
% Grid RMSE uses the known latent signal. Pre-update RMSE measures predictions
% on noisy incoming observations before each batch is assimilated.
% The cumulative metric excludes the initial fit and weights each streamed
% observation equally. Taking the mean of batch RMSEs would not be equivalent.
cumulativeRMSE = zeros(numKernels,1);
for k = 1:numKernels
    cumulativeRMSE(k) = sqrt(models{k}.Metrics.MeanSquaredError);
end
summary = table(string(kernelLabels(:)),gridRMSE(1,:).',gridRMSE(end,:).', ...
    meanIntervalWidth(end,:).',cumulativeRMSE, ...
    'VariableNames',{'Kernel','InitialGridRMSE','FinalGridRMSE', ...
    'FinalMeanIntervalWidth','CumulativePreUpdateRMSE'});
disp(summary);
