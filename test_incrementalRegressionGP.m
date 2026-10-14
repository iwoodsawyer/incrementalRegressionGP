function tests = test_incrementalRegressionGP
%TEST_INCREMENTALREGRESSIONGP Numerical and API regression tests.
%   Run with runtests('test_incrementalRegressionGP.m').
tests = functiontests(localfunctions);
end

function testInitialExactPosterior(testCase)
x = [0;0.5;1];
y = [1;2;-1];
opts = struct('LengthScale',0.3,'SignalVariance',2,'NoiseVariance',0.05);
m = incrementalRegressionGP(x,y,opts);
points = [-0.2;x;1.2];
[mu,sd,interval] = predict(m,points);
[expected,variance] = exactPrediction(x,y,points,opts);
verifyEqual(testCase,mu,expected,'AbsTol',1e-8);
verifyEqual(testCase,sd.^2,variance+opts.NoiseVariance,'AbsTol',1e-8);
verifySize(testCase,mu,[5 1]);
verifySize(testCase,sd,[5 1]);
verifySize(testCase,interval,[5 2]);
verifyEqual(testCase,interval,[mu-1.959963984540054*sd,mu+1.959963984540054*sd], ...
    'AbsTol',1e-10);
verifyEqual(testCase,m.NumTrainingObservations,3);
end

function testUpdateUsesTargetsAndPreservesValueSemantics(testCase)
m = incrementalRegressionGP([0;0.5;1],[1;2;-1]);
a = update(m,0.25,-100);
b = partialFit(m,0.25,100);
verifyGreaterThan(testCase,norm(predict(a,0.25)-predict(b,0.25)),1);
verifyNotEqual(testCase,a.mu_g,m.mu_g);
verifyEqual(testCase,m.NumTrainingObservations,3);
verifyEqual(testCase,a.NumTrainingObservations,4);
verifyEqual(testCase,a.X,m.X);
end

function testRepeatedBasisObservationsMatchExactGP(testCase)
x = [0;0.5;1];
y = [1;2;-1];
opts = struct('LengthScale',0.3,'SignalVariance',2,'NoiseVariance',0.05);
m = incrementalRegressionGP(x,y,opts);
newX = [0.5;0;1];
newY = [3;-2;0];
batch = fit(m,newX,newY);
for k = 1:numel(newY)
    m = fit(m,newX(k),newY(k));
end
points = [-0.1;0.2;0.7;1.1];
[expected,variance] = exactPrediction([x;newX],[y;newY],points,opts);
[mu,sd] = predict(batch,points);
[singleMu,singleSD] = predict(m,points);
verifyEqual(testCase,mu,expected,'AbsTol',1e-7);
verifyEqual(testCase,sd.^2,variance+opts.NoiseVariance,'AbsTol',1e-7);
verifyEqual(testCase,singleMu,mu,'AbsTol',1e-7);
verifyEqual(testCase,singleSD,sd,'AbsTol',1e-7);
end

function testSparseBasisMatchesRGPFormula(testCase)
for kernelName = supportedKernels()
basis = [-1;0;1];
x = [-0.7;0.2;0.6;1.3];
y = [2;-1;0.5;3];
opts = struct('BasisVectors',basis,'LengthScale',0.4, ...
    'SignalVariance',1.2,'NoiseVariance',0.03,'KernelFunction',kernelName{1});
m = incrementalRegressionGP(x,y,opts);
K = kernel(basis,basis,opts)+m.Jitter*opts.SignalVariance*eye(3);
cross = kernel(basis,x,opts);
innovation = kernel(x,x,opts)+opts.NoiseVariance*eye(4);
expectedMean = cross*(innovation\y);
expectedCovariance = K-cross*(innovation\cross.');
verifyEqual(testCase,m.mu_g,expectedMean,'AbsTol',1e-9);
verifyEqual(testCase,m.C_g,expectedCovariance,'AbsTol',1e-9);

newX = [-0.2;0.7];
newY = [4;-2];
J = kernel(newX,basis,opts)/K;
B = kernel(newX,newX,opts)-J*kernel(basis,newX,opts);
S = B+J*expectedCovariance*J.'+opts.NoiseVariance*eye(2);
G = (expectedCovariance*J.')/S;
updated = fit(m,newX,newY);
verifyEqual(testCase,updated.mu_g,expectedMean+G*(newY-J*expectedMean),'AbsTol',1e-9);
verifyEqual(testCase,updated.C_g,expectedCovariance-G*J*expectedCovariance,'AbsTol',1e-9);
verifyEqual(testCase,updated.X,basis);
end
end

function testKernelsMatchExactPosteriorAndUpdates(testCase)
x = [0 0;0.5 1;1 -0.5];
y = [1;2;-1];
points = [x;0.2 0.7;1.2 -0.3];
for kernelName = supportedKernels()
    for scales = {0.4,[0.3 1.7]}
        opts = struct('KernelFunction',kernelName{1},'LengthScale',scales{1}, ...
            'SignalVariance',2,'NoiseVariance',0.05);
        m = incrementalRegressionGP(x,y,opts);
        [expected,variance] = exactPrediction(x,y,points,opts);
        [mu,sd,interval] = predict(m,points);
        verifyEqual(testCase,m.KernelFunction,kernelName{1});
        verifyEqual(testCase,mu,expected,'AbsTol',1e-8);
        verifyEqual(testCase,sd.^2,variance+opts.NoiseVariance,'AbsTol',1e-8);
        verifyEqual(testCase,interval, ...
            [mu-1.959963984540054*sd,mu+1.959963984540054*sd],'AbsTol',1e-10);
        if startsWith(kernelName{1},'ard') && isscalar(scales{1})
            verifyEqual(testCase,m.LengthScale,[scales{1} scales{1}]);
        end
        newX = x([2 1],:);
        newY = [3;-2];
        batch = update(m,newX,newY);
        sequential = partialFit(partialFit(m,newX(1,:),newY(1)),newX(2,:),newY(2));
        [expected,variance] = exactPrediction([x;newX],[y;newY],points,opts);
        [mu,sd] = predict(batch,points);
        [singleMu,singleSD] = predict(sequential,points);
        verifyEqual(testCase,mu,expected,'AbsTol',1e-7);
        verifyEqual(testCase,sd.^2,variance+opts.NoiseVariance,'AbsTol',1e-7);
        verifyEqual(testCase,singleMu,mu,'AbsTol',1e-7);
        verifyEqual(testCase,singleSD,sd,'AbsTol',1e-7);
        prior = reset(batch);
        verifyEqual(testCase,prior.KernelFunction,m.KernelFunction);
        verifyEqual(testCase,prior.LengthScale,m.LengthScale);
        verifyEqual(testCase,prior.C_g, ...
            kernel(x,x,opts)+m.Jitter*m.SignalVariance*eye(3),'AbsTol',1e-12);
        [priorMu,priorSD] = predict(prior,points,struct('IncludeObservationNoise',false));
        verifyEqual(testCase,priorMu,zeros(5,1),'AbsTol',1e-12);
        verifyEqual(testCase,priorSD.^2,2*ones(5,1),'AbsTol',1e-12);
    end
end
end

function testKernelOptionValidationAndNormalization(testCase)
x = [0 0;1 2];
y = [1;2];
for kernelName = supportedKernels()
    m = incrementalRegressionGP(x,y,struct('KernelFunction',upper(string(kernelName{1}))));
    verifyEqual(testCase,m.KernelFunction,kernelName{1});
    m = initialize(m,x,y,struct('KernelFunction',kernelName{1}));
    verifyEqual(testCase,m.KernelFunction,kernelName{1});
    verifyError(testCase,@() incrementalRegressionGP(x,y, ...
        struct('KernelFunction',kernelName{1},'LengthScale',[1 2 3])), ...
        'incrementalRegressionGP:LengthScaleSize');
end
for value = {1,'unknown','',["matern32" "matern52"],['matern32';'matern52']}
    verifyError(testCase,@() incrementalRegressionGP(x,y, ...
        struct('KernelFunction',value{1})),'incrementalRegressionGP:UnsupportedKernel');
end
verifyEqual(testCase,incrementalRegressionGP(x,y).KernelFunction,'squaredexponential');
end

function testKernelsHandleOverflowingDistances(testCase)
for kernelName = supportedKernels()
    m = incrementalRegressionGP([0 0],1, ...
        struct('KernelFunction',kernelName{1},'LengthScale',realmin));
    points = [1 0;realmax realmax];
    [mu,sd] = predict(m,points);
    verifyEqual(testCase,mu,zeros(2,1));
    verifyEqual(testCase,sd.^2,(m.SignalVariance+m.NoiseVariance)*ones(2,1), ...
        'AbsTol',1e-12);
    updated = fit(m,points,[2;3]);
    verifyEqual(testCase,updated.mu_g,m.mu_g);
    verifyEqual(testCase,updated.C_g,m.C_g);
end
end

function testCovarianceRemainsPSDAndStateBounded(testCase)
opts = struct('LengthScale',0.3,'SignalVariance',2,'NoiseVariance',0.05);
initial = incrementalRegressionGP([0;0.5;1],[1;2;-1],opts);
for x = linspace(-1,2,31)
    m = update(initial,x,0);
    verifyGreaterThanOrEqual(testCase,min(eig(m.C_g)),-1e-10);
    verifyEqual(testCase,m.C_g,m.C_g.','AbsTol',1e-12);
end
m = initial;
for k = 1:100
    m = fit(m,sin(k),cos(k));
end
verifySize(testCase,m.mu_g,[3 1]);
verifySize(testCase,m.C_g,[3 3]);
verifyEqual(testCase,m.X,initial.X);
verifyGreaterThanOrEqual(testCase,min(eig(m.C_g)),-1e-10);
end

function testDefaultsAndReset(testCase)
x = [0;0.5;1];
y = [1;2;-1];
empty = incrementalRegressionGP();
verifyError(testCase,@() predict(empty,x),'incrementalRegressionGP:NotInitialized');
m = fit(empty,x,y);
reference = incrementalRegressionGP(x,y);
verifyEqual(testCase,predict(m,x),predict(reference,x),'AbsTol',1e-12);
m = updateMetrics(m,x,y);
m = reset(m);
[mu,sd] = predict(m,x);
verifyEqual(testCase,mu,zeros(3,1),'AbsTol',1e-12);
verifyEqual(testCase,sd.^2,ones(3,1)*(m.SignalVariance+m.NoiseVariance),'AbsTol',1e-12);
verifyEqual(testCase,m.NumTrainingObservations,0);
verifyEqual(testCase,m.Metrics.NumObservations,0);
verifyTrue(testCase,isnan(m.Metrics.MeanSquaredError));
verifyEqual(testCase,reset(empty).NumPredictors,0);
m = initialize(m,[1;2],[3;4]);
verifyEqual(testCase,m.X,[1;2]);
verifyEqual(testCase,m.NumTrainingObservations,2);
end

function testARDAndRowObservation(testCase)
x = [0 0;1 2;2 -1];
y = [0;1;-1];
opts = struct('LengthScale',[0.3 1.7],'SignalVariance',2,'NoiseVariance',0.05);
m = incrementalRegressionGP(x,y,opts);
points = [0.2 0.3];
[mu,sd] = predict(m,points);
[expected,variance] = exactPrediction(x,y,points,opts);
verifySize(testCase,mu,[1 1]);
verifySize(testCase,sd,[1 1]);
verifyEqual(testCase,mu,expected,'AbsTol',1e-8);
verifyEqual(testCase,sd.^2,variance+opts.NoiseVariance,'AbsTol',1e-8);
m = fit(m,points,0.7);
verifyEqual(testCase,m.NumTrainingObservations,4);
end

function testHyperparameterLearningIsOptIn(testCase)
x = [-1;0;1];
y = [0;1;0];
defaultModel = incrementalRegressionGP(x,y);
fixedModel = incrementalRegressionGP(x,y,struct('LearnHyperparameters',false));
learningModel = incrementalRegressionGP(x,y,struct('LearnHyperparameters',true));
verifyFalse(testCase,defaultModel.LearnHyperparameters);
verifyTrue(testCase,learningModel.LearnHyperparameters);
verifyEqual(testCase,learningModel.mu_g,defaultModel.mu_g,'AbsTol',1e-12);
verifyEqual(testCase,learningModel.C_g,defaultModel.C_g,'AbsTol',1e-12);
defaultModel = fit(defaultModel,0.25,2);
fixedModel = fit(fixedModel,0.25,2);
verifyEqual(testCase,defaultModel,fixedModel);
verifyEqual(testCase,defaultModel.LengthScale,1);
verifyEqual(testCase,defaultModel.SignalVariance,1);
verifyEqual(testCase,defaultModel.NoiseVariance,1e-3);
assertThrows(testCase,@() incrementalRegressionGP(x,y,struct('LearnHyperparameters',1)));
assertThrows(testCase,@() incrementalRegressionGP(x,y, ...
    struct('LearnHyperparameters',[true false])));
end

function testLearningMatchesSigmaPointConditioning(testCase)
for kernelName = supportedKernels()
basis = [-1;0;1];
opts = struct('LearnHyperparameters',true,'LengthScale',0.7, ...
    'SignalVariance',1.2,'NoiseVariance',0.3,'KernelFunction',kernelName{1});
m = reset(incrementalRegressionGP(basis,zeros(3,1),opts));
x = [-0.4;0.6];
y = [1;-0.5];
eta = log([opts.LengthScale;sqrt(opts.SignalVariance);sqrt(opts.NoiseVariance)]);
points = [eta,eta+sqrt(0.6)*eye(3),eta-sqrt(0.6)*eye(3)];
priorWeights = [0.5;ones(6,1)/12];
means = zeros(6,7);
covariances = zeros(3,3,7);
likelihoods = zeros(7,1);
for k = 1:7
    candidate = struct('LengthScale',exp(points(1,k)), ...
        'SignalVariance',exp(2*points(2,k)), ...
        'NoiseVariance',exp(2*points(3,k)),'KernelFunction',kernelName{1});
    K = kernel(basis,basis,candidate)+m.Jitter*candidate.SignalVariance*eye(3);
    cross = kernel(x,basis,candidate);
    J = cross/K;
    B = kernel(x,x,candidate)-J*cross.';
    S = B+J*m.C_g*J.'+candidate.NoiseVariance*eye(2);
    G = m.C_g*J.'/S;
    means(:,k) = [G*y;points(:,k)];
    covariances(:,:,k) = m.C_g-G*S*G.';
    likelihoods(k) = priorWeights(k)*exp(-0.5*(y.'*(S\y)))/sqrt(det(S));
end
weights = likelihoods/sum(likelihoods);
expectedMean = means*weights;
expectedCovariance = zeros(3);
for k = 1:7
    delta = means(1:3,k)-expectedMean(1:3);
    expectedCovariance = expectedCovariance+weights(k)*(covariances(:,:,k)+delta*delta.');
end
updated = fit(m,x,y);
verifyEqual(testCase,updated.mu_g,expectedMean(1:3),'AbsTol',1e-10);
verifyEqual(testCase,updated.C_g,expectedCovariance,'AbsTol',1e-10);
verifyEqual(testCase,updated.LengthScale,exp(expectedMean(4)),'AbsTol',1e-10);
verifyEqual(testCase,updated.SignalVariance,exp(2*expectedMean(5)),'AbsTol',1e-10);
verifyEqual(testCase,updated.NoiseVariance,exp(2*expectedMean(6)),'AbsTol',1e-10);
verifyEqual(testCase,updated.NumTrainingObservations,2);
verifyEqual(testCase,fit(m,[],[]),m);
verifyEqual(testCase,update(m,x,y),updated);
verifyEqual(testCase,partialFit(m,x,y),updated);
verifyEqual(testCase,m.NumTrainingObservations,0);
end
end

function testLearningSineStreamAndReset(testCase)
rng(1)
basis = (-10:9).';
opts = struct('LearnHyperparameters',true,'NoiseVariance',1);
m = reset(incrementalRegressionGP(basis,zeros(size(basis)),opts));
sorted = sort(-10+20*rand(50,1));
firstHalf = sorted(1:2:end);
x = [firstHalf;flipud(firstHalf)];
y = sin(x)+0.1*randn(size(x));
for k = 1:numel(y)
    m = fit(m,x(k),y(k));
end
points = (-10:0.2:9.8).';
[mu,sd,interval] = predict(m,points);
verifyTrue(testCase,all(isfinite([mu;sd;interval(:)])));
verifyLessThan(testCase,sqrt(mean((mu-sin(points)).^2)),0.4);
parameters = [m.LengthScale,m.SignalVariance,m.NoiseVariance];
verifyTrue(testCase,all(isfinite(parameters) & parameters > 0));
verifyTrue(testCase,all(abs(parameters-[1 1 1]) > 1e-6));
verifyEqual(testCase,m.X,basis);
verifySize(testCase,m.C_g,[20 20]);
verifyGreaterThanOrEqual(testCase,min(eig(m.C_g)),-1e-10);
verifyEqual(testCase,m.NumTrainingObservations,50);
prior = reset(m);
verifyTrue(testCase,prior.LearnHyperparameters);
verifyEqual(testCase,[prior.LengthScale,prior.SignalVariance,prior.NoiseVariance],parameters);
verifyEqual(testCase,prior.NumTrainingObservations,0);
[priorMean,priorSD] = predict(prior,points,struct('IncludeObservationNoise',false));
verifyEqual(testCase,priorMean,zeros(size(points)),'AbsTol',1e-12);
verifyEqual(testCase,priorSD.^2,m.SignalVariance*ones(size(points)),'AbsTol',1e-10);
fresh = reset(incrementalRegressionGP(basis,zeros(size(basis)), ...
    struct('LearnHyperparameters',true,'LengthScale',m.LengthScale, ...
    'SignalVariance',m.SignalVariance,'NoiseVariance',m.NoiseVariance)));
verifyEqual(testCase,fit(prior,0.2,0.5),fit(fresh,0.2,0.5));
reinitialized = initialize(m,[0;1],[1;2]);
verifyFalse(testCase,reinitialized.LearnHyperparameters);
end

function testLearningARDMeanAndMetrics(testCase)
for kernelName = supportedKernels()
basis = [-1 0;0 1;1 -1];
opts = struct('LearnHyperparameters',true,'LengthScale',[0.5 1.5], ...
    'NoiseVariance',0.2,'BasisFunction','linear','Beta',[1;2;-1], ...
    'KernelFunction',kernelName{1});
m = incrementalRegressionGP(basis,[0;1;2],opts);
x = [0.2 0.4;-0.1 0.7];
y = [2;0.5];
expected = loss(m,x,y);
updated = updateMetricsAndFit(m,x,y);
verifyEqual(testCase,updated.Metrics.MeanSquaredError,expected,'AbsTol',1e-12);
verifyEqual(testCase,updated.Metrics.NumObservations,2);
verifyEqual(testCase,updated.NumTrainingObservations,5);
verifySize(testCase,updated.LengthScale,[1 2]);
verifyEqual(testCase,updated.Beta,opts.Beta);
verifyEqual(testCase,updated.X,basis);
for k = 1:30
    updated = fit(updated,x,y);
end
[mu,responseSD] = predict(updated,x);
[~,latentSD] = predict(updated,x,struct('IncludeObservationNoise',false));
verifyTrue(testCase,all(isfinite([mu;responseSD;latentSD])));
verifyEqual(testCase,responseSD.^2-latentSD.^2, ...
    updated.NoiseVariance*ones(2,1),'AbsTol',1e-10);
verifyGreaterThanOrEqual(testCase,min(eig(updated.C_g)),-1e-10);
shared = incrementalRegressionGP(basis,[0;1;2], ...
    struct('LearnHyperparameters',true,'LengthScale',0.8,'KernelFunction',kernelName{1}));
shared = fit(shared,x,y);
if startsWith(kernelName{1},'ard')
    verifySize(testCase,shared.LengthScale,[1 2]);
else
    verifySize(testCase,shared.LengthScale,[1 1]);
end
verifyEqual(testCase,updated.KernelFunction,kernelName{1});
prior = reset(updated);
freshOptions = opts;
freshOptions.LengthScale = updated.LengthScale;
freshOptions.SignalVariance = updated.SignalVariance;
freshOptions.NoiseVariance = updated.NoiseVariance;
fresh = reset(incrementalRegressionGP(basis,[0;1;2],freshOptions));
verifyEqual(testCase,fit(prior,x,y),fit(fresh,x,y));
end
end

function testMeanFunctions(testCase)
x = [-1;0;1];
y = [1;2;3];
names = {'none','constant','linear','purequadratic'};
betas = {[],2,[1;2],[1;2;3]};
for k = 1:numel(names)
    opts = struct('BasisFunction',names{k});
    if ~isempty(betas{k})
        opts.Beta = betas{k};
    end
    m = reset(incrementalRegressionGP(x,y,opts));
    expected = zeros(3,1);
    if k == 2
        expected(:) = 2;
    elseif k == 3
        expected = 1+2*x;
    elseif k == 4
        expected = 1+2*x+3*x.^2;
    end
    verifyEqual(testCase,predict(m,x),expected,'AbsTol',1e-12);
end
end

function testIntervalsAndNoiseSemantics(testCase)
m = incrementalRegressionGP([0;0.5;1],[1;2;-1],struct('NoiseVariance',0.04));
points = [-0.2;0.2;1.2];
[mu,responseSD,wide] = predict(m,points,struct('Alpha',0.01));
[latentMean,latentSD,latentInterval] = predict(m,points, ...
    struct('Alpha',0.01,'IncludeObservationNoise',false));
[~,sd,narrow] = predict(m,points,struct('Alpha',0.2));
verifyEqual(testCase,responseSD,sd,'AbsTol',1e-12);
verifyEqual(testCase,latentMean,mu,'AbsTol',1e-12);
verifyEqual(testCase,responseSD.^2-latentSD.^2,0.04*ones(3,1),'AbsTol',1e-12);
verifyGreaterThan(testCase,wide(:,2)-wide(:,1),narrow(:,2)-narrow(:,1));
verifyLessThan(testCase,latentInterval(:,2)-latentInterval(:,1),wide(:,2)-wide(:,1));
end

function testDuplicateBasisAndSmallNoise(testCase)
x = [0;0;0.5;1];
y = [1;1;2;-1];
m = incrementalRegressionGP(x,y,struct('NoiseVariance',1e-8));
for k = 1:20
    m = fit(m,x,y);
end
[mu,sd] = predict(m,x);
verifyTrue(testCase,all(isfinite(mu)));
verifyTrue(testCase,all(isfinite(sd)));
verifyGreaterThanOrEqual(testCase,min(eig(m.C_g)),-1e-10);
end

function testEmptyBatchesAndInputValidation(testCase)
m = incrementalRegressionGP([0;1],[1;2]);
[mu,sd,interval] = predict(m,zeros(0,1));
verifySize(testCase,mu,[0 1]);
verifySize(testCase,sd,[0 1]);
verifySize(testCase,interval,[0 2]);
verifyEqual(testCase,fit(m,[],[]),m);
verifyError(testCase,@() loss(m,[],[]),'incrementalRegressionGP:EmptyLoss');
verifyError(testCase,@() incrementalRegressionGP([],[]),'incrementalRegressionGP:EmptyData');
verifyError(testCase,@() fit(m,[1;2],[1 2;3 4]),'incrementalRegressionGP:ResponseSize');
verifyError(testCase,@() predict(m,[1 2]),'incrementalRegressionGP:PredictorSize');
verifyError(testCase,@() predict(m,zeros(0,2)),'incrementalRegressionGP:PredictorSize');
verifyError(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('LengthScale',[1 2])), ...
    'incrementalRegressionGP:LengthScaleSize');
verifyError(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('BasisFunction','linear')), ...
    'incrementalRegressionGP:BetaSize');
verifyError(testCase,@() predict(m,1,struct('Typo',1)),'incrementalRegressionGP:UnknownOption');
verifyError(testCase,@() predict(m,1,[]),'incrementalRegressionGP:InvalidOptions');
verifyError(testCase,@() incrementalRegressionGP(struct('X',1,'Y',1)), ...
    'incrementalRegressionGP:InvalidInitialModel');
assertThrows(testCase,@() predict(m,NaN));
assertThrows(testCase,@() predict(m,1,struct('Alpha',2)));
assertThrows(testCase,@() predict(m,1,struct('IncludeObservationNoise',2)));
assertThrows(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('LengthScale',-1)));
assertThrows(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('NoiseVariance',0)));
assertThrows(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('LengthScale','a')));
assertThrows(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('SignalVariance',true)));
verifyError(testCase,@() incrementalRegressionGP([1;2],[1;2],struct('BasisFunction',1)), ...
    'incrementalRegressionGP:UnsupportedBasis');
verifyEqual(testCase,incrementalRegressionGP([0;1],[1;2],struct('Beta',[])).Beta,zeros(0,1));
end

function testMetricsArePreUpdate(testCase)
m = incrementalRegressionGP([0;0.5;1],[1;2;-1]);
x = [0.2;0.7];
y = [3;-2];
expected = (predict(m,x)-y).^2;
verifyEqual(testCase,perObservationLoss(m,x,y),expected,'AbsTol',1e-12);
verifyEqual(testCase,loss(m,x,y),mean(expected),'AbsTol',1e-12);
trained = updateMetricsAndFit(m,x,y);
verifyEqual(testCase,trained.Metrics.MeanSquaredError,mean(expected),'AbsTol',1e-12);
verifyEqual(testCase,trained.Metrics.NumObservations,2);
verifyEqual(testCase,trained.NumTrainingObservations,5);
metricsOnly = updateMetrics(m,x,y);
verifyEqual(testCase,metricsOnly.mu_g,m.mu_g);
again = updateMetrics(metricsOnly,x,y);
verifyEqual(testCase,again.Metrics.NumObservations,4);
verifyEqual(testCase,again.Metrics.MeanSquaredError,mean(expected),'AbsTol',1e-12);
end

function testFitrgpConversion(testCase)
x = [1 20;2 30;4 40;8 80;6 60;5 15;3 50;7 45];
y = [1;3;2;5;4;0;3;2];
points = [1.5 25;6 70;9 30];
for standardize = [false true]
    for kernelName = supportedKernels()
        for basisName = {'none','constant','linear','purequadratic'}
            g = fitrgp(x,y,'KernelFunction',kernelName{1}, ...
                'BasisFunction',basisName{1},'Standardize',standardize, ...
                'FitMethod','exact','PredictMethod','exact');
            m = incrementalRegressionGP(g);
            [expected,expectedSD,expectedInterval] = predict(g,points);
            [mu,sd,interval] = predict(m,points);
            verifyEqual(testCase,mu,expected,'AbsTol',1e-6);
            verifyEqual(testCase,sd,expectedSD,'AbsTol',1e-6);
            verifyEqual(testCase,interval,expectedInterval,'AbsTol',3e-6);
            verifyEqual(testCase,m.NoiseVariance,g.Sigma^2,'AbsTol',1e-12);
            verifyEqual(testCase,m.Beta,g.Beta,'AbsTol',1e-12);
            verifyEqual(testCase,m.KernelFunction,kernelName{1});
            verifyEqual(testCase,m.LengthScale,g.KernelInformation.KernelParameters(1:end-1).', ...
                'AbsTol',1e-12);
        end
    end
end
end

function testUnsupportedFitrgpModels(testCase)
x = (1:8).';
y = sin(x);
g = fitrgp(x,y,'KernelFunction','exponential');
verifyError(testCase,@() incrementalRegressionGP(g),'incrementalRegressionGP:UnsupportedKernel');
g = fitrgp(x,y,'PredictMethod','fic','ActiveSetSize',3);
verifyError(testCase,@() incrementalRegressionGP(g),'incrementalRegressionGP:UnsupportedInitialModel');
g = fitrgp(x,y,'Weights',(1:8).');
verifyError(testCase,@() incrementalRegressionGP(g),'incrementalRegressionGP:UnsupportedWeights');
end

function testImportedModelUpdateAndReset(testCase)
for kernelName = supportedKernels()
x = [1 20;2 30;4 40;8 80;6 60;5 15;3 50;7 45];
y = [1;3;2;5;4;0;3;2];
g = fitrgp(x,y,'KernelFunction',kernelName{1}, ...
    'BasisFunction','linear','Standardize',true);
m = incrementalRegressionGP(g);
newX = x([2 6],:);
newY = [5;-2];
updated = fit(m,newX,newY);
training = ([x;newX]-m.PredictorLocation)./m.PredictorScale;
points = [1.5 25;6 70;9 30];
standardPoints = (points-m.PredictorLocation)./m.PredictorScale;
opts = struct('LengthScale',m.LengthScale,'SignalVariance',m.SignalVariance, ...
    'NoiseVariance',m.NoiseVariance,'KernelFunction',kernelName{1});
centeredResponse = [y;newY]-[ones(10,1),training]*m.Beta;
[expected,variance] = exactPrediction(training,centeredResponse,standardPoints,opts);
expected = expected+[ones(3,1),standardPoints]*m.Beta;
[mu,sd] = predict(updated,points);
verifyEqual(testCase,mu,expected,'AbsTol',1e-6);
verifyEqual(testCase,sd.^2,variance+m.NoiseVariance,'AbsTol',1e-6);
prior = reset(updated);
verifyEqual(testCase,predict(prior,points),[ones(3,1),standardPoints]*m.Beta,'AbsTol',1e-12);
verifyEqual(testCase,prior.PredictorLocation,m.PredictorLocation);
verifyEqual(testCase,prior.PredictorScale,m.PredictorScale);
end
end

function testStandardizedConstantPredictor(testCase)
x = [(1:8).',ones(8,1)*4];
y = sin(x(:,1));
g = fitrgp(x,y,'Standardize',true,'KernelFunction','ardsquaredexponential');
m = incrementalRegressionGP(g);
points = [0.5 4;3.5 4;9 4];
[expected,expectedSD] = predict(g,points);
[mu,sd] = predict(m,points);
verifyEqual(testCase,mu,expected,'AbsTol',1e-6);
verifyEqual(testCase,sd,expectedSD,'AbsTol',1e-6);
end

function [mu,variance] = exactPrediction(x,y,points,opts)
K = kernel(x,x,opts)+opts.NoiseVariance*eye(size(x,1));
cross = kernel(points,x,opts);
mu = cross*(K\y);
variance = opts.SignalVariance-sum((cross/K).*cross,2);
end

function K = kernel(A,B,opts)
K = zeros(size(A,1),size(B,1));
name = 'squaredexponential';
if isfield(opts,'KernelFunction')
    name = opts.KernelFunction;
end
for i = 1:size(A,1)
    distance = (B-A(i,:))./opts.LengthScale;
    r = sqrt(sum(distance.^2,2));
    switch name
        case {'matern32','ardmatern32'}
            covariance = (1+sqrt(3)*r).*exp(-sqrt(3)*r);
        case {'matern52','ardmatern52'}
            covariance = (1+sqrt(5)*r+5*r.^2/3).*exp(-sqrt(5)*r);
        otherwise
            covariance = exp(-0.5*r.^2);
    end
    K(i,:) = opts.SignalVariance*covariance.';
end
end

function names = supportedKernels()
names = {'squaredexponential','ardsquaredexponential', ...
    'matern32','ardmatern32','matern52','ardmatern52'};
end

function assertThrows(testCase,f)
didThrow = false;
try
    f();
catch
    didThrow = true;
end
verifyTrue(testCase,didThrow,'Invalid input must raise an error.');
end
