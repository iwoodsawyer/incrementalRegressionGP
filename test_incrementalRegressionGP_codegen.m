function tests = test_incrementalRegressionGP_codegen
%TEST_INCREMENTALREGRESSIONGP_CODEGEN Compile C and compare MEX with MATLAB.
%   Run with runtests('test_incrementalRegressionGP_codegen.m').
%   Requires MATLAB Coder and a configured C compiler. Generated files are
%   placed in a temporary directory and removed after the test.
tests = functiontests(localfunctions);
end

function testGeneratedMexMatchesMATLAB(testCase)
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() removeGeneratedFiles(folder));
cfg = coder.config('mex');
cfg.TargetLang = 'C';
kernelNames = {'squaredexponential','ardsquaredexponential', ...
    'matern32','ardmatern32','matern52','ardmatern52'};
for numPredictors = 1:2
codegen('-config',cfg,'incrementalRegressionGP_codegen', ...
    '-args',{coder.typeof(0,[8 numPredictors],[true false]), ...
    coder.typeof(0,[8 1],[true false]), ...
    coder.typeof(0,[10 numPredictors],[true false]),false, ...
    coder.typeof('a',[1 21],[false true])}, ...
    '-d',folder,'-o',fullfile(folder,'gpNumeric_mex'));
addpath(folder);
for kernelName = kernelNames
for k = 1:8
    batchSizes = [0 1 3 7];
    predictionSizes = [0 1 5 9];
    index = mod(k-1,4)+1;
    learnHyperparameters = k > 4;
    locations = linspace(0.1,0.9,batchSizes(index)).'+(index-1)*0.02;
    Y = sin(6*locations)+k*0.01;
    queries = linspace(-0.1,1.1,predictionSizes(index)).';
    if numPredictors == 2
        X = [locations,1-locations];
        points = [queries,1-queries];
    else
        X = locations;
        points = queries;
    end
    [mu,sd,interval,mse] = incrementalRegressionGP_codegen(X,Y,points, ...
        learnHyperparameters,kernelName{1});
    [actualMu,actualSD,actualInterval,actualMSE] = gpNumeric_mex(X,Y,points, ...
        learnHyperparameters,kernelName{1});
    verifyEqual(testCase,actualMu,mu,'AbsTol',1e-9);
    verifyEqual(testCase,actualSD,sd,'AbsTol',1e-9);
    verifyEqual(testCase,actualInterval,interval,'AbsTol',1e-9);
    verifyEqual(testCase,actualMSE,mse,'AbsTol',1e-9);
end
end
clear gpNumeric_mex;
end
end

function removeGeneratedFiles(folder)
clear gpNumeric_mex;
if contains(path,folder)
    rmpath(folder);
end
rmdir(folder,'s');
end
