clc
clear
close all

%% Detimine the Objectives that you want
% 1 implyes the coressponding objective is selected
% 0 implyes the coressponding objective is ignored

% 1. Accuracy,    Acc
% 2. Sensitivity, Sen
% 3. Specificity, Sep
% 4. Precision,   Per
% 5. F1_score,    F1s
% 6. AUC,         Auc
% 7. Gmean        Gmn

% 8. Number of Selected Features: NSF

%            [ Acc, Sen, Sep, Per, F1s, Auc, Gmn, NSF]
Global.Objs = [ 0,    1,  1,   1,   0,   0,  0,   1];
%% Select Parameter
Global.MaxIt = 100;     % Popualtion size
Global.N = 100;         % Maximum generation
Global.Sim = 0.3;
Global.M = sum(Global.Objs);       % number of Objectives
Global.Show = 0;    % A Option for dispalaing during the search
Global.ClassificationMethod = 'Knn2w';

%% Select Data and Cost Function
DataName = 'Leukemia_2.mat';
Global.SelectData = DataName;
disp('JSEMO: Many-Objective Feature Selection with KNN2W ...');

%% Run JSEMO Algorithm
Global = SelctKnn2WModel(Global);
BestSolotion = JSEMO_Function(Global);

% Results for the first solution from the Pareto front
disp('f1:Sensitivity f2:Specificity f3:Precision f4:FN')
disp(100*(1 - BestSolotion(1).Cost))


disp(['Full Feature Numbers:', num2str(Global.D)])
disp(['Selected Feature Numbers:', num2str(BestSolotion(1).NFS)])

