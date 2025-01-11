function Global = SelctKnn2WModel(Global)
% Load Data
Name = Global.SelectData;
load(Name);
kfold = 5;
Features = data;
CVI = CrosValIndex(data,kfold);

Label =  Features(:,end);
H = hist(Label,1:max(Label));
W = zeros(size(Label));
for j = 1:max(Label)
    W(Label==j) = 1/H(j);
end

% 

Data.W = W;
Data.Inputs = Features(:,1:end-1);
Data.Targets = Label;
Data.Objs = Global.Objs;
Data.CVI = CVI;
% Data.CVI = CrosValIndex(Data.Targets,kf);
Data.kf = max(CVI);
Data.Knn = 2;
Global.Data = Data;
Data.ClassificationMethod = Global.ClassificationMethod;
Global.CostFunction = @(s) evaluatefeatures(s,Data);
Global.D = size(Data.Inputs,2);

Global.Th = 0.98;
end

function [z,Results] = evaluatefeatures(s,data)
s = s>0.5;
Objs = data.Objs;
ClassificationMethod = data.ClassificationMethod;
Objs = Objs>0.5;
W = data.W;
%% Classification Usin Decision Tree
[s1,s2] = size(s);
for ii = 1:s1
    if sum(s(ii,:))>0
        TrainFeaturs = data.Inputs(:,s(ii,:));
        TrainTargets = data.Targets;
        numFeatures = size(TrainFeaturs,2);
        %% Train the Classifier
        kf = data.kf;
        knn = data.Knn;
        CVI = data.CVI;
        for i = 1:kf

            TrainF = TrainFeaturs(CVI~=i,:);
            TrainL = TrainTargets(CVI~=i);

            ValidF = TrainFeaturs(CVI==i,:);
            ValidL = TrainTargets(CVI==i);
           

            Class = Knn2W(TrainF,TrainL,ValidF, knn,W(CVI~=i));

            CM(i).CMV = confusionmat(ValidL,Class);
            Oi(i,:) = ClassificationMetrics(CM(i).CMV); %#ok

            % Test data


        end
        Oi(isnan(Oi)) = 0;
        OTest = mean(Oi);

        % Objectives
        O = 1 - [OTest,(s2- numFeatures)/s2];
        if sum(isnan(O))>0
        oi =O(Objs); 
        z(ii,:) = ones(size(oi)); %#ok
        else
        z(ii,:) = O(Objs); %#ok
        end
        AllObjective = [Oi,ones(kf,1)*(s2- numFeatures)/s2];

        Results(ii).Objectives = 1 - z;%#ok
        Results(ii).SelectedFeatures = find(s(ii,:));%#ok
        Results(ii).NSelectedFeatures = numel(find(s(ii,:)));%#ok

        Results(ii).TestMetrics = OTest;
        Results(ii).ClassificationMethod = ClassificationMethod;
        Results(ii).NameMetrics = ['Acc', 'Sen', 'Sep', 'Per', 'F1s', 'Auc', 'Gmn', 'NSF'];
        Results(ii).AllObjectives = AllObjective;
    else
        z(ii,:) = inf(1,sum(Objs)); %#ok
        Results(ii).Objectives = 0.5;%#ok
        Results(ii).AllObjectives = 0.5;%#ok
    end
end
end


function indices = CrosValIndex(database,kf)
Flag = 0;
L = database(:,end);
M = max(L);
while Flag == 0
    indices = crossvalind('Kfold',numel(L),kf);
    flag = zeros(1,kf);
    for i = 1:kf
        Test = (indices == i);
        Train = ~Test;
        try
        if (numel(unique(L(Test)))==M) | (numel(unique(L(Train)))==M)
        flag(i) = 1;
        end
        catch
            lm = 0;
        end
    end
    if sum(flag)==kf
        break;
    end
end
end

function [Classes,Scores] = Knn2W(trainX, trainY,testZ, k,W)

% Classify using the Weighted Nearest neighbor algorithm

class               = unique(trainY);
nC = numel(class);
N                   = size(testZ, 1);
Classes              = zeros(N, 1);
Scores               = zeros(N, nC);

% [mIdx,mD] = knnsearch(trainX,testZ,'K',k,'Distance','seuclidean');
% if sum(sum(isnan(mD)))>1
[mIdx,mD] = knnsearch(trainX,testZ,'K',k,'Distance','euclidean');
% end
w1 = 1./(1+mD);
% w1 = w1./sum(w1,2);/sum((w2))
W = W.^2;
for i = 1:N
    w2 = (W(mIdx(i,:)))';
    w = (1*w2 + 1*w1(i,:));
    ClassesNeg = trainY(mIdx(i,:));
    vote = zeros(1,nC);
    for x = 1:nC % number of classes
         vote(x) = sum(w(ClassesNeg == x));
    end
    [~,Classes(i)] = max(vote);
    Scores(i,:) = vote;
end

end