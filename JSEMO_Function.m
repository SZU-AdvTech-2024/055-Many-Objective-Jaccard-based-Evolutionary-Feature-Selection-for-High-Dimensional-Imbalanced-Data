function BestSolotion = JISEMO_Function(Global)
%% Many-Objective SetBased Evolutionary with Jaccard Similarity
%% Generate random population
P = JaccardInitialization(Global);
NFE = 0;
BestSolotion = P;
A = P;
%% Main Loop Of Optimization
for i = 1:Global.MaxIt
    Global.r = i/Global.MaxIt;
    Global.mu = 0.05*(1 - Global.r) + 0.001;
    [Offspring,A]  = SetVariations(P,A,Global);
    [P,FrontNo] = EnvironmentalSelection([P;Offspring;A],Global.N);
    BestSolotion = P(FrontNo==1);

    Costs = cat(1,BestSolotion.Cost);

    NFE = NFE + numel(Offspring);
    disp(['Iter ',num2str(i),', OneBest = ',num2str(1-[BestSolotion(1).Cost])]);
end



end

function [pop,Global] = JaccardInitialization(Global)
nPop = Global.N;

CostFunction = Global.CostFunction;
empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.Objectives = [];
empty_individual.AllObjectives = [];
empty_individual.Mean = [];
empty_individual.STD = [];
empty_individual.NFS = [];
empty_individual.Others = [];


pop = repmat(empty_individual, nPop, 1);

i = 1;
pop(i).Position = randi([0,1], [1,Global.D]);
[pop(i).Cost,out] = CostFunction(pop(i).Position);
pop(i).NFS = sum(pop(i).Position);
pop(i).Objectives = 1 - pop(i).Cost;
pop(i).AllObjectives = out.AllObjectives;
pop(i).Mean = mean(1 - pop(i).Cost);
pop(i).STD = std(1 - pop(i).Cost);
pop(i).Others = out;
while i<nPop
    x = rand( 1,Global.D);
    x = x<(0.1+0.8*rand);
    X = cat(1,pop(1:i).Position);
    Sim = Jaccard(X,x);
    if (Sim<Global.Sim)
        i = i + 1;
        pop(i).Position = x;
        [pop(i).Cost,out] = CostFunction(pop(i).Position);
        pop(i).NFS = sum(pop(i).Position);
        pop(i).Objectives = 1 - pop(i).Cost;
        pop(i).AllObjectives = out.AllObjectives;
        pop(i).Mean = mean(1 - pop(i).Cost);
        pop(i).STD = std(1 - pop(i).Cost);
        pop(i).Others = out;
    end
end

end


%% Functions
function CrowdDis = CrowdingDistance(PopObj,FrontNo)
% Calculate the crowding distance of each solution front by front

[N,M]    = size(PopObj);
CrowdDis = zeros(1,N);
Fronts   = setdiff(unique(FrontNo),inf);
for f = 1 : length(Fronts)
    Front = find(FrontNo==Fronts(f));
    Fmax  = max(PopObj(Front,:),[],1);
    Fmin  = min(PopObj(Front,:),[],1);
    for i = 1 : M
        [~,Rank] = sortrows(PopObj(Front,i));
        CrowdDis(Front(Rank(1)))   = inf;
        CrowdDis(Front(Rank(end))) = inf;
        for j = 2 : length(Front)-1
            CrowdDis(Front(Rank(j))) = CrowdDis(Front(Rank(j)))+(PopObj(Front(Rank(j+1)),i)-PopObj(Front(Rank(j-1)),i))/(Fmax(i)-Fmin(i));
        end
    end
end
end


function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II/SDR


%% Normalization
PopObj = cat(1,Population.Cost);
Decs  = cat(1,Population.Position);
% Get unique individuals in decision space
[~, U_Decs, ~] = unique(Decs, 'rows');
Population = Population(U_Decs);
PopObj = PopObj(U_Decs,:);

% Get unique individuals in objective space
[~,x]      = unique(roundn(PopObj,-6),'rows');
PopObj     = PopObj(x,:);
Population = Population(x);
N          = min(N,length(Population));

%% Non-dominated sorting
[FrontNo,MaxFNo] = NDSort(PopObj,N);

%% Calculate the crowding distance of each solution
CrowdDis = CrowdingDistance(PopObj,FrontNo);

%% Select the solutions in the last front based on their crowding distances
Next = FrontNo < MaxFNo;

    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;



%% Population for next generation
Population = Population(Next);
FrontNo    = FrontNo(Next);
CrowdDis   = CrowdDis(Next);
end

function [NewPop,Archive] = SetVariations(Parent,Archive,Global)
%% Parameter setting

%% Set operators for binary encoding
Offspring = SetBVariations(Parent,Global);

[Offspring] = JaccardSimiarity(Archive,Offspring,Global);
%% Evaluation
CostFunction = Global.CostFunction;
[Costs,Out] = CostFunction(Offspring);

nnPop = size(Offspring,1);
NewPop.Position = [];  NewPop.Cost = [];
NewPop.Objectives = []; NewPop.NFS = [];
NewPop.Others = [];
NewPop.Mean = [];
NewPop.STD = [];
NewPop.AllObjectives = [];

NewPop = repmat(NewPop,nnPop,1);
for i = 1:nnPop
    NewPop(i).Position = Offspring(i,:);
    NewPop(i).Cost = Costs(i,:);
    NewPop(i).Objectives = 1 - Costs(i,:);
    NewPop(i).AllObjectives = Out(i).AllObjectives;
    NewPop(i).NFS = sum(NewPop(i).Position);
    NewPop(i).Others = Out(i);
    NewPop(i).Mean = mean(1 - Costs(i,1:end-1));
    NewPop(i).STD = std(1 - Costs(i,1:end-1));
end


Archive = [Archive;NewPop];
if numel(Archive)>2*nnPop
    [~,SO] = sort(cat(1,Archive.Mean),'descend');
    Archive = Archive(SO(1:2*nnPop));
end
end

function Offspring = SetBVariations(Parent,Global)
Parents = cat(1,Parent.Position);
mu = Global.mu;
[N,D] = size(Parents);
Offspring = zeros(N,D);
for k = 1:2:N
%         Th = Th0 + (1-Th0)*(r);
        % Select Parents Indices
        i1 = randi([1 N]);
        i2 = randi([1 N]);

        % Select Parents
        p1 = Parents(i1,:);
        p2 = Parents(i2,:);
 

        % Apply Crossover
        [Offspring(k,:),Offspring(k+1,:)]=...
            Crossover(p1,p2);
        Offspring(k,:) = Mutate(Offspring(k,:),mu);

        Offspring(k+1,:) = Mutate(Offspring(k+1,:),mu);
end

end
function [y1, y2]=Crossover(x1,x2)

    y1 = x1.*x2;
    y2 = x1 | x2;


if sum(y2)==0
    y2 = (x1 | x2);
end
if sum(y1)==0
    y1 = (x1 | x2);
end
if numel(y1)==sum(y1)
    y1 = randi([0 1],[1,numel(y1)]);
end
if numel(y2)==sum(y2)
    y2 = randi([0 1],[1,numel(y1)]);
end

end
function z=Mutate(X,mu)
nVar = numel(X);
x = find(X);
Temp = 1:nVar;
Temp = setdiff(Temp,x);
mu1 = 0.1*rand();
mu2 = mu;
nVar1=numel(Temp);
if rand>0.5
    nmu=ceil(mu1*nVar1);
    j=randsample(nVar1,nmu);
    y=sort([x,Temp(j)]);
else
    nVar2 = numel(x);
    nmu=ceil(mu2*nVar2);
    nmu = min(nmu,nVar1);
    j=randsample(nVar2,nmu);
    j1 = randsample(nVar1,nmu);
    x(j) = Temp(j1);
    y = sort(x);
end
z = zeros(1,nVar);
z(y) = 1;
if sum(z)==0 || sum(z)==numel(z)
    z = randi([0 1],[1,numel(z)]);
end
end



% NDSort - Do non-dominated sorting by efficient non-dominated sort.
function [FrontNo,MaxFNo] = NDSort(PopObj,nSort)
[PopObj,~,Loc] = unique(PopObj,'rows');
Table   = hist(Loc,1:max(Loc));
[N,M]   = size(PopObj);
FrontNo = inf(1,N);
MaxFNo  = 0;
while sum(Table(FrontNo<inf)) < min(nSort,length(Loc))
    MaxFNo = MaxFNo + 1;
    for i = 1 : N
        if FrontNo(i) == inf
            Dominated = false;
            for j = i-1 : -1 : 1
                if FrontNo(j) == MaxFNo
                    m = 2;
                    while m <= M && PopObj(i,m) >= PopObj(j,m)
                        m = m + 1;
                    end
                    Dominated = m > M;
                    if Dominated || M == 2
                        break;
                    end
                end
            end
            if ~Dominated
                FrontNo(i) = MaxFNo;
            end
        end
    end
end
FrontNo = FrontNo(:,Loc);
end

function [X,Flag] = JaccardSimiarity(KD,X,Global)
r = Global.r;
Th = Global.Th + (1 -r)*(1 - Global.Th);

N  = size(X,1);

nS = numel(KD);

for i = 1:N
    Flag = 1;
    x = X(i,:);
    nD = sum(x);
    if r<0.0
        A = cat(1,KD.Position);
    else
        A = cat(1,KD(find([KD.NFS]==nD)).Position);
    end

    if isempty(A)
        Sim = 0;
    else
        [Sim,xs] = Jaccard(A,x);
    end

    if Sim >=  Th
        iKD = sort(randperm(nS,min(20,nS)));
        for j = 1:numel(iKD)
            y = KD(iKD(j)).Position | x;
            nD = sum(y);
            if nD==0 || nD==numel(y)
                y = randi([0 1],[1,numel(y)]);
                nD = sum(y);
            end

            Sim = Jaccard(xs,y);

                if Sim<Th
                    Flag = 0;
                    x = y;
                    break;
                end
        end

    end

    KD(end+1).Position = x;
    KD(end).NFS = nD;
    KD(end).Flag = Flag;
    X(i,:) = x;
end
end

function [s,xs] = Jaccard(A,B)
n = size(A,1);
B = ones(n,1)*B;
s = sum((A & B),2)./sum( (A | B),2);
[s,id] = max(s);
xs = A(id,:);
end


function y=MutateB(x,mu)

nVar=numel(x);

nmu=ceil(mu*nVar);

j=randsample(nVar,nmu);
j = j(:)';
y=x;
y(j)=1-x(j);

end


