function [O,Result] = ClassificationMetrics(CM)
nC = size(CM,1);

switch nC
    case 2
        TP = CM(1,1);
        FN = CM(1,2);
        FP = CM(2,1);
        TN = CM(2,2);
        
    otherwise
        TP = zeros(1,nC);
        FN = zeros(1,nC);
        FP = zeros(1,nC);
        TN = zeros(1,nC);
        for i=1:nC
            TP(i) = CM(i,i);
            FN(i) = sum(CM(i,:))-CM(i,i);
            FP(i) = sum(CM(:,i))-CM(i,i);
            TN(i) = sum(CM(:))-TP(i)-FP(i)-FN(i);
        end
        
end


P = TP + FN;
N = FP + TN;
switch nC
    case 2
        accuracy = (TP+TN)/(P+N);
        Sensitivity = TP/P; % or Recall
        Specificity = TN/N;
        Precision = TP/(TP+FP);
        % Area Under Curve
        AUC = 0.5*(TP/P + TN/N);
        Gmean = (Sensitivity*Specificity).^0.5;
        %Kappa Calculation BY 2x2 Matrix Shape
        pox = sum(accuracy);
        Px = sum(P);TPx=sum(TP);FPx=sum(FP);TNx=sum(TN);FNx=sum(FN);Nx=sum(N);
        pex = ( (Px.*(TPx+FPx))+(Nx.*(FNx+TNx)) ) ./ ( (TPx+TNx+FPx+FNx).^2 );
        kappa_overall = ([( pox-pex ) ./ ( 1-pex );( pex-pox ) ./ ( 1-pox )]);
        kappa = max(kappa_overall);
        
        Result.Accuracy = accuracy;
    otherwise
        accuracy = (TP)./(P+N);
        
        Sensitivity = TP./P; % or Recall
        Specificity = TN./N;
        Precision = TP./(TP+FP);
        % Area Under Curve
        AUC = 0.5*(TP./P + TN./N);
        Gmean = (Sensitivity.*Specificity).^0.5;
        %Kappa Calculation BY nC x nC Matrix Shape
        po = accuracy;
        pe = ( (P.*(TP+FP))+(N.*(FN+TN)) ) ./ ( (TP+TN+FP+FN).^2 );
        kappa = ([( po-pe ) ./ ( 1-pe );( pe-po ) ./ ( 1-po )]);
        kappa = max(kappa);
        
        Result.Accuracy = sum(accuracy);
        
end



beta = 1;
F1_score = ( (1+(beta^2))*(Sensitivity.*Precision) ) ./ ( (beta^2)*(Precision+Sensitivity) );

% Matthews Correlation Coefficient
% MCC = [( TP.*TN - FP.*FN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) );...
%     ( FP.*FN - TP.*TN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) )] ;
% MCC = max(MCC);

%Output Struct for individual Classes
%  RefereceResult.Class=class_ref;
RefResult.AccuracyInTotal = accuracy';
% RefResult.ErrorInTotal = Error';
RefResult.Sensitivity = Sensitivity';
RefResult.Specificity = Specificity';
RefResult.Precision = Precision';

RefResult.F1_score = F1_score';
% RefResult.MCC = MCC';
RefResult.Kappa = kappa';
RefResult.AUC = AUC;
RefResult.TP = TP';
RefResult.FP = FP';
RefResult.FN = FN';
RefResult.TN = TN';
RefResult.Gmean = Gmean;


%Output Struct for over all class lists
Result.Sensitivity = mean(Sensitivity);
Result.Specificity = mean(Specificity);
Result.Precision = mean(Precision);
Result.F1_score = mean(F1_score);
% Result.MCC = mean(MCC);
Result.AUC = mean(AUC);
Result.Gmean = mean(Gmean);
Result.kappa = mean(kappa);
Result.RefResult = RefResult;
O = [Result.Accuracy,Result.Sensitivity,Result.Specificity,...
    Result.Precision,Result.F1_score,Result.AUC,Result.Gmean];