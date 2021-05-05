%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posteriors_x,g_x]=lab1_pt2(x,Training_Data)

% x = vector sample to be tested (to identify its probable class label)
% featureOfInterest = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posterior_y  = Posterior probabilities
% g_x = value of the discriminant function

D=Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[~,N]=size(D);    
 
%Col 2 was used to analyze Sepal Width

f1=D(:,1);  % feature samples
f2=D(:,2);  % feature samples
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hint: use the commands "find" and "length"

% Prior probabilities
Pr1 = length(find(la==1))/length(la);
Pr2 = length(find(la==2))/length(la);

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%
%Based off Sepal Width the mean and std dev. was calculated for each class

M1 = [ mean(f1(find(la==1))) mean(f2(find(la==1)))]; 
M2 = [ mean(f1(find(la==2))) mean(f2(find(la==2)))];
sig1 = cov(f1(find(la==1)), f2(find(la==1)));       
sig2 = cov(f1(find(la==2)), f2(find(la==2)));

%Conditional probabilities
cp11 = (1/(2*pi*det(sig1)^0.5))*exp(-0.5.*(x-M1).'.*inv(sig1).*(x-M1));
cp12 = (1/(2*pi*det(sig2)^0.5))*exp(-0.5.*(x-M2).'.*inv(sig2).*(x-M2));

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

%Posterior prob. for the test feature

px = cp11.*Pr1+cp12.*Pr2; %Calculate p(x) using eq(2)

pos11= cp11*Pr1/px  % p(w1/x) for the given test feature value

pos12= cp12*Pr2/px % p(w2/x) for the given test feature value

posteriors_x=[pos11(1,1)*pos11(2,2), pos12(1,1)*pos12(2,2)];

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

g_x  = pos11(1,1)*pos11(2,2)-pos12(1,1)*pos12(2,2)
%pos11(1,1)*pos11(2,2)-pos12(1,1)*pos12(2,2)% compute the g(x) for min err rate classifier.

