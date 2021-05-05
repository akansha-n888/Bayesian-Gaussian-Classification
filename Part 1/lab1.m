%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posteriors_x,g_x]=lab1(x,Training_Data,feature)

% x = individual sample to be tested (to identify its probable class label)
% featureOfInterest = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posterior_x  = Posterior probabilities
% g_x = value of the discriminant function

D = Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[M,N]=size(D);    
 
%Col 2 was used to analyze Sepal Width
f=D(:,feature);  % feature samples
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hint: use the commands "find" and "length"

% Prior probabilities
Pr1 = length(find(la==1))/length(la)
Pr2 = length(find(la==2))/length(la)

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%
%Based off Sepal Width the mean and std dev. was calculated for each class

% Mean & Std for class 1 & 2
m11 = mean(f(find(la==1))) ;    % mean of the class conditional density p(x2/w1)
std11 = std(f(find(la==1)));    % Standard deviation of the class conditional density p(x2/w1)

m12  = mean(f(find(la==2)));    % mean of the class conditional density p(x2/w2)
std12 = std(f(find(la==2)));    % Standard deviation of the class conditional density p(x2/w2)

disp(['Conditional probabilities for x=' num2str(x)]);

cp11= (1/(sqrt(2*pi*std11)))*exp(-.5*((x-m11)/std11)^2);% use the above mean, std and the test feature to calculate p(x/w1)

cp12= (1/(sqrt(2*pi*std12)))*exp(-.5*((x-m12)/std12)^2);% use the above mean, std and the test feature to calculate p(x/w2)

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

%Posterior prob. for the test feature

px = cp11*Pr1+cp12*Pr2; %Calculate p(x) using eq(2)

pos11= cp11*Pr1/px ; % p(w1/x) for the given test feature value

pos12= cp12*Pr2/px ; % p(w2/x) for the given test feature value

posteriors_x=[pos11,pos12];

%% %%%%%%Discriminant function for min error rate classifier%%%

%Discriminant function for the test feature

[~,I]=max(posteriors_x);

g_x = pos11-pos12  % compute the g(x) for min err rate classifier.