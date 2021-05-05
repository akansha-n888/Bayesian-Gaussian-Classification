%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 1, Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    1. sepal length in cm
%    2. sepal width in cm
%    3. petal length in cm
%    4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3
%% this script will run lab1 experiments..
clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% feature distribution of x1 for two classes
figure
   
subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),2),100), title('Iris Setosa, sepal width (cm)');
subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),2),100); title('Iris Veriscolour, sepal width (cm)');

%% feature distribution for length
figure

subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),1),100), title('Iris Setosa, sepal length (cm)');
subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),1),100); title('Iris Veriscolour, sepal length (cm)');
 
%% plot for two classes wrt to length and width
figure

plot(irisdata_features(find(numericLabels(:)==1),1),irisdata_features(find(numericLabels(:)==1),2),'rs'); title('x_1 vs x_2');
hold on;
plot(irisdata_features(find(numericLabels(:)==2),1),irisdata_features(find(numericLabels(:)==2),2),'k.');
axis([4 7 1 5]);

%% build training data set for two class comparison
% merge feature samples with numeric labels for two class comparison (Iris
% Setosa vs. Iris Veriscolour
trainingSet = [irisdata_features(1:100,:) numericLabels(1:100,1)];
%% Lab1 experiments (include here)

x1 = [3.3 4.4 5 5.7 6.3];

for i = 1:length(x1)
    feature = 2; 
   [posteriors_x,g_x]=lab1(x1(i),trainingSet,feature);
end

%% data in training set
f1=trainingSet(:,2);  % feature samples (width)
f2=trainingSet(:,1);  % feature samples (length)
la=trainingSet(:,5); % class labels

%% mean & standard deviation
m11 = mean(f1(find(la==1))) ;    % mean of the class conditional density p(x2/w1)
std11 = std(f1(find(la==1))) ;   % Standard deviation of the class conditional density p(x2/w1)

m12  = mean(f1(find(la==2))) ;   % mean of the class conditional density p(x2/w2)
std12 = std(f1(find(la==2))) ;   % Standard deviation of the class conditional density p(x2/w2)

%% threshold for x1
syms th
% 0.7 and 1.3 as multipliers
px = (1/(sqrt(2*pi*std11)))*exp(-.5*((th-m11)/std11)^2)-(1/(sqrt(2*pi*std12)))*exp(-.5*((th-m12)/std12)^2); %since p(w1)=p(w2) it cancels on each eq
Th1 = solve(px,th);
Th1 = double(Th1);
Th1 = (Th1(find(Th1>0)));

errorTh1w2 = (length(f1(find(la==2&f1>Th1))))/length(f1(find(la==1|la==2)));
disp(['The optimal Th for x1 is: ' num2str(Th1)])

%% mean & standard deviation

m11 = mean(f2(find(la==1))) ;    % mean of the class conditional density p(x1/w1)
std11 = std(f2(find(la==1)));    % Standard deviation of the class conditional density p(x1/w1)

m12  = mean(f2(find(la==2)));    % mean of the class conditional density p(x1/w2)
std12 = std(f2(find(la==2)));    % Standard deviation of the class conditional density p(x1/w2)

%% Threshold for x2
syms th
px = (1/(sqrt(2*pi*std11)))*exp(-.5*((th-m11)/std11)^2)-(1/(sqrt(2*pi*std12)))*exp(-.5*((th-m12)/std12)^2); %since p(w1)=p(w2) it cancels on each eq
Th2 = solve(px,th);
Th2 = double(Th2);
Th2 = (Th2(find(Th2>4)));

disp(['The optimal Th for x2 is: ' num2str(Th2)])

%% plot with decision boundaries for features length and width
figure

plot(irisdata_features(find(numericLabels(:)==1),1),irisdata_features(find(numericLabels(:)==1),2),'rs'); title('Th comparison for x_1 vs x_2');
hold on;
plot(irisdata_features(find(numericLabels(:)==2),1),irisdata_features(find(numericLabels(:)==2),2),'k.');
xline(Th2);
yline(Th1);
xlabel('x_1')
ylabel('x_2')
axis([4 7 1 5]);

%% errors assciated with Threshold 1 & 2
errorX1 = (length(f1(find(la==1&f1<Th1)))+length(f1(find(la==2&f1>Th1))))/length(f1(find(la==1|la==2)));
errorX2 = (length(f2(find(la==1&f2>Th2)))+length(f2(find(la==2&f2<Th2))))/length(f2(find(la==1|la==2)));

%% Lab 1 Part 2

x1 = [[2 6]; [4.4 3]; [5 3.5]; [5.3 2]; [5.5 2.5]; [6.6 3.5]; [4.5 6.1]]

figure
plot(irisdata_features(find(numericLabels(:)==1),1),irisdata_features(find(numericLabels(:)==1),2),'rs'); title('Th comparison for x_1 vs x_2');
hold on;
plot(irisdata_features(find(numericLabels(:)==2),1),irisdata_features(find(numericLabels(:)==2),2),'k.');
hold on
plot(x1(:,1),x1(:,2),'b*');
xlabel('x_1')
ylabel('x_2')
axis([2 7.5 1 6.5]);

for i = 1:length(x1)
   [posteriors_x,a]=lab1(x1(i,1),trainingSet,1);
   [posteriors_y,b]=lab1(x1(i,2),trainingSet,2);
   disp(x1(i,:))
   disp(['The g(x) = ' num2str(posteriors_x(1,1)+posteriors_y(1,1))])
   disp(['The g(y) = ' num2str(posteriors_x(1,2)+posteriors_y(1,2))])
   g_xy = a+b
end