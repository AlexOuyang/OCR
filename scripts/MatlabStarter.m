% ###### MATLAB STARTER CODE ######
% # RUN YOUR DATA ON ONE OF THE TWO EXAMPLE PIPELINES BELOW, DEPENDING ON WHETHER 
% # YOU WANT TO DO SUPERVISED (regression) OR UNSUPERVISED LEARNING (kmeans)
% #
% # things you need to do (ONLY ONE PER TEAM):
% #  - import your data into matlab and name the variables appropriately
% #    (as X or X and Y)
% #  - run one of the two cells below (lin reg or kmeans)
% #  - produce the plots and change the x/y axis labels and title to reflect your data
% #  - email me your plot and a short description of what the data you plotted represent
% #       i.e. X is height of all elephants I measured, and Y is their age, etc. or
% #       X[0] is the height of all elephants I measured, and X[1] is their weight
% #       
% #       NOTE: if you're working with images, X would represent the value
% #       of the same pixel across all the images, etc
% # 
% # the computations perform themselves, the point of this is to check in and see that
% # you've been able to enter your data at the very least

%% LINEAR REGRESSION
% # ENTER YOUR OWN DATA HERE:
% # if you're doing supervised learning, X should be the values of a 
% # single dimension (feature) for all your data points, and Y should
% # be your prediction (output) value
% #
data = dlmread('scaledfaithful.txt');
X = data(:,1);
Y = data(:,2);
% #__________________________

% # only want the first 100 examples
X = X(1:100);
Y = Y(1:100);

% % # we append 1's to the data vector because linear regression requires an offset
featX = [ones(100,1) X];
% 
% # get line of best fit (offset & slope) by running least squares regression
% '\' operator performs matrix division
theta = Y\featX;

% % # plotting
figure
plot(X,Y,'o')
line([min(X) max(X)], [min(X) max(X)]*theta(2)+theta(1), 'color','r')

% % #### CHANGE THESE LABELS TO MATCH YOUR DATA
title('Linear Regression')
xlabel('Feature 1')
ylabel('Predictor')

%% K-MEANS
% # ENTER YOUR OWN DATA HERE:
% # if you're doing unsupervised learning, X should be the values of 
% # 2 dimensions (feature) for all your data points
% #
X = dlmread('scaledfaithful.txt');
% #________________________________
 
% # take first 100 points and cluster into numK centroids
% # you can change numK to see how it changes the results
X = X(1:100,:);
numK = 2;
[class, centroids] = kmeans(X,numK);

% % # plotting each class
figure
hold on
for k = 1:numK
    plot(X(class==k,1),X(class==k,2), 'o')
end
hold off

% % #### CHANGE THESE LABELS TO MATCH YOUR DATA
title('K Means')
xlabel('Feature 1')
ylabel('Feature 2')