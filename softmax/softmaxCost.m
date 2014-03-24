function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1)); % 10 100

cost = 0;

thetagrad = zeros(numClasses, inputSize); % 10 8 

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


% disp(size(theta)) => 10 8
% disp(size(data)) => 8 100
% disp(size(labels)) => 100 1

%exp(theta * data); % 10 100
%sum(exp(theta * data)); % Calculate the denominator sum in a vector. 
%repmat(sum(exp(theta * data)),size(theta,1), 1); % Copy the values for the
%division

%log(exp(theta * data) ./ repmat(sum(exp(theta * data)),size(theta,1), 1)); 10 100


% I didn't saw the implementation tips. Reimplementing below
%cost = -mean(sum(groundTruth .* log(exp(M) ./ repmat(sum(exp(M)),size(theta,1), 1)))) + (lambda/2)*sum(sum(theta .^ 2));


M = theta * data;
M = bsxfun(@minus, M, max(M, [], 1));
P = bsxfun(@rdivide, exp(M), sum(exp(M))); % P(j,i) = p(y(i) = j|x(i);theta) 10 100

cost = -mean(sum(groundTruth .* log(P))) + (lambda/2)*sum(sum(theta .^ 2));
thetagrad = -(1/numCases) .* ((groundTruth - P) * data') + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

