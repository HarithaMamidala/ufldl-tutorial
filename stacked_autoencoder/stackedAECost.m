function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));

n = numel(stack);
%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

X = data;

% Feed forward
a = cell(n + 1);
z = cell(n + 1);
a{1} = X;
for d = 1:n
    z{d+1} = stack{d}.w * a{d} + repmat(stack{d}.b,1,m);
    a{d+1} = sigmoid(z{d+1});
end

M = softmaxTheta * a{n + 1};
M = bsxfun(@minus, M, max(M, [], 1));
P = bsxfun(@rdivide, exp(M), sum(exp(M))); % P(j,i) = p(y(i) = j|x(i);theta) 10 100

cost = -mean(sum(groundTruth .* log(P))) + (lambda/2)*sum(sum(softmaxTheta .^ 2));

softmaxThetaGrad = -(1/m) .* ((groundTruth - P) *  a{n+1}') + lambda * softmaxTheta;

% Backpropagation
delta = cell(n + 1);
delta{n + 1} = -(softmaxTheta' * (groundTruth - P)) .* sigmoidGradient(z{n+ 1});

for l = (n:-1:2)
    delta{l} = stack{l}.w' * delta{l+1} .* sigmoidGradient(z{l});
end

for l = (n:-1:1)
    stackgrad{l}.w = delta{l+1} * a{l}' / m;
    stackgrad{l}.b = sum(delta{l+1}, 2) / m;
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

function debug(x)
    disp(size(x));
end