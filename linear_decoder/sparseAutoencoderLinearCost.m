function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
% cost = 0;
% W1grad = zeros(size(W1)); 
% W2grad = zeros(size(W2));
% b1grad = zeros(size(b1)); 
% b2grad = zeros(size(b2));


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% I like to call it X, so that it's closer to the mathematical formulation 
X = data;

% How many training points we have
m = size(data,2);

% Feed forward
z2 = W1*X  + repmat(b1,1,m); 
a2 = sigmoid(z2);

z3 = W2*a2  + repmat(b2,1,m); 
a3 = z3;

% Compute the sparsity vector
rho = mean(a2,2);

% Backpropagation algorithm
delta3 = -(X - a3);
delta2 = ((W2'*delta3) + repmat(beta*(-sparsityParam ./ rho + (1-sparsityParam) ./(1-rho)),1,m)) .* sigmoidGradient(z2);

Delta_W1 = delta2*X';
Delta_W2 = delta3*a2';
Delta_b1 = sum(delta2,2);
Delta_b2 = sum(delta3,2);

W1grad = (1/m)*Delta_W1 + lambda*W1;
W2grad = (1/m)*Delta_W2 + lambda*W2;
b1grad = (1/m)*Delta_b1;
b2grad = (1/m)*Delta_b2;

% Compute the cost
cost = (1/(2*m))*sum(sum((a3 - X) .^ 2)) + (lambda/2)*(sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2))) + beta*sum(sparsityParam*log(sparsityParam ./ rho) + (1-sparsityParam)*log((1-sparsityParam) ./(1-rho)));

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
