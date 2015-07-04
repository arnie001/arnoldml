function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%calculating cost.

a1 = X; % 5000 samples, 401 features including 1 bias.
a1 = [ones(m, 1) a1];


%Theta1 is 25, 401
z2 = Theta1 * a1'; % 25, 401 * 401 * 5000
a2 = sigmoid(z2); % 25 , 5000
a2 = a2'; % 5000, 25

%Theta2 is 10, 26
a2_size = size(a2,1);
a2 = [ones(a2_size,1) a2]; %5000, 26
z3 = Theta2 * a2'; % 10,26 * 26,5000
a3 = sigmoid(z3);  % 10, 5000
hx = a3;

%hx = sigmoid(X*theta); %m,1
% ned to recode y here into a 10 wide vector for each point.
 y_wide = zeros(size(y,1), num_labels);
for c = 1:num_labels
    y_wide(:,c) = (y==c);
endfor
y_wide;
hx';

J = -  (y_wide.* (log (hx'))) - ((1- y_wide).*log(1-hx'));
J = sum(sum(J));
J = J/m;

Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;
Theta1_reg = Theta1_reg .^ 2;
Theta2_reg = Theta2_reg .^ 2;

J_reg_add = sum(sum(Theta1_reg)) + sum(sum(Theta2_reg));
J_reg_add = J_reg_add * lambda/ (2 * m);
J += J_reg_add;

d3 = a3 - y_wide'; %10 x 5000 (classes into samples)
Theta2_no_bias = Theta2(:,2:end);
d2_fac = Theta2_no_bias' * d3; % (10x25)' * 10*5000 = 26x5000
d2_fac = d2_fac'; % 5000x 25
d2 =  d2_fac.* sigmoidGradient(z2'); %5000x25

a2_orig = a2(:, 2:size(a2,2));%5000x26
Theta1_grad = d2' * a1;
Theta2_grad = d3 * a2;

Theta1_grad /= m;
Theta2_grad /= m;

Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;

Theta1_grad += (lambda/m) * Theta1_reg;
Theta2_grad += (lambda/m) * Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
