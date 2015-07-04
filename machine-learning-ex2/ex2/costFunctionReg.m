function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% x is m,3 y is m,1 theta is 3,1
%size(X)
%size(y)
hx = sigmoid(X*theta); %m,1
J = -  (y'*log (hx)) - ((1-y')*log(1-hx));
J = J/m;

%make theta0 as zero as that wont be regularized
theta_temp = theta;
theta_temp(1) = 0;

J_reg = sum(theta_temp.^2);
J_reg = J_reg *(lambda / (2.0*m));
J_reg += J;
J = J_reg;

%grad is 3,1
grad = X'*(hx-y);
grad = grad / m;


grad+= (lambda/m) * theta_temp;

% =============================================================

end
