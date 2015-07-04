function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
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

%Note at this point hx already contains he result of evaluating against
% the input. Now we can just check what is the maximum probability's index.
% for each of the 5000 samples.
[val, p] = max(hx);
p = p';
% At this point theta can make each input to probability for each class.
%out = 



% =========================================================================


end
