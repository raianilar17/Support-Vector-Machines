function visualizeBoundaryLinear(X, y, model)
%VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
%SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
%   learned by the SVM and overlays the data on it

%% HYPOTHESIS h_θ(X) = w_1 * x_1 +  w_2 * x_2 + b;
% x_2 =  -(w_1 * x_1 + b)/w_2;

w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100); % choose min - max value of x1;
yp = - (w(1)*xp + b)/w(2);
figure;
plotData(X, y);
hold on;
plot(xp, yp, '-b'); 
hold off

end
