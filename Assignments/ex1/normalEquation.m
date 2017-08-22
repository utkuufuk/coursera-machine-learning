function theta = normalEquation(X, y)

    theta = pinv(transpose(X) * X) * transpose(X) * y;    
end
