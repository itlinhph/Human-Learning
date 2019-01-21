import numpy as np

def cost_function(x):
    return x**2 + 5*np.sin(x)

def grad(x):
    return 2*x + 5*np.cos(x)

def test_gradient(x0, learning_rate):
    x = [x0]
    for item in range(100):
        x_new = x[-1] - learning_rate*grad(x[-1])
        print("Iteration ", item,"\tGrad: ",round(grad(x[-1]),3), "\tx= ", round(x_new,3), "\tCost: ", round(cost_function(x_new),3) )
        if(abs(grad(x_new)) < 0.001):
            break
        x.append(x_new)
    return (x, item)

def main():
    (x1, item1) = test_gradient(-5, 0.1)
    print("==>Iteration ", item1, ": x= ", x1[-1], "Cost: ", cost_function(x1[-1]) )


main()