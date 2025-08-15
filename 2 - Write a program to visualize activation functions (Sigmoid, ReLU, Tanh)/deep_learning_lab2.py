# **Lab 2. Write a program to visualize Activation Functions (Sigmoid ,ReLU, Tanh).**

'''

#**Activation functions->**

Activation functions are crucial components of neural networks that introduce non-linearity into the model, allowing it to learn complex patterns in data. Here are the basic, important points regarding three common activation functions: Sigmoid, Tanh, and ReLU.


###**1. Sigmoid Activation Function ->**


The sigmoid function, also known as the logistic function, squashes any input value into a range between 0 and 1. This makes it particularly useful for binary classification problems where the output can be interpreted as a probability.

Formula: σ(x)=1/(1+e^(-x))

Graph:  The graph is an S-shaped curve that is smooth and continuously differentiable.

Key Points:

* Outputs: The output is always between 0 and 1.
* Pros: Its output can be interpreted as a probability. It is smooth and differentiable everywhere, which is required for gradient-based optimization algorithms like backpropagation.
* Cons: It suffers from the vanishing gradient problem. For very large or very small input values, the function's slope becomes very close to zero. This causes the gradients to become very small, and the network's weights update very slowly, or even stop learning, especially in deep networks. The output is also not zero-centered, which can make training less efficient.



###**2. Tanh (Hyperbolic Tangent) Activation Function ->**

The Tanh function is a rescaled version of the sigmoid function. It maps input values to a range between -1 and 1.

Formula: tanh(x)= (e^(x)-e^(-x))/(e^(x)+e^(-x))


Graph:  The graph is also an S-shaped curve, but it's centered at the origin (0,0).

Key Points:

* Outputs: The output is always between -1 and 1.

* Pros: It is zero-centered, which generally makes optimization and training more efficient than with the sigmoid function. It still introduces non-linearity and is differentiable.

* Cons: Just like the sigmoid, it also suffers from the vanishing gradient problem when the input values are very large or very small.



###**3. ReLU (Rectified Linear Unit) Activation Function ->**

The ReLU function is the most popular and widely used activation function in deep learning. It is a simple piecewise linear function.

Formula: f(x)=max(0,x)

Graph:  The graph is a straight line at y=0 for all negative x values and a line with a slope of 1 for all positive x values.

Key Points:

* Outputs: The output is 0 for negative inputs and the input value itself for positive inputs.

* Pros: It is computationally very efficient to compute. It solves the vanishing gradient problem for positive inputs since the gradient is a constant 1. This allows the network to learn much faster and more effectively in deep architectures.

* Cons: It can suffer from the "dying ReLU" problem. If a neuron's weights are adjusted during training such that the weighted sum of inputs is always negative, the neuron will always output 0, and its gradient will be 0. As a result, it will stop updating its weights and become "dead" for all subsequent inputs.

'''

import numpy as np  #for numerical operations
import matplotlib.pyplot as plt #for creating plots and visualizations

# for sigmoid activation function
def sigmoid(x):
  return 1/(1+np.exp(-x))

# for relu activation function
def relu(x):
  return np.maximum(0,x)

# for tanh activation function
def tanh(x):
  return np.tanh(x)

# NumPy's linspace function to create an array of evenly spaced numbers.
# It starts at -10, ends at 10, and generates a total of 200 points.
x=np.linspace(-10,10,200) # now this array will be used as an input for activation functions

# Output of sigmoid function
y_sigmoid=sigmoid(x)

#Output of relu function
y_relu=relu(x)

# Output of tanh function
y_tanh=tanh(x)

plt.figure(figsize=(12,4))  # This creates a new figure, which is like the entire window or canvas for your plots.
# The figsize argument sets the width to 12 inches and the height to 4 inches, making the figure wide and short.

# for plotting sigmoid activation function
plt.subplot(1,3,1)  # This command divides the figure into a grid of plots. The arguments mean 1 row, 3 columns, and we are currently working on the 1st plot in that grid.
plt.plot(x,y_sigmoid,'b') # This is the main plotting command. It takes the x values and the y_sigmoid values and plots them. The 'b' argument specifies the line color as blue.
plt.title("sigmoid function") # This adds a title to the current subplot
plt.grid(True)  # This adds a grid to the background of the subplot


# for plotting relu activation function
plt.subplot(1,3,2)  # This command divides the figure into a grid of plots. The arguments mean 1 row, 3 columns, and we are currently working on the 2nd plot in that grid.
plt.plot(x,y_relu,'r')  # It takes the x values and the y_relu values and plots them. The 'r' argument specifies the line color as red.
plt.title("relu function")
plt.grid(True)

# for plotting tanh activation function
plt.subplot(1,3,3)  # This command divides the figure into a grid of plots. The arguments mean 1 row, 3 columns, and we are currently working on the 3rd plot in that grid.
plt.plot(x,y_tanh,'g')  # It takes the x values and the y_tanh values and plots them. The 'g' argument specifies the line color as green.
plt.title("tanh function")
plt.grid(True)

plt.tight_layout()  # This command automatically adjusts the spacing between subplots to prevent titles and labels from overlapping.
plt.show()  # This displays the final figure with all three subplots