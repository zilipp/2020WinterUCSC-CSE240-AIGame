import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_moons

np.random.seed(42)
data, labels = make_moons(n_samples=500, noise=0.1)
colors = ['r' if y else 'b' for y in labels]
print('data.shape =', data.shape,',  labels.shape =', labels.shape)
plt.scatter(data[:,0], data[:,1], c=colors)
# plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(x, y, learning_rate, num_steps=40):
    ''' Input:  x = the data
                y = the labels
                learning_rate = learning rate
                num_steps = number of iterations
        Output: w = the trained model weights    '''

    # Start by intializing the weights w with w_i = 1 for all i, and make it a 3x1
    # column vector (numpy array).  You can use the numpy function, ones.
    # YOUR CODE HERE
    w = np.ones((np.size(x, 1) + 1, 1))  # MODIFY THIS LINE!

    # Augment x with an initial column of ones for the bias term (the zeroth column of x).
    # You can use the numpy functions ones and hstack to accomplish this.
    # YOUR CODE HERE
    bias = np.ones((np.size(x, 0), 1))
    x = np.column_stack((bias, x))

    # Set z equal to the dot product of x and w.
    # YOUR CODE HERE
    z = np.dot(x, w)

    # Set h equal to the sigmoid function of z.
    # YOUR CODE HERE
    h = sigmoid(z)

    # Compute the initial accuracy, by finding the proportion of times your model's prediction
    # agrees with the labeled values y. Note that you'll have to use the numpy round function
    # to turn your h values into actual predictions.
    accuracy = np.sum(np.round(h, decimals=0) == y) / np.size(y, 0)
    print('Intial Accuracy: ', accuracy)

    for step in range(num_steps):
        # Set z equal to the dot product of x and W.
        # YOUR CODE HERE
        z = np.dot(x, w)

        # Set h equal to the sigmoid function of z.
        # YOUR CODE HERE
        h = sigmoid(z)

        # Calculate the negative of the gradient.
        neg_gradient = np.dot(x.transpose(), (y - h) * h * (1 - h))

        # Update weights.
        # Increase each weight by the corresponding learning_rate * neg_gradient.
        # YOUR CODE HERE
        w = w + learning_rate * neg_gradient

        # Compute the accuracy for this iteration, by finding the proportion of times your
        # model's prediction agrees with the label values y. Note that you'll have to use the
        # numpy round function to turn your h values into actual predictions.
        accuracy = np.sum(np.round(h, decimals=0) == y) / np.size(x, 0)
        print('Step', step + 1, ' Accuracy: ', accuracy)
    return w


ws = logistic_regression(data, labels.reshape((len(labels), 1)), 0.5)
bias = np.ones((np.size(data, 0), 1))
data = np.column_stack((bias, data))

# Set z equal to the dot product of x and W.
z = np.dot(data, ws)  # YOUR CODE HERE

# Set h equal to the sigmoid function of z.
h = sigmoid(z)  # YOUR CODE HERE

# Set y equal to your model's predictions.
# YOUR CODE HERE
# y = np.ones((len(labels), 1))   # YOUR CODE HERE
# y = B = np.where(h > 0.5, 1, 0)
y = np.round(h, decimals=0)

# Plot the correct classifications in green, and the classification errors in red.
colors = ['g' if h == y else 'r' for h, y in zip(y, labels.astype(np.int))]
plt.title('Classification Results')
plt.scatter(data[:, 1], data[:, 2], c=colors)
# plt.show()

coeffs = np.asarray(ws.transpose()).tolist()[0]  # the learned logistic regression coefficients
xvals = [-1.2, 2.2]

# Weirdly, plotting the decision boundary, it's vertically off by about 0.135.  This seems to be a bug
# in matplotlib, since it's off by this amount even changing the seed used to generate the data.
yvals = [-coeffs[1]/coeffs[2] * xval - coeffs[0]/coeffs[2]  for xval in xvals]
plt.plot([xvals[0], xvals[1]], [yvals[0], yvals[1]], 'b-')
plt.show()