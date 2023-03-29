import pickle as pickle
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.pyplot import imread

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the MLP classifier.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape the data
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


def softmax(x):
  e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
  return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
  return 1./(1.+np.exp(-x))

class MLP(object):
  """
  A multi-layer fully-connected neural network has an input dimension of
  d, a hidden layer dimension of h, and performs classification over c classes.
  You must train the network with a softmax loss function and L1 regularization on the
  weight matrices. The network uses a ReLU/LeakyReLU/etc nonlinearity after the first fully
  -connected layer.

  The network has the following architecture:

  Input - Linear layer - ReLU/LeakyReLU/etc - Linear layer - Softmax

  The outputs of the network are the labels for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, activation, std=1e-4):
    """
    An initialization function

    Parameters
    ----------
    input_size: integer
        the dimension d of the input data.
    hidden_size: integer
        the number of neurons h in the hidden layer.
    output_size: integer
        the number of classes c.
    activation: string
        activation method name
    std: float
        standard deviation
    """
    # w1: weights for the first linear layer
    # b1: biases for the first linear layer
    # w2: weights for the second linear layer
    # b2: biases for the second linear layer

    self.params = {}
    self.params['w1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['w2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    self.leaky_relu_c = 0.01
    self.selu_lambda  = 1.05
    self.selu_alpha   = 1.67
    self.activation_method = ['ReLU','LeakyReLU','SWISH','SELU'].index(activation)
    print("Selected using "+['ReLU','LeakyReLU','SWISH','SELU'][self.activation_method])


  def forward_pass(self, x, w1, b1, w2, b2):
    """
    A forward pass function

    Returns
    -------
    out:
        network output
    cache:
        intermediate values
    """
    h1     = None  # the activation after the first linear layer
    y1, y2 = None, None  # outputs from the first & second linear layers

    #############################################################################
    # PLACE YOUR CODE HERE                                                      #
    #############################################################################
    # TODO: Design the fully-connected neural network and compute its forward
    #       pass output,
    #        Input - Linear layer - LeakyReLU - Linear layer.
    #       You have use predefined variables above

    #  START OF YOUR CODE
    y1 = np.dot(x, w1) + b1

    if self.activation_method == 0:
      # ReLU
      h1 = np.maximum(0, y1)
    elif self.activation_method == 1:
      # Leaky ReLU
      h1 = np.maximum(self.leaky_relu_c * y1, y1)
    elif self.activation_method == 2:
      # SWISH
      h1 = y1 * sigmoid(y1)
    else:
      # SELU
      h1 = self.selu_lambda * np.where(y1 > 0, y1, self.selu_alpha * np.exp(y1))
      # scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    y2 = np.dot(h1, w2) + b2


    #  END OF YOUR CODE
    #############################################################################

    out  = y2#softmax(y2)
    cache = (y1, h1) # intermediate values

    return out, cache


  def softmax_loss(self, x, y):
    """
    Compute the loss and gradients for a softmax classifier

    Returns
    -------
    loss:
        the softmax loss
    dx:
        the gradient of loss

    """
    #############################################################################
    # PLACE YOUR CODE HERE                                                      #
    #############################################################################
    # TODO: Compute the softmax classification loss and its gradient.           #
    # The softmax loss is also known as cross-entropy loss.                     #

    n_classes = x.shape[1]
    y_one_hot = np.zeros((len(y), n_classes))
    y_one_hot[np.arange(len(y)), y] = 1
    epsilon = 1e-15
    x = np.clip(x, epsilon, 1 - epsilon)

    loss = -np.sum(y_one_hot * np.log(x + epsilon)) / len(x)


    dx = (x - y_one_hot)

    #  END OF YOUR CODE
    #############################################################################

    return loss, dx


  def backward_pass(self, dY2_dLoss, x, w1, y1, h1, w2):
    """
    A backward pass function

    Returns
    -------
    grads:


    """
    grads = {}

    #############################################################################
    # PLACE YOUR CODE HERE                                                      #
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # the gradient on W1 should be stored in grads['w1'] and be a matrix of same#
    # size                                                                      #

    #without regularization

    grads['w2'] = np.dot(h1.T, dY2_dLoss) / x.shape[0]
    grads['b2'] = np.sum(dY2_dLoss, axis=0) / x.shape[0]

    if self.activation_method == 0:
      # ReLU
      # d(Loss)/d(y1) = d(Loss)x/d(y2) * d(y2)/d(h1) * d(h1)/d(y1)
      dy1_dh1 = np.where(y1 > 0, 1, 0)
      dY1_dLoss = np.dot(dY2_dLoss, w2.T) * dy1_dh1

    elif self.activation_method == 1:
      # Leaky ReLU
        dy1_dh1 = np.where(y1 > 0, 1, self.leaky_relu_c)
        dY1_dLoss = np.dot(dY2_dLoss, w2.T) * dy1_dh1
    elif self.activation_method == 2:
      # SWISH
        dy1_dh1 = sigmoid(y1) * (1 - sigmoid(y1)) + sigmoid(y1)
        dY1_dLoss = np.dot(dY2_dLoss, w2.T) * dy1_dh1
    else:
      # SELU

        dy1_dh1 = self.selu_lambda * np.where(y1 > 0, 1, self.selu_alpha * np.exp(y1))
        dY1_dLoss = np.dot(dY2_dLoss, w2.T) * dy1_dh1


    # y1 = w1*x + b1
    # h1 = act(y1)
    # y2 = w2(h1) + b2
    # d(Loss)/d(w1) = d(Loss)x/d(y2) * d(y2)/d(h1) * d(h1)/d(y1) * d(y1)/d(w1)

    grads['w1'] = np.dot(x.T, dY1_dLoss) / x.shape[0]
    grads['b1'] = np.sum(dY1_dLoss, axis=0) / x.shape[0] #dY1_dLoss * 1

    #  END OF YOUR CODE
    #############################################################################

    return grads


  def loss(self, x, y=None, regular=0.0, enable_margin=False):
    """
    A loss function that returns the loss and gradients of the fully-connected
    neural network. This function requires designing forward and backward passes.

    If y is None, it returns a matrix labelsof shape (n, c) where labels[i, c]
    is the label score for class c on input x[i]. Otherwise, it returns a tuple
    of loss and grads.

    Parameters
    ----------
    x:  matrix
        an input data of shape (n, d). Each x[i] is a training sample.
    y:  vector
        a vector of training labels. Each y[i] is an integer in the range
        0 <= y[i] < c. y[i] is the label for x[i]. If it is passed then we
        return the loss and gradients.
    regular: float
        regularization strength.
    enable_margin: Bool
        enable to use soft-margin softmax

    Returns
    -------
    loss:
        Loss (data loss and regularization loss) for this batch of training
        samples.
    grads:
        Dictionary mapping parameter names to gradients of those parameters with
        respect to the loss function; has the same keys as self.params.
    """
    # Variables
    n, d   = x.shape # input dimensions
    w1, b1 = self.params['w1'], self.params['b1']
    w2, b2 = self.params['w2'], self.params['b2']
    h1     = None  # the activation after the first linear layer
    y1, y2 = None, None  # outputs from the first & second linear layers

    # Compute the forward pass
    out, cache = self.forward_pass(x,w1,b1,w2,b2)
    y2       = out
    (y1, h1) = cache

    # If the targets are not given then jump out, we're done
    if y is None:
      return y2

    # Compute the loss

    loss, dY2_dLoss = self.softmax_loss(y2, y)

    # Compute the backward pass
    grads = self.backward_pass(dY2_dLoss, x, w1, y1, h1, w2)

    #############################################################################
    # PLACE YOUR CODE HERE (REGULARIZATION)                                     #
    #############################################################################
    # TODO: Implement weight regularization

    L2_regularization_cost = (np.sum(np.square(self.params["w1"])) + np.sum(np.square(self.params["w2"])))*(regular/(2* y.shape[0]))
    loss += L2_regularization_cost
    # #
    # # #add regularization effect to the gradient terms
    grads['w2'] -= (regular / y.shape[0]) * self.params["w2"]
    grads['w1'] -= (regular / y.shape[0]) * self.params["w1"]

    #  END OF YOUR CODE
    #############################################################################

    return loss, grads


  def train(self, x, y, x_v, y_v,
            eta=1e-3, lamdba=0.95,
            regular=1e-5, num_iters=50,
            batch_size=100, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - x: A numpy array of shape (n, d) giving training data.
    - y: A numpy array f shape (n,) giving training labels; y[i] = C means that
      x[i] has label C, where 0 <= C < c.
    - x_v: A numpy array of shape (n_v, d) giving validation data.
    - y_v: A numpy array of shape (n_v,) giving validation labels.
    - eta: Scalar giving learning rate for optimization.
    - lamdba: Scalar giving factor used to decay the learning rate
      after each epoch.
    - regular: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = x.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use Stochastic Gradient Descent (SGD) to optimize the parameters in
    # self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      x_batch = None
      y_batch = None

      #########################################################################
      # PLACE YOUR CODE HERE                                                  #
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in x_batch and y_batch respectively.                             #

      random_sample = np.random.choice(num_train,batch_size)
      #print(random_sample)
      x_batch = x[random_sample]
      y_batch = y[random_sample]

      #  END OF YOUR CODE
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(x_batch, y=y_batch, regular=regular)
      loss_history.append(loss)

      #########################################################################
      # PLACE YOUR CODE HERE                                                  #
      #########################################################################
      # TODO: Update the parameters of the network stored in self.params by   #
      # using the gradients in the grads dictionary. For that, use stochastic #
      # gradient descent.                                                     #

      self.params['w1'] -= (eta * grads['w1'])
      self.params['w2'] -= (eta * grads['w2'])
      self.params['b1'] -= (eta * grads['b1'])
      self.params['b2'] -= (eta * grads['b2'])

      #  END OF YOUR CODE
      #########################################################################

      #########################################################################
      # PLACE YOUR CODE HERE                                                  #
      #########################################################################
      # For printing out validation acc
      if verbose and it % 100 == 0:
        # get validataion loss
        # print out
        val_out, _ = self.forward_pass(X_val, self.params["w1"], self.params["b1"], self.params["w2"], self.params["b2"])
        val_loss, _ =self.softmax_loss(val_out, y_val)
        val_acc = (self.predict(X_val) == y_val).mean()
        print("validation loss :", val_loss)
        print("validation acc :", val_acc)


      #  END OF YOUR CODE
      #########################################################################

      if verbose and it % 100 == 0:
        print('The #iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % int(iterations_per_epoch) == 0:
        # Check accuracy
        train_acc = (self.predict(x_batch) == y_batch).mean()
        val_acc = (self.predict(x_v) == y_v).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        eta *= lamdba

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }




  def predict(self, x):
    """
    Use the trained weights of this MLP network to predict labels for
    data points. For each data point we predict labels for each of the C
    classes, and assign each data point to the class with the highest label
    score.

    Inputs:
    - x: A numpy array of shape (n, d) giving n d-dimensional data points to
      classify.

    Returns:
    - y_pr: A numpy array of shape (n,) giving predicted labels for each of
      the elements of x. For all i, y_pred[i] = c means that x[i] is predicted
      to have class C, where 0 <= C < c.
    """
    y_pr = None

    ###########################################################################
    # PLACE YOUR CODE HERE                                                    #
    ###########################################################################
    # TODO: Implement the predict function
    # forward_pass(x,w1,b1,w2,b2) #
    out, _ = self.forward_pass(x, self.params["w1"], self.params["b1"], self.params["w2"], self.params["b2"])
    out = softmax(out)
    y_pr = np.argmax(out, axis=1)


    #y_pr =

    # END OF YOUR CODE
    ###########################################################################

    return y_pr

np.random.seed(1)
input_size = 32 * 32 * 3
hidden_size = 64
num_classes = 10
# activation = 'SELU' # Select one in [ReLU, LeakyReLU, SWISH, 'SELU']
# net_mlp = MLP(input_size, hidden_size, num_classes, activation)
#
# # Train the network
# stats = net_mlp.train(X_train, y_train, X_val, y_val,
#             num_iters=1000, batch_size=200,
#             eta=2*1e-3, lamdba=0.95,
#             regular=1.0, verbose=True)
#
#
# # Predict on the validation set
# val_acc = (net_mlp.predict(X_val) == y_val).mean()
# print(stats["train_acc_history"])
# print('Validation accuracy: ', val_acc)
#
# # Plot the loss function and train / validation accuracies
# # I want to show which is the train and which is the validation by the color
#
# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'])
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
#
# # i want to notice plot label color
#
# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.legend()
#
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Classification accuracy')
#
# plt.show()
#
# # visualize the weights of the network
import math
from math import *
def visualize_grid(xs, ubound=255.0, padding=1):
  """
  Reshape an image data of 4D tensor to a grid for the better understanding and visualization.

  Inputs:
  - xs: Data of shape (n, h, w, c)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (n, h, w, c) = xs.shape
  grid_size = int(ceil(sqrt(n)))
  grid_height = h * grid_size + padding * (grid_size - 1)
  grid_width = w * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, c))
  next_idx = 0
  y0, y1 = 0, h
  for y in range(grid_size):
    x0, x1 = 0, w
    for x in range(grid_size):
      if next_idx < n:
        img = xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += w + padding
      x1 += w + padding
    y0 += h + padding
    y1 += h + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid

def show_net_weights(net):
  w1 = net_mlp.params['w1']

  ###########################################################################
  # PLACE YOUR CODE HERE                                                    #
  ###########################################################################
  # TODO: Implement the weight visualization
  xs = w1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  xs_min, xs_max = np.min(xs), np.max(xs)
  xs = 255.0 * (xs - xs_min) / (xs_max - xs_min)

  plt.imshow(visualize_grid(xs, padding=3).astype('uint8'))
  # END OF YOUR CODE
  ###########################################################################

  plt.gca().axis('off')
  plt.show()



#show_net_weights(net_mlp)

# I want to experiment with different activation functions and compare the results which is the best one
# make a plot for the loss function and train / validation accuracies together
# I need to make a function for the activation function
# I want to run all the activation functions in one time
# put all the result in list and plot all the results together


activation_method = ['ReLU','LeakyReLU','SWISH','SELU']

for activation in activation_method:


    net_mlp = MLP(input_size, hidden_size, num_classes, activation)

    # Train the network
    stats = net_mlp.train(X_train, y_train, X_val, y_val,
                num_iters=1000, batch_size=200,
                eta=2*1e-3, lamdba=0.95,
                regular=1.0, verbose=True)


    # Predict on the validation set
    val_acc = (net_mlp.predict(X_val) == y_val).mean()
    print(stats["train_acc_history"])
    print('Validation accuracy: ', val_acc)

    # Plot the loss function and train / validation accuracies
    # I want to show which is the train and which is the validation by the color
    # I want to show the activation function name in the plot

    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'], label=activation)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # i want to notice plot label color

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.legend()

    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.show()




