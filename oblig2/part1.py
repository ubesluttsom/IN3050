import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.datasets import make_blobs

rng = np.random.RandomState()

# DATASETS

X, t = make_blobs(
    n_samples=[400, 400, 400, 400, 400],
    centers=[[0, 1], [4, 1], [8, 1], [2, 0], [6, 0]],
    n_features=2,
    random_state=2019,
    cluster_std=0.1
)

rng.shuffle(indices := np.arange(X.shape[0]))

# Section data into training, validation, and test sets.
X_train = X[indices[:1000],:]
X_val   = X[indices[1000:1500],:]
X_test  = X[indices[1500:],:]
t_train = t[indices[:1000]]
t_val   = t[indices[1000:1500]]
t_test  = t[indices[1500:]]

# Create binary set.
t2_train = (t_train >= 3).astype('int')
t2_val   = (t_val   >= 3).astype('int')
t2_test  = (t_test  >= 3).astype('int')

def with_bias(X):
  """Add bias to arbitrary dataset."""
  return np.concatenate([np.ones((X.shape[0],1)), X], axis=1) 

def plot_datasets():
  plt.figure(figsize=(8,6)) # You may adjust the size
  plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, s=20.0)
  plt.show()

  plt.figure(figsize=(8,6))
  plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=20.0)
  plt.show()


# GENERIC CLASSIFIER

class classifier():

  def __init__(self, η=0.1):
    self.η = η   # Learning rate

  def fit(self, X, t, η=None, epochs=1000, loss_diff=0., val=(None, None)):
    """Train model on `X` and `t` dataset.

    This algorithm is the same for all classifiers, but makes calls to vital
    helper functions (`initialize_data()`, `update_weights()`, etc.) that are
    overridden in in specific classifiers. I.e. the classifier “guts” are
    modular. (Don't repeat yourself.)

    Stop algorithm if the 20 epoch rolling average loss improvement is less
    than or equal `loss_diff`.
    
    If a validation set is passed, do validation testing and store statistics.
    `val` is a tuple of a X and a t.
    """

    # Set the learning rate.
    η = η if η else self.η

    # Boolean, check if we should store stats on validation set.
    if validate := (type(val[0]) is np.ndarray and type(val[0]) is np.ndarray):
      (X_val, t_val) = self.initialize_data(*val)
      self.val_accu = list()
      self.val_loss = list()

    (X, t) = self.initialize_data(X, t)
    self.initialize_weights(X)
    self.train_loss = list()

    for e in range(epochs):
      self.update_weights(X, t, η)
      self.train_loss.append(self.loss(X, t))
      if validate: 
        self.report(X_val, t_val)

      # Calculate 20 epoch rolling average loss improvement, and potentially
      # `break` if less than or equal `loss_diff`. Also if accuracy on
      # validation set is consistently 1.0. I guess this is a form of early
      # stopping.
      if e > 20:
        if abs(np.mean(np.diff(self.train_loss[e-20:e]))) <= loss_diff:
          break
        elif validate and np.mean(self.val_accu[e-20:e]) == 1.0:
          break

  def update_weights(self, X, t, η):
    """Weight update step. Override me!"""
    pass

  def forward(self, X):
    """Forward calculation. Override me!"""
    pass

  def loss(self, X, t):
    """ Loss function. Override me!"""
    pass

  def initialize_data(self, X, t):
    """Initialize data properly. Override me!"""
    pass

  def initialize_weights(self, X):
    """Initialize weights properly. Override me!"""
    pass

  def with_bias(self, X):
    """Add bias to arbitrary dataset."""
    return np.concatenate([np.ones((X.shape[0],1)), X], axis=1) 

  def predict(self, X, threshold=0.5):
    """Predict of the classes of `X` using current model (weights)."""
    return self.forward(self.with_bias(X)) > threshold

  def accuracy(self, X, t, **kwargs):
    """Calculate the accuracy of current model. Return ratio of correctly
    classified `X` compared to targets `t`."""
    predictions = self.predict(X, **kwargs)
    if len(predictions.shape) > 1:
        predictions = predictions[:,0]
    return np.sum(predictions == t) / len(predictions)  

  def report(self, X, t):
    """Stores statistics from validation set."""
    self.val_accu.append(self.accuracy(X[:,1:], t))
    self.val_loss.append(self.loss(X, t))

  # Some mathematical utility functions:

  def logistic(self, x, β=1):
    return 1/(1+np.exp(-β*x))

  def mean_square_error(self, X, t):
    return np.average((self.forward(X) - t)**2)

  def cross_entropy(self, X, t):
    y_hat = self.forward(X)
    return -np.average(t * np.log(y_hat) + (1 - t) * np.log(1 - y_hat))

  # Plotter

  def plot_decision_regions(self, X, t, size=(8,6)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h=0.02   # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=size) # You may adjust this
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='Paired')
    plt.scatter(X[:,0], X[:,1], c=t, s=20.0, cmap='Paired')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()


# BINARY CLASSIFIERS

class linear_regression_classifier(classifier):

  def initialize_data(self, X, t):
    # Add bias where appropriate
    return (self.with_bias(X), t)

  def initialize_weights(self, X):
    """Initialize weights to zeros."""
    self.w = np.zeros(X.shape[1])

  def update_weights(self, X, t, η):
    self.w -= η / X.shape[0] * X.T @ (self.forward(X) - t)

  def forward(self, X):
    return X @ self.w

  def loss(self, X, t):
    """Calculate loss using Mean Square Error"""
    return self.mean_square_error(X, t)

class logistic_regression_classifier(linear_regression_classifier):

  def forward(self, X):
    return self.logistic(X @ self.w)

  def loss(self, X, t):
    """Calculate loss using Cross-Entropy"""
    return self.cross_entropy(X, t)


# MULTI-CLASS CLASSIFIER

class one_vs_rest(logistic_regression_classifier):

  def fit(self, X, t, η=None, epochs=1000, loss_diff=0., val=(None, None)):
    # Boolean, check if we should store stats on validation set.
    validate = (type(val[0]) is np.ndarray and type(val[1]) is np.ndarray)
    # Using dictionary to store classifiers to preserve the proper labels,
    # which are used as keys here. This way, the labels need not be structured
    # integers.
    self.classifiers = {}
    for label in np.unique(t):
      # Split dataset into positive and negative classes (binary).
      t2 = (t == label).astype('int')
      t2_val = (val[1] == label).astype('int') if validate else None
      # Make a classifier for the binary data set, then train it.
      cl = logistic_regression_classifier()
      cl.fit(X, t2, η, epochs, loss_diff,  val=(val[0], t2_val))
      # Store classifier
      self.classifiers[label] = cl

  def predict(self, X):
    # Make nd-array for storing predictions from all classifiers. Also a list
    # of their class labels; this is to remember which index refer to which
    # class.
    predictions = np.empty([len(self.classifiers), len(X)])
    labels = list()
    for label, classifier in self.classifiers.items():
      predictions[len(labels)] = classifier.forward(self.with_bias(X))
      labels.append(label)
    # Using the most likely predictions as indices, choose the respective
    # class from `labels`.
    return np.array(labels)[np.argmax(predictions, axis=0)]


if __name__ == "__main__":
  cl1 = linear_regression_classifier()
  cl2 = logistic_regression_classifier()
  cl3 = one_vs_rest()

  # It seems these parameters are alright for linear regression. At η≥0.08 the
  # algorithm overshoots, oscillating wildly, but at η=0.07 it improves fast.
  # 150-ish epochs seems sufficient, but using 1000 here for testing.

  cl1.fit(X_train, t2_train, η=0.07, epochs=1000, val=(X_val, t2_val))
  print(f"Lin.reg. classifier accuracy: {cl1.accuracy(X_val, t2_val)}")

  # In general, lower η gives a smoother loss and accuracy improvement, but
  # at the cost of speed.

  cl2.fit(X_train, t2_train, η=0.7, epochs=1000, val=(X_val, t2_val))
  print(f"Log.reg. classifier accuracy: {cl2.accuracy(X_val, t2_val)}")

  # For logistic regression I can actually use a ridiculous η of, say, 10, and
  # the algorithm finds the right weights in like 15 epochs! I don't understand
  # why. I get some division by zero warnings, though, and it doesn't scale to
  # the one vs. rest classifier.

  cl3.fit(X_train, t_train, η=0.5, epochs=1000, val=(X_val, t_val))
  print(f"One vs. rest classifier accuracy: {cl3.accuracy(X_val, t_val)}")

  # Plot iterative improvement in the linear and logistic classifiers.
  
  plt.plot(cl1.val_accu)
  plt.plot(cl2.val_accu)
  plt.title("Lin./log. reg. accuracy")
  plt.show()

  plt.plot(cl1.val_loss)
  plt.plot(cl2.val_loss)
  plt.title("Lin./log. reg. loss")
  plt.show()

  # Plot decision regions for all classifiers.

  cl1.plot_decision_regions(X_val, t2_val)
  cl2.plot_decision_regions(X_val, t2_val)
  cl3.plot_decision_regions(X_val, t_val)
