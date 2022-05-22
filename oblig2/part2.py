from part1 import *

rng = np.random.RandomState()

class mmn(classifier):

  def __init__(self, η=0.01, dim_hidden=3):
    self.η = η
    self.dim_hidden = dim_hidden

  def initialize_data(self, X, t):
    t = self.one_hot_encode(t)
    # Find and set dimensions in and out.
    self.dim_in, self.dim_out = (X.shape[1], t.shape[1])
    return (self.with_bias(X), t)

  def initialize_weights(self, X):
    f = lambda x, y: (rng.rand(x+1, y) * 2 - 1) / np.sqrt(x)
    self.weights1 = f(self.dim_in, self.dim_hidden)
    self.weights2 = f(self.dim_hidden, self.dim_out)

  def update_weights(self, X, t, η):
    # Getting the dimensions here was awful. I'm particularly unsure about the
    # `weights2[1:]`, which means ignore the first row of weights; it has
    # something to do with the bias term, I think ...
    hidden, output = self.forward(X, hidden=True)
    δ_o = (output - t) * output * (1 - output)
    δ_h = hidden * (1 - hidden) * (δ_o @ self.weights2[1:].T)  
    self.weights2 -= η * with_bias(hidden).T  @ δ_o
    self.weights1 -= η * with_bias(X_train).T @ δ_h

  def forward(self, X, hidden=False):
    """Calculate activations for hidden and output layers. By default, only
    return output activation, not hidden layer."""
    a_hidden = self.logistic(X @ self.weights1)
    a_output = self.logistic(with_bias(a_hidden) @ self.weights2)
    return (a_hidden, a_output) if hidden else a_output

  def loss(self, X, t):
    return self.cross_entropy(X, t)

  def predict(self, X):
    """Modify predict function for one-hot encoding."""
    return self.one_hot_decode(super().predict(X))

  def one_hot_encode(self, t):
    """One-hot encode a list of integer encoded targets."""
    return np.eye(len(np.unique(t)))[t]

  def one_hot_decode(self, Y):
    return np.argmax(Y, axis=1)

  def report(self, X, t):
    """Stores statistics from validation set."""
    self.val_accu.append(self.accuracy(X[:,1:], self.one_hot_decode(t)))
    self.val_loss.append(self.loss(X, t))

def standardize(train, val, test):
  """Standardize training, validation and test data, return as tuple."""
  avg = train.mean(axis=0)
  std = train.std(axis=0)
  return ((train - avg)/std, (val - avg)/std, (test - avg)/std)

if __name__ == '__main__':

  # MULTI CLASS

  X_train, X_val, X_test  = standardize(X_train, X_val, X_test)

  """
  Since I've implemented some kind of rudimentary early stopping, the
  `epochs` parameter doesn't really matter if the other parameters are tuned;
  it just sets an upper limit (default 1000, in my implementation).
  
  In general, increasing `dim_hidden` and lowering `η` makes a more accurate
  classifier, but is computationally expensive. Lowering `η` requires more epochs.
  
  After some testing, I've outlined two options, both always giving 1.0
  accuracy. Here is some data after training 1000 classifiers with these
  parameters. I didn't time them, but option 2 seemed faster to train.
  
  Option 1: test_mmn(η=0.02, dim_hidden=5, n=1000)
    Validation accuracy mean: 1.0
    Validation accuracy std.: 0.0
    Validation epochs mean: 55.581
    Validation epochs std.: 7.1188088188966
    Validation epochs max:  93
  
  Option 2: test_mmn(η=0.03, dim_hidden=3, n=1000)
    Validation accuracy mean: 1.0
    Validation accuracy std.: 0.0
    Validation epochs mean: 62.823
    Validation epochs std.: 21.812328417663256
    Validation epochs max:  221
  """

  def test_mmn(η=0.02, dim_hidden=5, n=10, binary=False):
    cls = list()
    acc = list()
    epochs = list()
    for i in range(n):
      cl = mmn(η=η, dim_hidden=dim_hidden)
      if not binary:
        cl.fit(X_train, t_train, val=(X_val, t_val))
      else:
        cl.fit(X_train, t2_train, val=(X_val, t2_val))
      cls.append(cl)
      acc.append(cl.val_accu[-1])
      epochs.append(len(cl.val_accu))

    print(f"test_mmn(η={η}, dim_hidden={dim_hidden}, n={n}, binary={binary})")
    print(f"Validation accuracy mean: {np.mean(acc)}")
    print(f"Validation accuracy std.: {np.std(acc)}")
    print(f"Validation epochs mean: {np.mean(epochs)}")
    print(f"Validation epochs std.: {np.std(epochs)}")
    print(f"Validation epochs max:  {np.max(epochs)}")

    for cl in cls:
      plt.plot(cl.val_accu, color='b', alpha=0.2)
    plt.title("MMN validation accuracy")
    plt.show()

    for cl in cls:
      plt.plot(cl.val_loss, color='b', alpha=0.2)
    plt.title("MMN validation loss")
    plt.show()

    # The boundaries of the decision regions are curved, appearing more organic
    # compared to the regression classifiers. And there seem to be a fallback
    # class (class 0, blue in my plot).

    cl.plot_decision_regions(X_val, t_val)

  print("==> Multi class")
  test_mmn()


  # BINARY CLASS

  # Binary classification seems to work just fine without additional tuning.

  print("==> Binary class")
  test_mmn(binary=True)
