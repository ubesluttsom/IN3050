from part2 import *

if __name__ == "__main__":

  X_train, X_val, X_test  = standardize(X_train, X_val, X_test)


  # BINARY CLASSIFICATION

  cl1 = linear_regression_classifier()
  cl2 = logistic_regression_classifier()
  cl3 = mmn(η=0.03, dim_hidden=3)

  cl1.fit(X_train, t2_train, η=0.07, val=(X_val, t2_val))
  cl2.fit(X_train, t2_train, η=0.7, val=(X_val, t2_val))
  cl3.fit(X_train, t2_train, val=(X_val, t2_val))

  # All classifiers seem about equivalent, really. No inferior results from
  # test data either.

  print("==> Binary class")
  print("Classifier: Training, Validation, Test")
  print(f"Lin.reg. classifier accuracy: {cl1.accuracy(X_train, t2_train)}," +
                                     f" {cl1.accuracy(X_val,   t2_val  )}," +
                                     f" {cl1.accuracy(X_test,  t2_test )}")
  print(f"Log.reg. classifier accuracy: {cl2.accuracy(X_train, t2_train)}," +
                                     f" {cl2.accuracy(X_val,   t2_val  )}," +
                                     f" {cl2.accuracy(X_test,  t2_test )}")
  print(f"MMN classifier accuracy:      {cl3.accuracy(X_train, t2_train)}," +
                                     f" {cl3.accuracy(X_val,   t2_val  )}," +
                                     f" {cl3.accuracy(X_test,  t2_test )}")

  cl1.plot_decision_regions(X_test, t2_test)
  cl2.plot_decision_regions(X_test, t2_test)
  cl3.plot_decision_regions(X_test, t2_test)


  # MULTI CLASSIFICATION

  cl4 = one_vs_rest()
  cl5 = mmn(η=0.03, dim_hidden=3)

  cl4.fit(X_train, t_train, η=2, val=(X_val, t_val))
  cl5.fit(X_train, t_train, val=(X_val, t_val))

  print("==> Multi class")
  print("Classifier: Training, Validation, Test")
  print(f"One vs. rest classifier accuracy: {cl4.accuracy(X_train, t_train)}," +
                                         f" {cl4.accuracy(X_val,   t_val  )},"   +
                                         f" {cl4.accuracy(X_test,  t_test )}")
  print(f"MMN classifier accuracy:          {cl5.accuracy(X_train, t_train)}," +
                                         f" {cl5.accuracy(X_val,   t_val  )},"   +
                                         f" {cl5.accuracy(X_test,  t_test )}")

  # Here, for some reason, the One vs. rest classifier is not in its element.
  # Which is strange considering I had much better results in Part 1. I suspect
  # it doesn't like the data standardisation, or something. Significantly
  # increasing η fixes this. It seems different scaling requires different
  # tuning.

  cl4.plot_decision_regions(X_test, t_test)
  cl5.plot_decision_regions(X_test, t_test)

