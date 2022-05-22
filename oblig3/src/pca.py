import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import syntheticdata

mpl.rcParams['axes.facecolor'] = 'e0e0e0'
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['lines.markeredgecolor'] = 'white'
mpl.rcParams['legend.edgecolor'] = 'e0e0e0'
mpl.rcParams['scatter.edgecolors'] = 'white'
mpl.rcParams['image.cmap'] = 'viridis'

# I use LaTeX for nicer figures. Turn this off if you don't have it installed.
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage{sansmathfonts}
    \usepackage[T1]{fontenc}
    \renewcommand*\familydefault{\sfdefault}
"""

def center_data(A):
  return A - np.mean(A, axis=0)

def compute_covariance_matrix(A):
  return np.cov(A.T)

def compute_eigenvalue_eigenvectors(A):
  return (eig := np.linalg.eig(A))[0].real, eig[1].real

def sort_eigenvalue_eigenvectors(eigval, eigvec):
  return eigval[i := np.argsort(eigval)[::-1]], eigvec[:, i]

def pca(A, m):
  A = center_data(A)
  C = compute_covariance_matrix(A)
  eigval, eigvec = compute_eigenvalue_eigenvectors(C)
  eigval, eigvec = sort_eigenvalue_eigenvectors(eigval, eigvec)
  eigvec = eigvec[:,:m]   # Slice `m` dims with lowest variance
  return eigvec, np.dot(eigvec.T, A.T).T

def encode_decode_pca(A, m):
  return ((X := pca(A,m))[0] @ X[1].T).T + np.mean(A, 0)


if __name__ == '__main__':

  # 1.2 IMPLEMENTATION: HOW IS PCA IMPLEMENTED?
  # ===========================================

  # 1.2.1 Centering the Data
  # ------------------------

  testcase = np.array([[3., 11., 4.3],
                       [4., 5.,  4.3],
                       [5., 17., 4.5],
                       [4,  13., 4.4]])
  answer = np.array([[-1., -0.5, -0.075],
                     [0.,  -6.5, -0.075],
                     [1.,  5.5,   0.125],
                     [0.,  1.5,   0.025]])
  try:
    np.testing.assert_array_almost_equal(center_data(testcase), answer)
  except:
    print("`center_data` function fail!")
    exit()
  print("`center_data` function passes!")
  
  # 1.2.3 Computing Covariance Matrix
  # ---------------------------------

  testcase = center_data(np.array([[22., 11., 5.5],
                                   [10., 5.,  2.5],
                                   [34., 17., 8.5],
                                   [28., 14., 7  ]]))
  answer = np.array([[580., 290., 145.],
                     [290., 145., 72.5],
                     [145., 72.5, 36.25]])

  # Depending on implementation the scale can be different:
  to_test = compute_covariance_matrix(testcase)

  answer = answer/answer[0, 0]
  to_test = to_test/to_test[0, 0]

  try:
    np.testing.assert_array_almost_equal(to_test, answer)
  except:
    print("`compute_covariance_matrix` function fail!")
    exit()
  print("`compute_covariance_matrix` function passes!")

  # 1.2.4 Computing Eigenvalues and Eigenvectors
  # --------------------------------------------

  testcase = np.array([[2, 0, 0],
                       [0, 5, 0],
                       [0, 0, 3]])
  answer1 = np.array([2., 5., 3.])
  answer2 = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
  x,y = compute_eigenvalue_eigenvectors(testcase)
  try:
    np.testing.assert_array_almost_equal(x, answer1)
    np.testing.assert_array_almost_equal(y, answer2)
  except:
    print("`compute_eigenvalue_eigenvectors` function fail!")
    exit()
  print("`compute_eigenvalue_eigenvectors` function passes!")

  # 1.2.5 Sorting Eigenvalues and Eigenvectors
  # ------------------------------------------
    
  answer1 = np.array([5., 3., 2.])
  answer2 = np.array([[0., 0., 1.],
                      [1., 0., 0.],
                      [0., 1., 0.]])
  x,y = compute_eigenvalue_eigenvectors(testcase)
  x,y = sort_eigenvalue_eigenvectors(x,y)
  try:
    np.testing.assert_array_almost_equal(x, answer1)
    np.testing.assert_array_almost_equal(y, answer2)
  except:
    print("`sort_eigenvalue_eigenvectors` function fail!")
    exit()
  print("`sort_eigenvalue_eigenvectors` function passes!")

  # 1.2.6 PCA Algorithm
  # -------------------

  testcase = np.array([[22.,11.,5.5],[10.,5.,2.5],[34.,17.,8.5]])
  x,y = pca(testcase,2)

  import pickle
  answer1_file = open('PCAanswer1.pkl','rb')
  answer2_file = open('PCAanswer2.pkl','rb')
  answer1 = pickle.load(answer1_file)
  answer2 = pickle.load(answer2_file)

  test_arr_x = np.sum(np.abs(np.abs(x) - np.abs(answer1)), axis=0)
  test_arr_y = np.sum(np.abs(np.abs(y) - np.abs(answer2)))
  try:
    np.testing.assert_array_almost_equal(test_arr_x, np.zeros(2))
    np.testing.assert_almost_equal(test_arr_y, 0)
  except:
    print("`pca` function fail!")
    exit()
  print("`pca` function passes!")


  # 1.3 UNDERSTANDING: HOW DOES PCA WORK?
  # =====================================

  # 1.3.1 Loading the Data
  # ----------------------

  X = syntheticdata.get_synthetic_data1()

  # 1.3.2 Visualizing the Data
  # --------------------------

  fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
  colormap = mpl.cm.get_cmap(lut=3)(range(3))
  ax1.scatter(X[:,0],X[:,1], marker='.', color=colormap[0], alpha=.5,
              ec='white', label='Original data')

  # 1.3.3 Visualizing the Centered Data
  # -----------------------------------

  X = center_data(X)
  ax1.scatter(X[:,0], X[:,1], color=colormap[1], alpha=.5,
              ec='white', label='Centered data')

  # 1.3.4 Visualizing the Centered Data
  # -----------------------------------

  pca_eigvec, P = pca(X, 2)
  first_eigvec = pca_eigvec[0]

  x = np.linspace(-3, 3, 1000)
  y = first_eigvec[1]/first_eigvec[0] * x
  ax1.plot(x, y, color=colormap[1],
           path_effects=[patheffects.withStroke(linewidth=3, foreground="w")],
           ls='--', label='Comp. slope')
  ax1.legend()

  # 1.3.5 Visualize the PCA Projection
  # ----------------------------------

  pca_eigvec, P = pca(X, 1)
  ax2.eventplot(P[:,0], color=colormap[1], alpha=.7)
  ax2.set_yticks([])
  ax2.legend(title='Projected 1-dim data')

  fig.savefig('fig135.pdf')
  fig.suptitle('Visualizing PCA')
  fig.show()


  # 1.4 EVALUATION: WHEN ARE THE RESULTS OF PCA SENSIBLE?
  # =====================================================

  # 1.4.1 Loading the first set of labels
  # -------------------------------------

  X, y = syntheticdata.get_synthetic_data_with_labels1()
  y = y.astype(int)[:,0]

  # 1.4.2 Running PCA
  # -----------------

  fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
  pca_eigvec, P = pca(X, 1)
  ax1.scatter(X[:,0], X[:,1], c=colormap[y], alpha=.7)
  for i in range(2):
    ax2.eventplot(P[:,0][y==i], color=colormap[i], alpha=.7)
  ax2.set_yticks([])
  ax1.legend(title='Original data')
  ax2.legend(title='Projected data')
  fig.savefig('fig142.pdf')
  fig.suptitle('Visualizing PCA with labeled data (1)')
  fig.show()
  
  # The data is now centered around the origin, and is one-dimensional. It
  # seems the information necessary to separate the labels is preserved. The
  # variability perpendicular to the principal component looks like noise, and
  # would probably just make a classifier overfit.

  # 1.4.3 Loading the Second Set of Labels
  # --------------------------------------

  X, y = syntheticdata.get_synthetic_data_with_labels2()
  y = y.astype(int)[:,0]

  # 1.4.3 Running PCA
  # -----------------

  X = center_data(X)
  pca_eigvec, P = pca(X, 2)

  sp_kw = dict(xticks=[], yticks=[])
  gs_kw = dict(width_ratios=[1, 1, 1/6], height_ratios=[1, 1/6])
  fig_kw = dict(constrained_layout=True, subplot_kw=sp_kw)
  fig, axd = plt.subplot_mosaic([['before', 'after',      'vertical'],
                                 ['.',      'horizontal', '.'       ]],
                                 gridspec_kw=gs_kw, **fig_kw)
  axd['before'].scatter(X[:,0], X[:,1], c=colormap[y], alpha=.7)
  axd['after'].scatter(P[:,0], P[:,1], c=colormap[y], alpha=.7)
  for i, ax in enumerate(['horizontal', 'vertical']):
    for label in np.unique(y):
      axd[ax].eventplot(P[:,i][y==label], color=colormap[label],
                        alpha=.5, orientation=ax)
  axd['before'].legend(title='Before PCA')
  axd['after'].legend(title='After PCA')
  fig.savefig('fig143.pdf')
  fig.suptitle('Visualizing PCA with labeled data (2)')
  fig.show()

  # I've plotted 2 components here. The first component, the one with most
  # variability, is not suitable for separating the two classes, but the second
  # one is (not _perfect_ by any means, but better). Using both could
  # potentially yield even better results, but might overfit if not careful, I
  # think, since the data appears noisy.


  # 1.5 CASE STUDY 1: PCA FOR VISUALIZATION
  # =======================================

  # 1.5.1 Loading the Data
  # ----------------------
  
  X, y = syntheticdata.get_iris_data()

  # 1.5.2 Visualizing the Data by Selecting Features
  # ------------------------------------------------

  fig, axes = plt.subplots(4, 4, **fig_kw)
  features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
  dims = len(features)
  colormap = mpl.cm.get_cmap(lut=dims-1)(range(dims))
  for i in range(dims):
    axes[i][0].set_ylabel(features[i])
    axes[-1][i].set_xlabel(features[i])
    for j in range(dims):
      if i == j:
        for k in range(dims):
          axes[i][j].hist(X[:,j][y==k], color=colormap[k], alpha=.5,
                          histtype='stepfilled', edgecolor='white')
        axes[i][j].set_xlim(np.min(X[:,j]), np.max(X[:,j]))
      else:
        axes[i][j].scatter(X[:,j], X[:,i], c=y, marker='.', alpha=.5)
  fig.savefig('fig152.pdf')
  fig.suptitle('Scatter plot matrix of iris dataset features')
  fig.show()

  # 1.5.3 Visualizing the Data by PCA
  # ---------------------------------

  X = center_data(X)
  pca_eigvec, P = pca(X, 4)

  fig, axes = plt.subplots(4, 1, **fig_kw)
  for i in range(4):
    for j in range(4):
      axes[i].eventplot(P[:,i][y==j], color=colormap[j], alpha=.5)
      axes[i].set(ylabel=f'Comp. {i+1}')
  fig.savefig('fig153.pdf')
  fig.suptitle('Visualizing iris dataset using PCA')
  fig.show()

  # Honestly, unless I've made some mistake, visualizing by selecting features
  # is much more informative for me. The first component contains useful
  # information for sure (the others less so), but it is difficult to
  # interpret. Whereas in the previous plot, I'm able to tell _why_ an iris is
  # classified the way it is.


  # 1.6 CASE STUDY 2: PCA FOR COMPRESSION
  # =====================================
  input("[Press enter to start image (de)compression. This may take a while!]")

  mpl.rcParams['image.cmap'] = 'gray'

  # 1.6.1 Loading the Data
  # ----------------------

  print("Downloading image data ...")
  X, y, h, w = syntheticdata.get_lfw_data()

  # 1.6.2 Inspecting the Data
  # -------------------------

  fig, ax = plt.subplots(**fig_kw)
  ax.imshow(X[0,:].reshape((h, w)))
  fig.savefig('fig162.pdf')
  fig.suptitle('Sample image (uncompressed)')
  fig.show()

  # 1.6.3 Implementing a Compression-Decompression Function
  # -------------------------------------------------------

  # See function definition above.

  # 1.6.4 Compressing and Decompressing the Data
  # --------------------------------------------

  print("Compressing and decompressing image data ...")
  Xhat = encode_decode_pca(X, 200)

  # 1.6.5 Inspecting the Reconstructed Data
  # ---------------------------------------

  fig, axes = plt.subplots(2, 4, **fig_kw)
  for i in range(4):
    axes[0][i].imshow(X[i+10,:].reshape((h, w)))
    axes[1][i].imshow(Xhat[i+10,:].reshape((h, w)))
  axes[0][0].set_ylabel('Uncompressed')
  axes[1][0].set_ylabel('Compressed')
  fig.savefig('fig165.pdf')
  fig.suptitle('Compressed–decompressed images using PCA, 200 dimensions')
  fig.show()

  # 5 samples of uncompressed and compressed images is shown. It works
  # surprisingly well, actually, considering the compressed data is approx.
  # 200/2914 ≈ 6.9% of the original (if you disregard the vectors needed to
  # apply the (de)compression).

  # 1.6.6 Evaluating Different Compressions
  # ---------------------------------------

  Xhat_200 = Xhat
  print("Compressing and decompressing image data (100 dimensions) ...")
  Xhat_100 = encode_decode_pca(X, 100)
  print("Compressing and decompressing image data (500 dimensions) ...")
  Xhat_500 = encode_decode_pca(X, 500)
  print("Compressing and decompressing image data (1000 dimensions) ...")
  Xhat_1000 = encode_decode_pca(X, 1000)

  datasets = [X, Xhat_1000, Xhat_500, Xhat_200, Xhat_100]
  fig, axes = plt.subplots(len(datasets), 9, **fig_kw)
  for i in range(9):
    for j in range(len(datasets)):
      axes[j][i].imshow(datasets[j][i+10,:].reshape((h, w)))
  axes[0][0].set_ylabel('Original')
  axes[1][0].set_ylabel('1000')
  axes[2][0].set_ylabel('500')
  axes[3][0].set_ylabel('200')
  axes[4][0].set_ylabel('100')
  fig.savefig('fig166.pdf')
  fig.suptitle('Compressed–decompressed images using PCA')
  fig.show()

  # I find it difficult to spot any difference between the originals and 1000
  # dimension compression. Even at 500 I have to squint to notice some slight
  # changes in texture. At 200 dimension compression, though, it's really
  # noticeable: sharp lines gets smoothed, textures scrambled, and some details
  # disappears entirely. 100 is worse, but the essence of the original is still
  # there.
