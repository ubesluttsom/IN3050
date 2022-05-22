from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from pca import *

mpl.rcParams['image.cmap'] = 'viridis'
colormap = mpl.cm.get_cmap(lut=3)(range(4))


# 2. QUALITATIVE ASSESSMENT OF K-MEANS CLUSTERING
# ===============================================

# Get data and do PCA

X, y = syntheticdata.get_iris_data()
vec, P = pca(X, 2)

# Plot fancy scatter plot of original data

sp_kw = dict(xticks=[], yticks=[])
gs_kw = dict(width_ratios=[1, 1/6], height_ratios=[1/6, 1])
fig_kw = dict(constrained_layout=True, subplot_kw=sp_kw)
fig, axd = plt.subplot_mosaic([['horizontal', '.'], ['main', 'vertical']],
                               gridspec_kw=gs_kw, **fig_kw)
axd['main'].scatter(P[:,0], P[:,1], c=y, alpha=.7)
axd['main'].set(xlabel='Comp. 1', ylabel=f'Comp. 2')
for i, ax in enumerate(['horizontal', 'vertical']):
  for label in np.unique(y):
    axd[ax].eventplot(P[:,i][y==label], color=colormap[label],
                      alpha=.5, orientation=ax)
fig.suptitle('PCA on iris dataset')
fig.show()

# Calculate k-means and plot predictions

# Q: Why would I use P and not X here? Doesn't X contain more information?
Ks = (2, 3, 4, 5)
yhats = {k: KMeans(k).fit_predict(P) for k in Ks}
fig, axd = plt.subplot_mosaic(np.reshape(Ks, [2, 2]), **fig_kw)
for k in axd:
  axd[k].scatter(P[:,0], P[:,1], c=yhats[k], alpha=.7)
  axd[k].legend(title=f"$k={k}$")
fig.suptitle('K-means clustering on iris dataset')
fig.show()

# The k-means predictions are not _too_ bad. Especially for $k=3$: though the
# boundary between class 1 and 2 (as $y$ labels them) is not correct, it is not
# far off, and—in my opinion—looks smoother than the “correct” $y$ labeling.
# There are some minor, but glaring and annoying, “mistakes” (at least by human
# standards) in the $k=2$ case. For $k={4, 5}$ the clustering seems also
# logical, if there actually was more classes.


# 3. QUANTITATIVE ASSESSMENT OF K-MEANS
# =====================================

# Train a classifier on true y labellings and measure accuracy

y_acc = accuracy_score(y, LogisticRegression().fit(P, y).predict(P))
print(f"True y accuracy: {y_acc}")

# Train classifiers on k-means output: (1) one-hot-encode, (2) fit classifiers,
# and (3) measure accuracy.

yhats = {k: np.eye(max(yhats[k] + 1))[yhats[k]] for k in Ks}
LMs   = {k: LogisticRegression().fit(yhats[k], y) for k in Ks}
acc   = {k: accuracy_score(y, LMs[k].predict(yhats[k])) for k in Ks}
for k in acc:
  print(f"ŷ(k={k}) accuracy: {acc[k]}")

# Plot results

fig, ax = plt.subplots()
ax.step(Ks, [acc[k] for k in Ks], where='mid', c='indigo')
ax.hlines(y_acc, min(Ks), max(Ks), ls='--', color='indigo')
ax.set(xticks=Ks, xlabel='$k$', ylabel='Accuracy')
ax.tick_params(left=False, bottom=False)
ax.grid(True, color='white')
ax.legend(['$\hat y(k)$ accuracy', 'True $y$ accuracy'], loc='lower right')
fig.suptitle('Accuracy of logistic classifiers trained on $k$-means output')
fig.show()
