from astropy.io import ascii
cat = ascii.read('Table1.all.txt', format='cds')
cat = ascii.read('Table1.all.txt')
cat = ascii.read('Table1.all.txt', format='cds')
cat = ascii.read('Table_not_header', format='cds', readme='readme_tab')
cat = ascii.read('Table_not_header', format='cds', readme='readme_tab')
cat = ascii.read('Table_not_header', format='cds', readme='readme_tab', fill_values='-')
cat = ascii.read('Table_not_header', format='cds', readme='readme_tab', fill_values=99)
cat = ascii.read('Table_not_header', format='cds', readme='readme_tab')
cat = ascii.read('Table1.all.txt', format='cds')
cat
print cat[0]
print cat[1]
cat = ascii.read('table_with_-', format='cds')
cat = ascii.read('Table1.all.txt', format='cds')
cat = ascii.read('Table1.all.txt', format='cds')
cat
cat = ascii.read('Table1.all.txt', format='cds', fill_values='74658')
cat = ascii.read('Table1.all.txt', format='cds', fill_values=74658)
cat = ascii.read('Table1.all.txt', format='cds', fill_values=7468)
cat = ascii.read('Table1.all.txt', format='cds')
cat
from astropy.table import Column
visual = []
for arow in cat:
    if row['Id'].endswith('*'):
        visual.append(1)
    else: visual.append(0)
visual = []
for arow in cat:
    if arow['Id'].endswith('*'):
        visual.append(1)
    else: visual.append(0)
visual
sum(visual)
Column?
Column(data=visual, name='Visual', dtype='Bool')
vis = Column(data=visual, name='Visual', dtype='Bool')
cat.add_column(vis)
cat
cat.write('cat_lau.csv', format='csv')
import matplotlib.pyplot as plt
plt.plot(cat['C'], cat['n'])
plt.show()
plt.plot(cat['C'], cat['n'], '.')
plt.show()
plt.scatter(cat['C'], cat['n'], c=cat['Visual'])
plt.show()
plt.scatter(cat['C'], cat['n'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.show()
plt.scatter(cat['C'], cat['n'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.scatter(cat['C'], cat['n'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.scatter(cat['H'], cat['n'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.scatter(cat['H'], cat['Y'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.scatter(cat['H'], cat['K'], c=cat['Visual'], cmap='viridis')
plt.scatter(cat['H'], cat['Ks'], c=cat['Visual'], cmap='viridis')
plt.scatter(cat['H'], cat['K'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.scatter(cat['RA'], cat['Dec'], c=cat['Visual'], cmap='viridis')
plt.scatter(cat['RAh'], cat['Decd'], c=cat['Visual'], cmap='viridis')
plt.scatter(cat['RAh'], cat['Decd'], c=cat['Visual'], cmap='viridis')
cat.colnames
plt.scatter(cat['RAh'], cat['DEd'], c=cat['Visual'], cmap='viridis')
cat.colnames
plt.show()
plt.scatter(cat['RAm'], cat['DEm'], c=cat['Visual'], cmap='viridis')
plt.show()
cat_pd = cat.to_pandas()
cat_pd.plot()
plt.show()
cat_pd.plot.kde()
plt.show()
import pandas as pd
pd.scatter_matrix(cat, alpha=0.3, figsize=(10, 10), diagonal='kde')
plt.scatter(cat['H'], cat['Y'], c=cat['Visual'], cmap='viridis')
plt.show()
plt.scatter(cat['Js'], cat['H'], c=cat['Visual'], cmap='viridis')
plt.scatter(cat['J'], cat['H'], c=cat['Visual'], cmap='viridis')
plt.show()
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=2)
cat_pd.sample?
pd.scatter_matrix(cat, alpha=0.3, figsize=(10, 10), diagonal='kde')
pd.scatter_matrix(cat_pd, alpha=0.3, figsize=(10, 10), diagonal='kde')
plt.show()
cat_pd.columns
cat_pd = cat_pd[cat_pd.columns[[False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True,True, True, True, True, True]]]
pd.scatter_matrix(cat_pd, alpha=0.3, figsize=(10, 10), diagonal='kde')
plt.show()
gm
X = cat_pd.as_matrix()
X
clusters = gm.fit(X)
cat_pd = cat.to_pandas()
cat_pd = cat_pd[cat_pd.columns[[False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True,True, True, True, True, False]]]
X = cat_pd.as_matrix()
X
clusters = gm.fit(X)
import numpy as np
np.sum(np.isnan(X))
cat_pd.colnames
cat_pd.columns
%history
cat_pd = cat.to_pandas()
cat_pd = cat_pd[cat_pd.columns[[False, False, False, False, False, False, False, False, False, False, True, True, True, False, False, True, True, True,True, True, True, True, False]]]
cat_pd.columns
X = cat_pd.as_matrix()
clusters = gm.fit(X)
np.sum(np.isnan(X))
X
np.where(X = Nan)
np.where(np.isnan(X))
X[0, 378]
X[378, 0]
X[442, 0]
cat[378]
cat[442]
X2 = np.delete(X, [378, 442], axis=1)
X2[378]
X2 = np.delete(X, [[378, 442]], axis=1)
X2[378]
X2 = np.delete(X, [[378, 442]], axis=0)
X2[378]
X2[442]
clusters = gm.fit(X2)
clusters
gm = GaussianMixture(n_components=2)
clusters = gm.fit(X2)
clusters.n_components
print clusters
clusters.converged_
clusters.covariances_
%history
%history -f 'script.py'
