"""
================================
K-Nearest Neighbors Classification
Медот K-ближайших соседей
================================

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15 #K-целое значение, указанное пользователем. 

# import some data to play with
# импортировать некоторые данные, чтобы играть с
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
# Мы принимаем только первые две функции. Мы могли бы избежать этого уродливого
# нарезка с использованием двухмерного набора данных
X = iris.data[:, :2]
y = iris.target

h = 0.02  # step size in the mesh # размер шага в сетке

# Create color maps
# Создание цветовых карт
cmap_light = ListedColormap(['#bafffc', '#ffdede','#abffbd']) #цвета площадок
cmap_bold = ListedColormap(['Blue', 'Red', 'Lime']) #цвета точке

#weights_ ='uniform'




def narisovat(weights_):
    #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    # we create an instance of Neighbours Classifier and fit the data.
    # мы создаем экземпляр классификатора соседей и помещаем данные.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights = weights_)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # График границы решения. Для этого мы назначим каждому цвет пункт 
    # в сетки [x_min, x_max]х[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])    
    # Put the result into a color plot
    # Поместите результат в цветовую схему
    Z = Z.reshape(xx.shape)
    plt.figure() #рисует 'uniform' . без этой строки рисует только 'distance'
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)#отрисовка цветов фона плоскости

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='black', s=20)#отрисовка точек на плоскости
        #edgecolor='b' цвет ободков вокруг точек.
    plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights_ ))#отрисовка заголовка
    plt.show() #показать на экране рисунки


narisovat('uniform')
narisovat('distance')