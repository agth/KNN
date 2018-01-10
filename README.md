# KNN
Метод К-ближайших соседей на языке python с использованием библиотеки scikit-learn.

Задача классификации: предполагается, что уже имеется какое-то количество объектов с точной классификацией (т.е. для каждого них точно известно, какому классу он принадлежит). Нужно выработать правило, позволяющее отнести новый объект к одному из возможных классов.

Пример: есть плоскость на которой расположено некоторое количество точек. Каждая точка имеет две координаты и принадлежит одному из двух классов. То есть точки являются объетами имеющими точную классификацию. Задача классификации заключается в том что на основе заданой выборки нужно найти к какому классу принадлежит любая новая точка. Или же раскрасить всю плоскость цветами так чтобы было видно к какому классу будет принадлежать новая точка.

![1](https://user-images.githubusercontent.com/33224690/32176534-3171a94a-bd45-11e7-833b-24a62552b085.png)

В данном случае очевидно, что точка, которую нужно классифицировать лежит ближе к синим точкам. 

Метод K-ближайших соседей (англ.: k-nearest neighbors method, kNN) – один из методов решения задачи классификации. 
Его суть заключается в том, что к точке, которую необходимо классифицировать находятся ближайшие соседи в количестве k, где k - целое число. 
Вводится метрика (расстоние между точками), стандартная это d=sqrt((x-x_i)^2+(y-y_i)^2), где (x,y) координата точки которую нужно классифицировать, (x_i,y_i) - координаты остальных точек. Находится расстояние от классифицируемой точки до каждой и берётся во внимание только k наиболее близко расположенных. 

Пример: K=3
ближайшие 3 точки одна жёлтая 2 синие. 
Далее есть 2 подхода. в python  при вызове функции используются 2 ключевых слова 'uniform' или 'distance'.

Первый подход заключается в том, что вклад в классификацию каждой точки одинаковый. То есть расстояние на котором расположены точки не учитывается. Учитывается только количество точек каждого класса. Или же можно сказать, что используется равномерная весовая функция. **weight='uniform'**. В данном случае 2 синие одна жёлтая, значит точку нужно классифицировать к синему классу. 

Второй подход. Используется весовая функция расстояния. **weight='distance'**. Стандартная весовая функция это 1/расстояние. То есть вклад каждой точки разный и зависит от расстояния. На картинке видно, что жёлтая точка хоть одна, но в отличии от двух синих, находится ближе. Если же отмерить расстояние от классифицируемой точки до каждой из трёх и сравнить 

1/(расстояние до жёлтой)        со значением       1/(расстояние до первой синей)+1/(расстояние до второй синей)

то получится, что первое значение больше второго, и следовательно точку нужно классифицировать к жёлтому классу.
![image](https://user-images.githubusercontent.com/33224690/32180825-3b549b42-bd50-11e7-8b75-5a393335a0b7.png)

Более подробно можно почитать [здесь](http://scikit-learn.org/stable/modules/neighbors.html)


В качестве практической реализации можете ознакомиться с несколькими примерами.

1. [простейший пример обучения кассификатора с использованием библиотеки scikit-learn](https://github.com/cgth/KNN/blob/master/%D0%BF%D1%80%D0%BE%D1%81%D1%82%D0%B5%D0%B9%D1%88%D0%B8%D0%B9%20%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80.py)

Выводит на экран сообщение:

точки
 [[-1 -1]
 [-2 -1]
 [-3 -2]]  принадлежат классу = 1
 
точки
 [[1 1]
 [2 1]
 [3 2]]  принадлежат классу = 2
 
новая точка  [-0.8, -1]  принадлежит классу =  [1]

2. [Допишу](https://github.com/cgth/KNN/blob/master/%D0%BF%D1%80%D0%BE%D1%81%D1%82%D0%B5%D0%B9%D1%88%D0%B8%D0%B9%20%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%20%D1%81%20%D1%80%D0%B8%D1%81%D1%83%D0%BD%D0%BA%D0%BE%D0%BC.py) этот пример так, чтобы на экран выводилось изображение с точками. Точки первого класса обозначены красным, второго - синим.
Новая точка больше по размеру и обозначается цветом в зависимости от того, к какому классу её отнёс классификатор.

![4](https://user-images.githubusercontent.com/33224690/34764212-9c17ee18-f5a2-11e7-9b61-91f797f9fe1a.png)


