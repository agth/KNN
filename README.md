# KNN
Метод К-ближайших соседей на языке python с использованием библиотеке scikit-learn.

Задача классификации: предполагается, что уже имеется какое-то количество объектов с точной классификацией (т.е. для каждого них точно известно, какому классу он принадлежит). Нужно выработать правило, позволяющее отнести новый объект к одному из возможных классов.

Пример: есть плоскость на которой расположено некоторое количество точек. Каждая точка имеет координату и принадлежит одному из двух классов. Задача классификации заключается в том что на основе заданой выборки нужно найти к какому классу принадлежит новая точка. 
![1](https://user-images.githubusercontent.com/33224690/32176534-3171a94a-bd45-11e7-833b-24a62552b085.png)

В данном случае очевидно, что точка, которую нужно классифицировать лежит ближе к синим точкам. 

Метод K-ближайших соседей (англ.: k-nearest neighbors method, k-NN) – один из методов решения задачи классификации. 
Его суть заключается в том, что к точке, которую необходимо классифицировать находятся ближайшие соседи в количестве k, где k - целое число. 
Вводится метрика (расстоние медду точками), стандартная это d=sqrt((x-x_i)^2+(y-y_i)^2), где (x,y) координата точки которую нужно классифицировать, (x_i,y_i) - координаты остальных точек. Находится расстояние от классифицируемой точки до каждой и берётся во внимание только k наиболее близко расположенных. 

Пример: K=3

