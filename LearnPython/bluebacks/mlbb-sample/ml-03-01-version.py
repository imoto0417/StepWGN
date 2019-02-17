# -*- coding: utf-8 -*-

try:
    import sklearn
    sklearnExists = True
except ImportError:
    sklearnExists = False

try:
    import numpy
    numpyExists = True
except ImportError:
    numpyExists = False

try:
    import scipy
    scipyExists = True
except ImportError:
    scipyExists = False

try:
    import matplotlib
    matplotlibExists = True
except ImportError:
    matplotlibExists = False

try:
    from PIL import Image
    PILExists = True
except ImportError:
    PILExists = False

try:
    import keras
    kerasExists = True
except ImportError:
    kerasExists = False

try:
    import theano
    theanoExists = True
except ImportError:
    theanoExists = False

if sklearnExists == True:
    print('scikit-learnのバージョンは{0}です'.format(sklearn.__version__))
else:
    print('scikit-learnはインストールされていません')

if numpyExists == True:
    print('numpyのバージョンは{0}です'.format(numpy.__version__))
else:
    print('numpyはインストールされていません')

if scipyExists == True:
    print('scipyのバージョンは{0}です'.format(scipy.__version__))
else:
    print('scypyはインストールされていません')

if matplotlibExists == True:
    print('matplotlibのバージョンは{0}です'.format(matplotlib.__version__))
else:
    print('matplotlibはインストールされていません')

if PILExists == True:
    print('PIL(Pillow)のバージョンは{0}です'.format(Image.PILLOW_VERSION))
else:
    print('PIL(Pillow)はインストールされていません')

if kerasExists == True:
    print('kerasのバージョンは{0}です'.format(keras.__version__))
else:
    print('kerasがインストールされていないか、まだ設定が済んでいません')

if theanoExists == True:
    print('theanoのバージョンは{0}です'.format(theano.__version__))
else:
    print('theanoはインストールされていません')

