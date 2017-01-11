import sys

# Python 2 vs 3 compatibility
PY2 = int(sys.version[0]) == 2

if PY2:
    range_ = xrange
    bytes_ = str
    unicode_ = unicode
    string_ = (str, unicode)
    import cPickle as pickle
    from StringIO import StringIO as bytes_io
else:
    range_ = range
    bytes_ = bytes
    unicode_ = str
    string_ = (bytes, str)
    import pickle
    from io import BytesIO as bytes_io

# scikit-learn version compatibility
from sklearn import __version__ as sklearn_version

if '0.15.2' <= sklearn_version <= '0.17.1':
    sklearn_path = 'sklearn_0.15.2_0.17.1'
elif sklearn_version >= '0.18.0':
    sklearn_path = 'sklearn_0.18.0'
