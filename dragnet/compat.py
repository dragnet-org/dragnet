import sys

# Python 2 vs 3 compatibility
PY2 = int(sys.version[0]) == 2

if PY2:
    range_ = xrange
    bytes_ = str
    unicode_ = unicode
    string_ = (str, unicode)
    from itertools import izip as zip_
    import cPickle as pickle
    from StringIO import StringIO as bytes_io
else:
    range_ = range
    bytes_ = bytes
    unicode_ = str
    string_ = (bytes, str)
    zip_ = zip
    import pickle
    from io import BytesIO as bytes_io


def str_cast(maybe_bytes, encoding='utf-8'):
    """
    Converts any bytes-like input to a string-like output, with respect to
    python version

    Parameters
    ----------
    maybe_bytes : if this is a bytes-like object, it will be converted to a string
    encoding  : str, default='utf-8'
        encoding to be used when decoding bytes
    """
    if isinstance(maybe_bytes, bytes_):
        return maybe_bytes.decode(encoding)
    else:
        return maybe_bytes


def bytes_cast(maybe_str, encoding='utf-8'):
    """
    Converts any string-like input to a bytes-like output, with respect to
    python version

    Parameters
    ----------
    maybe_str : if this is a string-like object, it will be converted to bytes
    encoding  : str, default='utf-8'
        encoding to be used when encoding string
    """
    if isinstance(maybe_str, unicode_):
        return maybe_str.encode(encoding)
    else:
        return maybe_str


def str_list_cast(list_, **kwargs):
    """
    Converts any bytes-like items in input list to string-like values, with
    respect to python version

    Parameters
    ----------
    list_ : list
        any bytes-like objects contained in the list will be converted to
        strings
    kwargs:
        encoding: str, default: 'utf-8'
            encoding to be used when decoding bytes
    """
    return [str_cast(elem, **kwargs) for elem in list_]


def bytes_list_cast(list_, **kwargs):
    """
    Converts any string-like items in input list to bytes-like values, with
    respect to python version

    Parameters
    ----------
    list_ : list
        any string-like objects contained in the list will be converted to bytes
    kwargs:
        encoding: str, default: 'utf-8'
            encoding to be used when encoding string
    """
    return [bytes_cast(elem, **kwargs) for elem in list_]


def str_dict_cast(dict_, include_keys=True, include_vals=True, **kwargs):
    """
    Converts any bytes-like items in input dict to string-like values, with
    respect to python version

    Parameters
    ----------
    dict_ : dict
        any bytes-like objects contained in the dict will be converted to a
        string
    include_keys : bool, default=True
        if True, cast keys to a string, else ignore
    include_values : bool, default=True
        if True, cast values to a string, else ignore
    kwargs:
        encoding: str, default: 'utf-8'
            encoding to be used when decoding bytes
    """
    new_keys = str_list_cast(dict_.keys(), **kwargs) if include_keys else dict_.keys()
    new_vals = str_list_cast(dict_.values(), **kwargs) if include_vals else dict_.values()
    new_dict = dict(zip_(new_keys, new_vals))
    return new_dict


def bytes_dict_cast(dict_, include_keys=True, include_vals=True, **kwargs):
    """
    Converts any string-like items in input dict to bytes-like values, with
    respect to python version

    Parameters
    ----------
    dict_ : dict
        any string-like objects contained in the dict will be converted to bytes
    include_keys : bool, default=True
        if True, cast keys to bytes, else ignore
    include_values : bool, default=True
        if True, cast values to bytes, else ignore
    kwargs:
        encoding: str, default: 'utf-8'
            encoding to be used when encoding string
    """
    new_keys = bytes_list_cast(dict_.keys(), **kwargs) if include_keys else dict_.keys()
    new_vals = bytes_list_cast(dict_.values(), **kwargs) if include_vals else dict_.values()
    new_dict = dict(zip_(new_keys, new_vals))
    return new_dict


def str_block_cast(block,
                   include_text=True,
                   include_link_tokens=True,
                   include_css=True,
                   include_features=True,
                   **kwargs):
    """
    Converts any bytes-like items in input Block object to string-like values,
    with respect to python version

    Parameters
    ----------
    block : blocks.Block
        any bytes-like objects contained in the block object will be converted
        to a string
    include_text : bool, default=True
        if True, cast text to a string, else ignore
    include_link_tokens : bool, default=True
        if True, cast link_tokens to a string, else ignore
    include_css : bool, default=True
        if True, cast css to a string, else ignore
    include_features : bool, default=True
        if True, cast features to a string, else ignore
    kwargs:
        encoding: str, default: 'utf-8'
            encoding to be used when decoding bytes
    """
    if include_text:
        block.text = str_cast(block.text, **kwargs)
    if include_link_tokens:
        block.link_tokens = str_list_cast(block.link_tokens, **kwargs)
    if include_css:
        block.css = str_dict_cast(block.css, **kwargs)
    if include_features:
        block.features = str_dict_cast(block.features, **kwargs)
    return block


def bytes_block_cast(block,
                     include_text=True,
                     include_link_tokens=True,
                     include_css=True,
                     include_features=True,
                     **kwargs):
    """
    Converts any string-like items in input Block object to bytes-like values,
    with respect to python version

    Parameters
    ----------
    block : blocks.Block
        any string-like objects contained in the block object will be converted
        to bytes
    include_text : bool, default=True
        if True, cast text to bytes, else ignore
    include_link_tokens : bool, default=True
        if True, cast link_tokens to bytes, else ignore
    include_css : bool, default=True
        if True, cast css to bytes, else ignore
    include_features : bool, default=True
        if True, cast features to bytes, else ignore
    kwargs:
        encoding: str, default: 'utf-8'
            encoding to be used when encoding string
    """
    if include_text:
        block.text = bytes_cast(block.text, **kwargs)
    if include_link_tokens:
        block.link_tokens = bytes_list_cast(block.link_tokens, **kwargs)
    if include_css:
        block.css = bytes_dict_cast(block.css, **kwargs)
    if include_features:
        block.features = bytes_dict_cast(block.features, **kwargs)
    return block


def str_block_list_cast(blocks, **kwargs):
    """
    Converts any bytes-like items in input lxml.Blocks to string-like values,
    with respect to python version

    Parameters
    ----------
    blocks : list[lxml.Block]
        any bytes-like objects contained in the block object will be converted
        to a string
    kwargs:
        include_text : bool, default=True
            if True, cast text to a string, else ignore
        include_link_tokens : bool, default=True
            if True, cast link_tokens to a string, else ignore
        include_css : bool, default=True
            if True, cast css to a string, else ignore
        include_features : bool, default=True
            if True, cast features to a string, else ignore
        encoding: str, default: 'utf-8'
            encoding to be used when decoding bytes
    """
    return [str_block_cast(block, **kwargs) for block in blocks]


def bytes_block_list_cast(blocks, **kwargs):
    """
    Converts any string-like items in input lxml.Blocks to bytes-like values,
    with respect to python version

    Parameters
    ----------
    blocks : list[lxml.Block]
        any string-like objects contained in the block object will be converted
        to bytes
    kwargs:
        include_text : bool, default=True
            if True, cast text to bytes, else ignore
        include_link_tokens : bool, default=True
            if True, cast link_tokens to bytes, else ignore
        include_css : bool, default=True
            if True, cast css to bytes, else ignore
        include_features : bool, default=True
            if True, cast features to bytes, else ignore
        encoding: str, default: 'utf-8'
            encoding to be used when decoding bytes
    """
    return [bytes_block_cast(block, **kwargs) for block in blocks]


# scikit-learn version compatibility
from sklearn import __version__ as sklearn_version

if sklearn_version < '0.18.0':
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV

try:
    from sklearn.externals import joblib
except:
    import joblib


# generate model paths
if '0.15.2' <= sklearn_version <= '0.17.1':
    if PY2:
        model_path = 'py2_sklearn_0.15.2_0.17.1'
    else:
        model_path = 'py3_sklearn_0.15.2_0.17.1'
elif sklearn_version >= '0.18.0' and sklearn_version < '1.0.0':
    if PY2:
        model_path = 'py2_sklearn_0.18.0'
    else:
        model_path = 'py3_sklearn_0.18.0'
elif sklearn_version >= '1.0.0':
    if PY2:
        raise Exception('incompatible scikit-learn version: "{}" with Python 2.'.format(sklearn_version))
    else:
        model_path = 'py3_sklearn_1.1.2'
else:
    raise Exception('incompatible scikit-learn version: "{}"'.format(sklearn_version))
