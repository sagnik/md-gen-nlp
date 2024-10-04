import typing as T
import yaml

from nlp_generalization.yamlenv import env, loader
from nlp_generalization.yamlenv.types import Stream


__version__ = "0.7.1"


def join(_loader, node):
    seq = _loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def substr(_loader, node):
    _data = _loader.construct_sequence(node)
    assert len(_data) == 3
    return _data[0][_data[1] : _data[2]]


def replace(_loader, node):
    _data = _loader.construct_sequence(node)
    assert len(_data) == 3
    return str(_data[0]).replace(str(_data[1]), str(_data[2]))


def load(stream):
    # type: (Stream) -> T.Any
    yaml.add_constructor("!join", join, loader.Loader)
    yaml.add_constructor("!substr", substr, loader.Loader)
    yaml.add_constructor("!replace", replace, loader.Loader)
    data = yaml.load(stream, loader.Loader)
    return env.interpolate(data)


def load_all(stream):
    # type: (Stream) -> T.Iterator[T.Any]
    for data in yaml.load_all(stream, loader.Loader):
        yield env.interpolate(data)
