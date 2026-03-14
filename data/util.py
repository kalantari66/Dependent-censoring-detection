import pickle

class _PickleDotDict(dict):
    """Compatibility class for pickles created from ucimlrepo.dotdict."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


class _UCIPickleUnpickler(pickle.Unpickler):
    """Unpickler that resolves ucimlrepo.dotdict without ucimlrepo installed."""

    def find_class(self, module, name):
        if module == "ucimlrepo.dotdict" and name == "dotdict":
            return _PickleDotDict
        return super().find_class(module, name)


def load_pickle_compat(path: str):
    with open(path, "rb") as f:
        return _UCIPickleUnpickler(f).load()

