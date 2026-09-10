from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("amf-convert")
except PackageNotFoundError:
    # package is not installed
    pass
