from usingversion import getattr_with_version

# Generate a version number.
__getattr__ = getattr_with_version("impactchart", __file__, __name__)
