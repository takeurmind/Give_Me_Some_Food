import numpy as np
import os
from . import testdef as test


def __init__(self):
    print(os.environ.get('ENVIRONMENT'))
    


def on():
    # from . import testdef as test
    from . import modifi as md   
    # test
    np.exp = md.exp
    np.arccos = md.arccos
    np.arcsin = md.arcsin
    np.arcsinh = md.arcsinh
    np.cos = md.cos
    np.cosh = md.cosh
    np.exp = md.exp
    np.exp2 = md.exp2
    np.expm1 = md.expm1
    np.sin = md.sin
    np.sinh = md.sinh
    np.tan = md.tan
    np.tanh = md.tanh
    
def off():
    # from . import testdef as test
    np.exp = test.exp
    np.arccos = test.arccos
    np.arcsin = test.arcsin
    np.arcsinh = test.arcsinh
    np.cos = test.cos
    np.cosh = test.cosh
    np.exp = test.exp
    np.exp2 = test.exp2
    np.expm1 = test.expm1
    np.sin = test.sin
    np.sinh = test.sinh
    np.tan = test.tan
    np.tanh = test.tanh