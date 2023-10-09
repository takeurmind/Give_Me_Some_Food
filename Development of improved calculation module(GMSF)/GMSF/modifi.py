"""
This contains several function approximation using
taylor series and osculating circle for optimizing
numerical operations.

This optimization is to use min-max scaling 
so most of functions are optimized for interval [-1, 1].

Functions are optimized with use of lower polynomial degree
and some modified coefficient would work well to
deal with errors.

list of functions
0. normalize_radian
1. arccos   2. arcsin   3. arcsinh
4. cos      5. cosh     6. exp
7. exp2     8. expm1    9. sin
10. sinh    11. tan     12. tanh
"""

# -----------------------------------------------

import numpy as np
from . import testdef as test

test.exp = np.exp
test.arccos = np.arccos
test.arcsin = np.arcsin
test.arcsinh = np.arcsinh
test.cos = np.cos
test.cosh = np.cosh
test.exp = np.exp
test.exp2 = np.exp2
test.expm1 = np.expm1
test.sin = np.sin
test.sinh = np.sinh
test.tan = np.tan
test.tanh = np.tanh

# 0. radian normalization for [ninf, inf] to [-pi, pi]

pi_float128 = 3.141592653589793
pi_float128_over_2 = 1.5707963267948966

# normalization for sine, cosine
def normalize_radian(n):
    normalized = n % (2 * pi_float128)
    if normalized > pi_float128:
        normalized -= 2 * pi_float128
    return normalized

# normalization for tangent
def normalize_radian_tan(n):
    normalized = n % (2 * pi_float128_over_2)
    if normalized > pi_float128_over_2:
        normalized -= 2 * pi_float128_over_2
    return normalized

# -----------------------------------------------

# 1. approximation of arccosine function(arccos or acos or cosine inverse)
# Reference : glibc
# Domain of function :  [-1, 1] -> taylor approximation
#                       else -> osculating circle
# note that arccos(x) + arcsin(x)=pi/2, symmetric.

# constants for modified estimation
arccos_b1 = 1
arccos_b3 = 0.16666666666666666666666666666667
arccos_b5 = 0.075
arccos_b7 = 0.04464285714285714285714285714286
arccos_b9 = 0.03038194444444444444444444444444
arccos_b11 = 0.02734375
arccos_b13 = 0.02208533653846153846153846153846
arccos_pi_over_2 =  1.57079632679489661923132169163975

# modified arccos function with degree 13, some modified coefficients

def arccos(n):
    if -0.76 < n < 0.76:
        p = n ** 2
        result = arccos_pi_over_2 - (n * (arccos_b1 + p * (arccos_b3 + p * (arccos_b5 + p * (arccos_b7 + p * (arccos_b9 + p * (arccos_b11 + arccos_b13 * p)))))))
    elif 0.76 <= n <= 1:
        result = arccos_pi_over_2 - 1.61 + np.sqrt(1.64 - (n + 0.28)**2)
    elif -1 <= n <= -0.76:
        result = arccos_pi_over_2 + 1.61 - np.sqrt(1.64 - (n - 0.28)**2)
    else:
        raise ValueError("arccos : Input should be in the range [-1, 1]")
    return result

# -----------------------------------------------

# 2. approximation of arcsin function(arcsin or asin or sin inverse)
# Reference : glibc
# Domain of function :  [-1, 1] -> taylor approximation
#                       else -> osculating circle

# constants for modified estimation
arcsin_b1 = 1
arcsin_b3 = 0.16666666666666666666666666666667
arcsin_b5 = 0.075
arcsin_b7 = 0.04464285714285714285714285714286
arcsin_b9 = 0.03038194444444444444444444444444
arcsin_b11 = 0.02734375
arcsin_b13 = 0.02208533653846153846153846153846

# modified arcsin function with degree 13, some modified coefficients

def arcsin(n):
    if -0.76 <= n < 0.76:
        p = n ** 2
        result = n * (arcsin_b1 +p * (arcsin_b3 + p * (arcsin_b5 + p * (arcsin_b7 + p * (arcsin_b9 + p * (arcsin_b11 + arcsin_b13 * p))))))
    elif 0.76 <= n <= 1:
        result = 1.61 - np.sqrt(1.64 - (n + 0.28)**2)
    elif -1 <= n <= -0.76:
        result = -1.61 + np.sqrt(1.64 - (n - 0.28)**2)
    else:
        raise ValueError("arcsin : Input should be in the range [-1, 1]")
    return result

# -----------------------------------------------

# 3. approximation function y=arcsinh(x) function (arcsinh or asinh)
# Reference : glibc
# Domain of function : [-1, 1] -> taylor approximation
#                       else -> original definition, not optimized

# constants for modified estimation
arcsinh_b1  = 1
arcsinh_b3  = -0.16666666666666666666666666666667
arcsinh_b5  = 0.075
arcsinh_b7  = -0.04553571428571428571428571428571
arcsinh_b9  = 0.03602430555555555555555555555556
arcsinh_b11 = -0.01811079545454545454545454545455

# modified arcsinh function with degree 11, some modified coefficients

def arcsinh(n):
    if -1 <= n <= 1:
        p = n ** 2
        result = n * (arcsinh_b1 + p * (arcsinh_b3 + p * (arcsinh_b5 + p * (arcsinh_b7 + p * (arcsinh_b9 + p * arcsinh_b11)))))
    else:
        result = np.log(n + np.sqrt(n**2 + 1))
    return result

# -----------------------------------------------

# 4. approximation of cosine function(cos)
# Reference : glibc
# Domain of function : [ninf, inf]

# constants for modified estimation
cos_b0 = 1
cos_b2 = -0.5
cos_b4 = 0.04166666666666666666666666666667
cos_b6 = -0.00138888888888888888888888888889
cos_b8 = 2.480158730158730158730158730158715e-5
cos_b10 =-2.755731922398589065255731922398589e-7
cos_b12 = 2.087675698786809897921009032120143e-9

# modified cosine function with degree 12, some modified coefficients

def cos(n):
    n = normalize_radian(n)
    p = n ** 2
    result = 1 + p * (cos_b2 + p * (cos_b4 + p * (cos_b6 + p * (cos_b8 + p * (cos_b10 + p * cos_b12)))))
    return result


# -----------------------------------------------

# 5. approximation of hyperbolic cosine function(cosh)
# Reference : glibc
# Domain of function :  [-1, 1] -> taylor approximation
#                       else -> originally defined
# constants for modified estimation
cosh_b0 = 1
cosh_b2 = 0.5
cosh_b4 = 0.04166666666666666666666666666667
cosh_b6 = 0.00138988888888888888888888888889

# modified hyperbolic cosine function with degree 6, some modified coefficients

def cosh(n):
    if -1 <= n <= 1:
        p = n ** 2
        result = cosh_b0 + p * (cosh_b2 + p * (cosh_b4 + p * cosh_b6))
    else:
        result = test.cosh(n)
    return result

# -----------------------------------------------

# 6. approximation of exp() function y=e^x or y=e**x or y=exp(x)
# Reference : glibc
# Domain of function :  [-1, 1] -> taylor approximation
#                       else -> originally defined

# constants for modified estimation
exp_b0 = 1
exp_b1 = 1
exp_b2 = 0.500000000000000000000000000000 
exp_b3 = 0.168067226890756302521008403361
exp_b4 = 0.045454545454545454545454545455 

# modified exponential function with degree 4, some modified coefficients
def exp(n):
    if -1 <= n <= 1:
        result = exp_b0 + n * (exp_b1 + n * (exp_b2 + n * (exp_b3 + n * exp_b4)))
    else:
        
        result = test.exp(n)
    return result

# -----------------------------------------------


# 7. approximation of  exponential x to the base 2 function(2^x or 2 ** x)
# Domain of function : [-1, 1] -> taylor approximation
#                      else -> orininally defined
# note that e ** (log2(x)) = 2 ** x

# constants for modified estimation
exp2_b0 = 1
exp2_b1 = 1
exp2_b2 = 0.500000000000000000000000000000 
exp2_b3 = 0.168067226890756302521008403361
exp2_b4 = 0.045454545454545454545454545455 
exp2_log2 = 0.693147180559945309417232121458

# modified exponential x to the base 2 function with degree 4, some modified coefficients
def exp2(n):
    # 0.693147180559945309417232121458 = log(2)
    if -1 <= n <= 1:
        n = exp2_log2 * n
        result = exp2_b0 + n * (exp2_b1 + n * (exp2_b2 + n * (exp2_b3 + n * exp2_b4)))
    else:
        result = test.exp2(n)
    return result

# -----------------------------------------------

# 8. approximation of exponential function -1 y=e^x - 1 or y=e**x - 1 or y=exp(x) - 1(expm1)
# Domain of function : [0, 1]

# constants formodified estimation
expm1_b0 = 1
expm1_b1 = 1
expm1_b2 = 0.500000000000000000000000000000 
expm1_b3 = 0.168067226890756302521008403361
expm1_b4 = 0.045454545454545454545454545455 

# modified exponential function - 1 with degree 4, some modified coefficients
def expm1(n):
    if -1 <= n <= 1:
        result = expm1_b0 + n * (expm1_b1 + n * (expm1_b2 + n * (expm1_b3 + n * expm1_b4))) - 1
    else:
        test.expm1(n)
    return result

# -----------------------------------------------

# 9. approximation of sine function(sin)
# Reference : glibc
# Domain of function : [ninf, inf] -> taylor approximation

# constants for modified estimation
sin_b1 = 1
sin_b3 = -0.16666666666666666666666666666667
sin_b5 = 0.00833333333333333333333333333333
sin_b7 =-0.00019841269841269841269841269841
sin_b9 = 2.75573192239858906525573192239859e-6
sin_b11 =-2.50521083854417187750521083854417e-8

# modified sine function with degree 11, some modified coefficients

def sin(n):
    n = normalize_radian(n)
    p = n ** 2
    result = n * (sin_b1 + p * (sin_b3 + p * (sin_b5 + p * (sin_b7 + p * (sin_b9 + p * sin_b11)))))
    return result

# -----------------------------------------------

# 10. approximation of hyperbolic sine function(sinh)
# Reference : glibc
# Domain of function : [-1, 1] -> taylor approximation
#                      else -> originally defined
# constants for modified estimation
sinh_b1 = 1
sinh_b3 = 0.16666666666666666666666666666667
sinh_b5 = 0.00833333333333333333333333333333

# modified hyperbolic sine function with degree 5, some modified coefficients
def sinh(n):
    if -1 <= n <= 1:
        p = n ** 2
        result = n * (sinh_b1 + p * (sinh_b3 + p * sinh_b5))
    else:
        test.sinh(n)
    return result

# -----------------------------------------------

# 11. approximation of exp() function y=tan(x)
# Domain of function : [-0.8, -0.8] -> taylor approximation
#                      else -> originally defined
# constants for modified estimation
tan_b1 = 1
tan_b3 = 0.33333333333333333333333333333333
tan_b5 = 0.10666666666666666666666666666667
tan_b7 = 0.11460317460317460317460317460317

# modified tangent function with degree 7, some modified coefficients
def tan(n):
    if -0.8 <= n <= 0.8:
        n = normalize_radian_tan(n)
        p = n ** 2
        result = n * (tan_b1 + p * (tan_b3 + p * (tan_b5 + p * tan_b7)))
    else:
        result = test.tan(n)
    return result
    

# -----------------------------------------------

# 12. approximation of y=tanh(x)
# Domain of function : [-1, 1] -> taylor approximation
#                      else -> originally defined
# constants formodified estimation
tanh_b1 = 1
tanh_b3 = -0.33333333333333333333333333333333
tanh_b5 = 0.13333333333333333333333333333333
tanh_b7 = -0.03809523809523809523809523809524

# modified hyperbolic tangent function with degree 7, some modified coefficients
def tanh(n):
    if -1 <= n <= 1:
        p = n ** 2
        result = n * (tanh_b1 + p * (tanh_b3 + p * (tanh_b5 + p * tanh_b7)))
    else :
        test.tanh(n)
    return result

# -----------------------------------------------

# vectorize whole functions
arccos = np.vectorize(arccos)
arcsin = np.vectorize(arcsin)
arcsinh = np.vectorize(arcsinh)
cos = np.vectorize(cos)
cosh = np.vectorize(cosh)
exp = np.vectorize(exp)
exp2 = np.vectorize(exp2)
expm1 = np.vectorize(expm1)
normalize_radian = np.vectorize(normalize_radian)
normalize_radian_tan = np.vectorize(normalize_radian_tan)
sin = np.vectorize(sin)
sinh = np.vectorize(sinh)
tan = np.vectorize(tan)
tanh = np.vectorize(tanh)