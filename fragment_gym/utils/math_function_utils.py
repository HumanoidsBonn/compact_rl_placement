#!/usr/bin/python3

import math

class MathFunction():
    def __init__(self):
        pass

    def positive_tanh(self, x, x_factor=1.0, y_factor=1.0):
        y = y_factor*(1-math.tanh(x_factor*x))
        return y
    
    def negative_tanh(self, x, x_factor=1.0, y_factor=1.0):
        y = (y_factor*(1-math.tanh(x_factor*x)))-y_factor
        return y
    
    def positive_quotient_x(self, x, x_factor=1.0, y_factor=1.0):
        y = y_factor*(0.1/((x_factor*x)+0.1))
        return y
    
    def positive_linear(self, x, x_factor=1.0, y_factor=1.0):
        y = y_factor*((-x_factor*x)+x_factor)
        return y