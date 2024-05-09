# -*- coding: utf-8 -*-
from __future__ import print_function, division

class AbstractTransform(object):
    def __init__(self, params = None):
        pass

    def __call__(self, sample):
        return sample
