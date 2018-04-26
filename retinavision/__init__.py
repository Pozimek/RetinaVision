#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:20:52 2018

@author: Piotr Ozimek
"""

from .retina import Retina
import cortex
import utils
from os.path import dirname, join

datadir = join(dirname(dirname(__file__)), "data")

del dirname, join
