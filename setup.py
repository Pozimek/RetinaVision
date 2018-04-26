#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:38:55 2018

@author: Piotr Ozimek
"""

import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "retinavision",
    version = "0.8",
    author = "Piotr Ozimek",
    author_email = "piotrozimek9@gmail.com",
    description = ("An artificial software retina for deep learning purposes."),
    url = "",
    packages=['retinavision', 'examples', 'tests'],
    long_description=read('README.md')
)