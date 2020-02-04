# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:37:34 2019

@author: Hp
"""

import time

class ResumableTimer():

    def __init__(self):
        self.time = time
        self.elapsed_time = 0
        
    def start(self):
        self.elapsed_time = 0
        self.start_time = time.time()

    def pause(self):
        self.pause_time = time.time()
        self.elapsed_time += self.pause_time-self.start_time

    def resume(self):
        self.start_time = time.time()

    def get_actual_time (self):
        return self.elapsed_time

