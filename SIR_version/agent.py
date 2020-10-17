#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:04:32 2020

@author: joannafan
"""


from mesa import Agent


#Agent class
class human(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pos = unique_id
        self.infected, self.susceptible, self.severe = self.model.SIR_instance.initial_infection()
        self.was_infected = False
        self.recovered = False
        self.alive = True
        self.day = 0
        self.induced_infections = 0
        self.infected_others = False
        
    def step(self):

        self.model.SIR_instance.interact(self)
        self.day += 1
        