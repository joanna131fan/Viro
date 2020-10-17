#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:51:49 2020

@author: joannafan
"""

# FIX SIR GRAPH SHAPE, ADD RECOVERED/SUSCEPTIBLE
# ADD COST
import time
import math
import numpy as np
import pandas as pd
import pylab as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from enum import Enum
import networkx as nx


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    SEVERE = 2
    REMOVED = 3
    
def number_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])

def sum_state(model, state, amount):
    return sum([amount for a in model.grid.get_all_cell_contents() if a.state is state])

def number_infected(model):
    return number_state(model, State.INFECTED)


def number_susceptible(model):
    return number_state(model, State.SUSCEPTIBLE)

def number_severe(model):
    return number_state(model, State.SEVERE)

def number_removed(model):
    return number_state(model, State.REMOVED)


class InfectionModel(Model):
    """Over the last decade, an intensive effort has been undertaken to develop global surveillance networks that combat pandemics of emergent infectious diseases. To better understand the spread of a specific virus, computational models have been created. In contemporary mathematical epidemiology, agent-based modeling (ABM) represents the state-of-the-art for simulating complex epidemic systems. Taking into account transportation infrastructure of the simulated area, the mobility of the population, and evolutionary aspects of the disease, ABMs are by nature a “brute force” method that details a bottom-up stochastic approach to approximating the dynamics of real-world cases, i.e. individual-level behaviors define system-level results. To mirror the 2020 coronavirus outbreak, a non-linear network structure similar to a SIR mathematical model is applied. Then, a step further is taken to connect the outbreak with its economic implications in a medical cost-effectiveness analysis. In this study, the agent-based model is utilized to model the epidemiological and economic implications of viral pandemics, specifically the SARS-cov-2 virus.

""" #ABSTRACT

    def __init__(self, N=100, avg_degree = 3, initial_outbreak = 1, ptrans=0.03, severe_rate=0.3,
                 death_rate=0.01, screening_freq = 0.4, recovery_rate = 0.98, recovery_days=21,
                 recovery_sd=7, screening_cost = 300, hospitalization_cost = 30000):

        self.num_agents = N
        prob = avg_degree / N
        self.G = nx.erdos_renyi_graph(n=self.num_agents, p=prob)
        self.grid = NetworkGrid(self.G)
        
        self.schedule = RandomActivation(self)
        self.initial_outbreak = initial_outbreak
        self.ptrans = ptrans
        
        self.recovery_days = recovery_days
        self.recovery_sd = recovery_sd
        self.recovery_rate = recovery_rate
        self.death_rate = death_rate
        self.screening_freq = screening_freq
        self.screening_cost = screening_cost
        self.hospitalization_cost = hospitalization_cost
        self.cost = 0

        self.datacollector = DataCollector({"Infected": number_infected,
                                            "Susceptible": number_susceptible,
                                            "Removed": number_removed})
        # Create agents
        for i, node in enumerate(self.G.nodes()):
            a = MyAgent(i, self, State.SUSCEPTIBLE, ptrans, severe_rate, screening_freq, recovery_rate, screening_cost, hospitalization_cost)
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)
            
        # Initial infection 
        infected_nodes = self.random.sample(self.G.nodes(), self.initial_outbreak)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.INFECTED
        
        self.running = True
        self.datacollector.collect(self)
    
    def removed_percentage(self):
        try:
            return number_state(self, State.REMOVED) / self.num_agents
        
        except ZeroDivisionError:
            return math.inf
        
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        
    # Costs
    def total_cost(self): #check and start back up here 9/20
        screening = sum_state(self, State.INFECTED, self.screening_freq*self.screening_cost)
        hospitalization = sum_state(self, State.SEVERE, self.hospitalization_cost)
        self.cost += screening + hospitalization
        return (self.cost)

    def run_model(self,n=100):
        for i in range(n):
            self.step()
        
class MyAgent(Agent):
    """ An agent in an epidemic model."""
    def __init__(self, unique_id, model, initial_state, ptrans, severe_rate, screening_freq, recovery_rate, screening_cost, hospitalization_cost):
        super().__init__(unique_id, model)      
        self.state = initial_state
        self.ptrans = ptrans
        self.screening_freq = screening_freq
        self.recovery_rate = recovery_rate
        self.severe_rate = severe_rate
        self.screening_cost = screening_cost
        self.hospitalization_cost = hospitalization_cost
        
        self.cost = 0


    def contact(self):
        """Find close contacts and infect"""

        cellmates = self.model.grid.get_cell_list_contents([self.pos])       
        if len(cellmates) > 1:
            for other in cellmates:
                if self.random.random() > self.ptrans:
                    continue
                other.state = State.INFECTED
    
    def try_remove_infection(self): #TODO: Distinguish cost between susceptible and severe
        # Try to remove
        if self.random.random() < self.recovery_rate:
            # Success
            self.state = State.SUSCEPTIBLE
        else:
            # Failed
            self.state = State.REMOVED
            self.model.schedule.remove(self)

    def make_severe(self):
        # From Infected to Severe
        if self.state == State.INFECTED:
            if self.random.random() < self.severe_rate:
                self.state = State.SEVERE
                self.cost += self.hospitalization_cost

    def try_check_situation(self):
        if self.random.random() < self.screening_freq:
            self.cost = self.screening_cost
            # Checking...
            if self.state is State.INFECTED:
                self.make_severe()
                self.try_remove_infection()
        # else: #assume that if not screened, infected people continued to progress to advanced stages of the disease.
        #     if self.state is State.INFECTED:
        #         self.make_severe()

    def step(self):
        if self.state is State.INFECTED:
            self.contact()
        self.try_check_situation()


