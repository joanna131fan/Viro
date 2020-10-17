#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:49:07 2020

@author: joannafan
"""


# import time
# import numpy as np
# import pandas as pd
# import pylab as plt
# import enum
# from mesa import Agent, Model
# from mesa.time import RandomActivation
# from mesa.space import MultiGrid
# from mesa.datacollection import DataCollector

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import agent
from agent import human
import pandas as pd
import numpy as np
import pylab as plt
from mesa_SIR import SIR
import model_params
import mesa_SIR.calculations_and_plots as c_p
from mesa.visualization.ModularVisualization import ModularServer
import networkx as nx

class COVID_model(Model):
    """Economic Evaluation Agent-based Model of the SARS-CoV-2 Virus"""
    def __init__(self, ptrans=0.03, reinfection = 0.01):
        super().__init__()
        
        self.ptrans = ptrans
        self.reinfection = reinfection
        self.susceptible = 0
        self.dead = 0
        self.recovered = 0
        self.infected = 0
        interactions = model_params.parameters['interactions']
        self.population = model_params.parameters['population']
        self.SIR_instance = SIR.Infection(self, ptrans = ptrans,
                                          reinfection_rate = reinfection,
                                          I0= model_params.parameters["I0"],
                                          severe = model_params.parameters["severe"],
                                          progression_period = model_params.parameters["progression_period"],
                                          progression_sd = model_params.parameters["progression_sd"],
                                          death_rate = model_params.parameters["death_rate"],
                                          recovery_days = model_params.parameters["recovery_days"],
                                          recovery_sd = model_params.parameters["recovery_sd"])


        self.G = SIR.build_network(interactions, self.population) #FIX: nx.erdos_renyi_graph?
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.dead_agents = []
        self.running = True
    
        for node_id in range(self.population):
            new_agent = agent.human(node_id, self) #what was self.next_id()
            self.grid.place_agent(new_agent, node_id)
            self.schedule.add(new_agent)

        #self.meme = 0
        self.datacollector = DataCollector(model_reporters={"infected": lambda m: c_p.compute(m,'infected'),
                                                            "recovered": lambda m: c_p.compute(m,'recovered'),
                                                            "susceptible": lambda m: c_p.compute(m,"susceptible"),
                                                            "dead": lambda m: c_p.compute(m, "dead"),
                                                            "R0": lambda m: c_p.compute(m, "R0"),
                                                            "severe_cases": lambda m: c_p.compute(m,"severe")})
        self.datacollector.collect(self)
    
    def step(self):
        self.schedule.step()
        
        self.datacollector.collect(self)
        '''
        for a in self.schedule.agents:
            if a.alive == False:
                self.schedule.remove(a)
                self.dead_agents.append(a.unique_id)
        '''

        if self.dead == self.schedule.get_agent_count():
            self.running = False
        else:
            self.running = True

