#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:18:59 2020

@author: joannafan
"""


from mesa import Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import agent
from agent import human
from model import COVID_model
import pandas as pd
import numpy as np
import pylab as plt
from mesa_SIR import SIR
import model_params
import mesa_SIR.calculations_and_plots as c_p
from mesa.visualization.ModularVisualization import ModularServer
import networkx as nx

from mesa.batchrunner import BatchRunner
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement, NetworkVisualization
from mesa.visualization.UserParam import UserSettableParameter

# class NetworkGrid:
#     def __init__(self, G):
#         self.G = G
#         nx.set_node_attributes(self.G, 'agent', None)

#     def place_agent(self, agent, node_id):
#         """ Place a new agent in the space. """

#         self._place_agent(agent, node_id)
#         agent.pos = node_id

#     def get_neighbors(self, node_id, include_center=False):
#         """ Get all adjacent nodes """

#         neighbors = self.G.neighbors(node_id)
#         if include_center:
#             neighbors.append(node_id)

#         return neighbors

#     def move_agent(self, agent, node_id):
#         """ Move an agent from its current node to a new node. """

#         self._remove_agent(agent, agent.pos)
#         self._place_agent(agent, node_id)
#         agent.pos = node_id

#     def _place_agent(self, agent, node_id):
#         """ Place the agent at the correct node. """

#         self.G.node[node_id]['agent'] = agent

#     def _remove_agent(self, agent, node_id):
#         """ Remove an agent at a given point, and update the internal grid. """

#         self.G.node[node_id]['agent'] = None

#     def is_cell_empty(self, node_id):
#         """ Returns a bool of the contents of a cell. """
#         return True if self.G.node[node_id]['agent'] is None else False

#     def get_cell_list_contents(self, cell_list):
#         return list(self.iter_cell_list_contents(cell_list))

#     def iter_cell_list_contents(self, cell_list):
#         return [self.G.node[node_id]['agent'] for node_id in cell_list if not self.is_cell_empty(node_id)]


# class NetworkElement(VisualizationElement):
#     local_includes = ["network_canvas.js",
#                       "sigma.min.js"]

#     def __init__(self, canvas_height=500, canvas_width=500):
#         self.canvas_height = canvas_height
#         self.canvas_width = canvas_width
#         new_element = ("new Network_Module({}, {})".
#                        format(self.canvas_width, self.canvas_height))
#         self.js_code = "elements.push(" + new_element + ");"

#     def render(self, model):
#         return [convert_network_to_dict(model.G)]



def convert_network_to_dict(G):
    d = dict()
    d['nodes'] = [{'id': n_id,
                   'agent_id': getattribute(n['agent'], 'unique_id'),
                   'size': 1 if n['agent'] is None else 3,
                   'color': 'red' if getattribute(n, 'infected', default=0) > 0 else 'green',
                   'label': None if n['agent'] is None else 'Agent:' + str(n['agent'].unique_id) + ' Infected: True'
                   }
                  for n_id, n in G.nodes(data=True)]

    d['edges'] = [{'id': i, 'source': source, "target": target} for i, (source, target, d) in
                  enumerate(G.edges(data=True))]

    return d


def getattribute(n, prop_name, default=None):
    return default if n['agent'] is None else getattr(n['agent'], prop_name)

#FIX GRID
def agent_portrayal(agent):
    if agent is None:
        return
    portrayal = dict()
    portrayal['nodes'] = [{'size': 4,
                           'color': "Blue",
                           'shape': "rect"}
                          for (_, agents) in G.nodes.data('agent')]
    if agent.was_infected == True :
        portrayal['nodes', 'color'] = ["#FF0000", "#FF9999"]
        # portrayal["stroke_color"] = "#00FF00"

    elif agent.alive == True :
        portrayal['nodes', 'color'] = ["#0000FF", "#9999FF"]
        # portrayal["stroke_color"] = "#000000"

    return portrayal

grid = NetworkElement()
chart = ChartModule([{"Label": "infected",
                      "Color": "Blue"}, 
                     {"Label":"dead", 
                      "Color": "Red"},
                     {"Label": "recovered",
                      "Color": "Yellow"}],
                    data_collector_name='datacollector')

params = {
    "ptrans":UserSettableParameter("slider", "Probability of Transmission", 0.03, 0.01, 0.08, 0.005), 
    "reinfection":UserSettableParameter("slider", "Reinfection Rate", 0.01, 0.00, 0.08, 0.005)
}
server = ModularServer(COVID_model,
                        [grid],
                        "COVID-19 Model", 
                        params)
server.port = 8523
server.launch()

