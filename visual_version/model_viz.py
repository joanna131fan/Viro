#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:20:14 2020

@author: joannafan
"""

import time
import math
import numpy as np
import pandas as pd
import pylab as plt
from mesa.visualization.ModularVisualization import ModularServer
from model import InfectionModel, State, MyAgent, number_state, number_infected, number_removed, number_susceptible
import panel as pn
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource, GeoJSONDataSource, ColorBar, HoverTool, Legend, LinearColorMapper, ColorBar
from bokeh.plotting import figure
from bokeh.palettes import brewer
from bokeh.models.glyphs import Line, Glyph
from bokeh.palettes import Category10, Viridis
output_notebook()
import panel as pn
import panel.widgets as pnw
from mesa.batchrunner import BatchRunner
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement, NetworkModule
from mesa.visualization.UserParam import UserSettableParameter


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_color(agent):
        return {
            State.INFECTED: '#FF0000',
            State.SUSCEPTIBLE: '#008000'
        }.get(agent.state, '#808080')

    def edge_color(agent1, agent2):
        if State.REMOVED in (agent1.state, agent2.state):
            return '#000000'
        return '#e8e8e8'

    def edge_width(agent1, agent2):
        if State.REMOVED in (agent1.state, agent2.state):
            return 3
        return 2

    def get_agents(source, target):
        return G.node[source]['agent'][0], G.node[target]['agent'][0]

    portrayal = dict()
    portrayal['nodes'] = [{'size': 6,
                           'color': node_color(agents[0]),
                           'tooltip': "id: {}<br>state: {}".format(agents[0].unique_id, agents[0].state.name),
                           }
                          for (_, agents) in G.nodes.data('agent')]

    portrayal['edges'] = [{'source': source,
                           'target': target,
                           'color': edge_color(*get_agents(source, target)),
                           'width': edge_width(*get_agents(source, target)),
                           }
                          for (source, target) in G.edges]

    return portrayal


network = NetworkModule(network_portrayal, 500, 500, library='d3')
chart = ChartModule([{'Label': 'Infected', 'Color': '#FF0000'},
                     {'Label': 'Susceptible', 'Color': '#008000'},
                     {'Label': 'Removed', 'Color': '#808080'}])


class MyTextElement(TextElement):
    def render(self, model):
        percent = model.removed_percentage()
        cost = model.total_cost()
        percent_text = '&infin;' if percent is math.inf else '{0:.2f}'.format(percent)
        infected_text = str(number_infected(model))

        return "Percent Removed from Population: {}<br>Infected Remaining: {}<br> Estimated Economic Costs: {}".format(percent_text, infected_text, cost)


model_params = {
    'N': UserSettableParameter('slider', 'Number of agents', 100, 100, 2000, 1,
                                       description='Choose how many agents to include in the model'),
    'avg_degree': UserSettableParameter('slider', 'Avg Node Degree', 3, 3, 8, 1,
                                             description='Avg Node Degree'),
    'initial_outbreak': UserSettableParameter('slider', 'Initial Outbreak Size', 10, 1, 2000, 1,
                                                   description='Initial Outbreak Size'),
    'ptrans': UserSettableParameter('slider', 'Probability of Transmission', 0.03, 0.0, 1.0, 0.01,
                                                 description='Probability that susceptible neighbor will be infected'),
    'screening_freq': UserSettableParameter('slider', 'Screening Frequency', 0.4, 0.0, 1.0, 0.1,
                                                   description='Frequency the nodes check whether they are infected by '
                                                               'a virus'),
    'recovery_rate': UserSettableParameter('slider', 'Recovery Rate', 0.98, 0.0, 1.0, 0.01,
                                             description='Probability that the virus will be removed'),
    'screening_cost': UserSettableParameter('slider', 'Screening Cost', 300, 20, 1000, 10, 
                                            description='Cost of screening for COVID-19'),
    'hospitalization_cost': UserSettableParameter('slider', 'Hospitalization Cost', 30000, 1000, 90000, 1000,
                                                  description = 'Cost of Hospitalization in the case of severe or critical conditions')
}

server = ModularServer(InfectionModel, [network, MyTextElement(), chart], 'COVID-19 ABM Economic Evaluation Model', model_params)
server.port = 8521

server.launch()

# def get_column_data(model):
#     """pivot the model dataframe to get states count at each step"""
#     agent_state = model.datacollector.get_agent_vars_dataframe()
#     X = pd.pivot_table(agent_state.reset_index(),index='Step',columns='State',aggfunc=np.size,fill_value=0)    
#     labels = ['Susceptible','Infected','Removed']
#     X.columns = labels[:len(X.columns)]
#     return X

# def plot_states_bokeh(model,title=''):
#     """Plot cases per country"""

#     X = get_column_data(model)
#     X = X.reset_index()
#     source = ColumnDataSource(X)
#     i=0
#     colors = Category10[3]
#     items=[]
#     p = figure(plot_width=600,plot_height=400,tools=[],title=title,x_range=(0,100))        
#     for c in X.columns[1:]:
#         line = Line(x='Step',y=c, line_color=colors[i],line_width=3,line_alpha=.8,name=c)
#         glyph = p.add_glyph(source, line)
#         i+=1
#         items.append((c,[glyph]))

#     p.xaxis.axis_label = 'Step'
#     p.add_layout(Legend(location='center_right',   
#                 items=items))
#     p.background_fill_color = "#e1e1ea"
#     p.background_fill_alpha = 0.5
#     p.legend.label_text_font_size = "10pt"
#     p.title.text_font_size = "15pt"
#     p.toolbar.logo = None
#     p.sizing_mode = 'scale_height'    
#     return p

# def grid_values(model):
#     """Get grid cell states"""

#     agent_counts = np.zeros((model.grid.width, model.grid.height))
#     w=model.grid.width
#     df=pd.DataFrame(agent_counts)
#     for cell in model.grid.coord_iter():
#         agents, x, y = cell
#         c=None
#         for a in agents:
#             c = a.state
#         df.iloc[x,y] = c
#     return df

# def plot_cells_bokeh(model):

#     agent_counts = np.zeros((model.grid.width, model.grid.height))
#     w=model.grid.width
#     df=grid_values(model)
#     df = pd.DataFrame(df.stack(), columns=['value']).reset_index()    
#     columns = ['value']
#     x = [(i, "@%s" %i) for i in columns]    
#     hover = HoverTool(
#         tooltips=x, point_policy='follow_mouse')
#     colors = Category10[3]
#     mapper = LinearColorMapper(palette=colors, low=df.value.min(), high=df.value.max())
#     p = figure(plot_width=500,plot_height=500, tools=[hover], x_range=(-1,w), y_range=(-1,w))
#     p.rect(x="level_0", y="level_1", width=1, height=1,
#        source=df,
#        fill_color={'field':'value', 'transform': mapper},
#        line_color='black')
#     p.background_fill_color = "black"
#     p.grid.grid_line_color = None    
#     p.axis.axis_line_color = None
#     p.toolbar.logo = None
#     return p



# pn.extension()
# plot_pane = pn.pane.Bokeh()
# grid_pane = pn.pane.Bokeh()
# pn.Row(plot_pane,grid_pane,sizing_mode='stretch_width')

# steps=100
# pop=400
# st=time.time()
# model = InfectionModel(pop, 20, 20, ptrans=0.03, death_rate=0.01)
# for i in range(steps):
#     model.step()    
#     p1=plot_states_bokeh(model,title='step=%s' %i)
#     plot_pane.object = p1
#     p2=plot_cells_bokeh(model)
#     grid_pane.object = p2
#     time.sleep(0.2)
# print (time.time()-st)
# agent_state = model.datacollector.get_agent_vars_dataframe()
# print (get_column_data(model))

# def compute_max_infections(model):
#     X=get_column_data(model)
#     try:
#         return X.Infected.max()
#     except:
#         return 0

# compute_max_infections(model)

# fixed_params = {
#     "width": 20,
#     "height": 20,
#     "N": 300,
# }
# #variable_params = {"N": range(20, 200, 10)}
# variable_params = {"ptrans": np.arange(.05, .9, .1)}
# batch_run = BatchRunner(
#     InfectionModel,
#     variable_params,
#     fixed_params,
#     iterations=3,
#     max_steps=100,
#     model_reporters={"Inf": compute_max_infections}
# )

# batch_run.run_all()
# run_data = batch_run.get_model_vars_dataframe()
# #run_data#.head()
# plt.scatter(run_data.ptrans, run_data.Inf)
# plt.show()


