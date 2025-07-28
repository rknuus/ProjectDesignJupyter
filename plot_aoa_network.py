#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to plot the AoA network using draw_aoa_network_float_based().
"""
import sys
import os
import matplotlib
# Use a standard interactive backend
matplotlib.use('MacOSX')  # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date, timedelta
from project_utils.project import Project

project = Project("iDesign Lab Example Project", date(2020, 1, 6))
project.add_task("1", 10)
project.add_task("2", 20, predecessors=[1])
project.add_task("3", 40, predecessors=[1])
project.add_task("4", 30, predecessors=[1])
project.add_task("5", 10, predecessors=[2])
project.add_task("6", 0, predecessors=[3])
project.add_task("7", 10, predecessors=[3])
project.add_task("8", 30, predecessors=[5])
project.add_task("9", 20, predecessors=[6,8])
project.add_task("10", 25, predecessors=[6,8])
project.add_task("11", 10, predecessors=[4,7,9])
project.add_task("12", 10, predecessors=[10])
project.add_task("13", 0, predecessors=[11,12])
project.add_task("14", 10, predecessors=[11,12])
project.add_task("15", 5, predecessors=[11,12])
project.add_task("16", 5, predecessors=[13])
project.add_task("17", 5, predecessors=[14,16])
project.add_task("18", 5, predecessors=[15,17])

print("Project starts on:", project.start_date.strftime("%A, %B %d, %Y"))

# Plot the AoA network (this no longer calls plt.show())
g = project.draw_aoa_network_float_based(title=f"Arrow diagram for {project.name}")

# Get current figure and axis immediately after plotting
fig = plt.gcf()
ax = plt.gca()

# Use the stored node order from the project
node_order = getattr(project, 'aoa_node_order', list(g.nodes()))
node_collection = None
node_positions = {}
node_indices = {}
node_labels = {}  # Store references to node label text artists

# Build correct mapping from nodes to their positions and indices
for idx, node in enumerate(node_order):
    if node in project.aoa_node_artists:
        artist, pos = project.aoa_node_artists[node]
        node_collection = artist  # All nodes share the same PathCollection
        node_positions[node] = pos
        node_indices[node] = idx

# Find and store node label text artists
for text_artist in ax.texts:
    # Node labels have format like "E0\n(5d)" or similar
    if hasattr(text_artist, 'get_text'):
        text_content = text_artist.get_text()
        if text_content.startswith('E') and '\n' in text_content:
            # Extract node number from "E0\n(5d)" format
            try:
                node_num = int(text_content.split('\n')[0][1:])  # Remove 'E' and get number
                if node_num in node_positions:
                    node_labels[node_num] = text_artist
            except (ValueError, IndexError):
                pass

edge_artists = getattr(project, 'aoa_edge_artists', {})
label_artists = getattr(project, 'aoa_label_artists', {})

dragging = {'node': None, 'offset': (0, 0)}

def hit_test(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return None
    
    tol = 0.8  # Tolerance for clicking
    closest_node = None
    closest_distance = float('inf')
    
    # Find the closest node within tolerance
    for node, (x, y) in node_positions.items():
        distance = np.hypot(event.xdata - x, event.ydata - y)
        if distance < tol and distance < closest_distance:
            closest_distance = distance
            closest_node = node
    
    # Debug output to help identify issues
    if closest_node is not None:
        print(f"Hit test: clicked at ({event.xdata:.2f}, {event.ydata:.2f}), found E{closest_node} at distance {closest_distance:.2f}")
    else:
        print(f"Hit test: clicked at ({event.xdata:.2f}, {event.ydata:.2f}), no node found within tolerance {tol}")
        # Show nearby nodes for debugging
        nearby = [(node, np.hypot(event.xdata - x, event.ydata - y)) for node, (x, y) in node_positions.items()]
        nearby.sort(key=lambda x: x[1])
        print(f"  Nearest nodes: {[(f'E{n}', f'{d:.2f}') for n, d in nearby[:3]]}")
    
    return closest_node

def on_press(event):
    node = hit_test(event)
    if node is not None:
        dragging['node'] = node
        x, y = node_positions[node]
        dragging['offset'] = (event.xdata - x, event.ydata - y)
        print(f"Grabbed node E{node} at position ({x:.2f}, {y:.2f})")
        
        # Debug: show all current node positions
        print("Current node positions:")
        for n in sorted(node_positions.keys()):
            px, py = node_positions[n]
            print(f"  E{n}: ({px:.2f}, {py:.2f})")
        print("---")

def on_motion(event):
    node = dragging['node']
    if node is not None and event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        new_x = event.xdata - dragging['offset'][0]
        new_y = event.ydata - dragging['offset'][1]
        node_positions[node] = (new_x, new_y)
        
        # Update PathCollection offsets
        if node in node_indices and node_collection is not None:
            idx = node_indices[node]
            offsets = node_collection.get_offsets()
            offsets[idx] = [new_x, new_y]
            node_collection.set_offsets(offsets)
        
        # Update node label position
        if node in node_labels:
            node_labels[node].set_position((new_x, new_y))
        
        # Update connected edges and labels
        for (u, v), edge_artist in edge_artists.items():
            if u == node or v == node:
                x1, y1 = node_positions[u]
                x2, y2 = node_positions[v]
                # Handle FancyArrowPatch objects
                if hasattr(edge_artist, 'set_positions'):
                    edge_artist.set_positions((x1, y1), (x2, y2))
                elif hasattr(edge_artist, 'set_data'):
                    edge_artist.set_data([x1, x2], [y1, y2])
                else:
                    # For FancyArrowPatch, update the path
                    edge_artist.set_position_a((x1, y1))
                    edge_artist.set_position_b((x2, y2))
        
        for (u, v), label_artist in label_artists.items():
            if u == node or v == node:
                x1, y1 = node_positions[u]
                x2, y2 = node_positions[v]
                label_x = (x1 + x2) / 2
                label_y = (y1 + y2) / 2
                label_artist.set_position((label_x, label_y))
        
        fig.canvas.draw_idle()

def on_release(event):
    node = dragging['node']
    if node is not None:
        emphasize_node(node, False)
        print(f"Released node E{node}")
        dragging['node'] = None

# Connect events
cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

print('Interactive AoA editing enabled. Drag nodes to move them.')

# Show the plot once, after all interactive setup is complete
plt.show()