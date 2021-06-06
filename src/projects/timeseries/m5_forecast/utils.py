import numpy as np
import plotly.graph_objects as go


def fig_2_add_trace(fig, x_start, x_end, y, mode, name, color, row, col):
    fig.add_trace(
        go.Scatter(
            x = np.arange(x_start, x_end), 
            y = y, 
            showlegend = False, 
            mode = mode, 
            name = name,
            marker = dict(color = color)
        ),
        row = row, 
        col = col
    )