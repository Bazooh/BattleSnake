from typing import ValuesView
import igraph as ig
from Snakes.UCT.UCT import Root
import plotly.graph_objects as go
import math
from Constants import *

Pos = tuple[int, int]


def create_igraph_tree(root: Root, depth: int = 0, max_depth: int | None = 5, action: tuple[int, int] | None = None) -> ig.Graph:
    g = ig.Graph()
    g.add_vertex(name=str(root), value=root.winner, action=action, is_root=depth == 0, nb_visited=root.nb_visited)
    
    if max_depth is not None and depth >= max_depth - 1:
        return g

    if root.children is not None:
        for main_action in root.playable_actions[0]:
            for other_action in root.playable_actions[1]:
                if (child := root.children[main_action][other_action]) is None:
                    continue
                
                child_graph = create_igraph_tree(child, depth + 1, max_depth, (main_action, other_action))
                g = g.union(child_graph)
                g.add_edge(str(root), str(child))

    return g


def show_tree(root: Root, max_depth: int | None = 5):
    graph = create_igraph_tree(root, max_depth=max_depth)
    lay = graph.layout_reingold_tilford(mode="in", root=[graph.vs.find(is_root=True).index])
    position: dict[int, Pos] = {k: lay[k] for k in range(graph.vcount())}
    Y = [lay[k][1] for k in range(graph.vcount())]
    M = max(Y)

    Xn = [position[k][0] for k in range(graph.vcount())]
    Yn = [2 * M - position[k][1] for k in range(graph.vcount())]

    Xe = []
    Ye = []
    for edge in graph.es:
        Xe += [position[edge.source][0], position[edge.target][0], None]
        Ye += [2 * M - position[edge.source][1], 2 * M - position[edge.target][1], None]

    labels = graph.vs["name"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=Xe,
        y=Ye,
        mode='lines',
        line=dict(color='rgb(210,210,210)', width=1),
        hoverinfo='none'
    ))
    
    def get_color(nb_visited: int, max_nb_visited: int, value: float | None):
        visited_value = math.log(1 + nb_visited) / math.log(1 + max_nb_visited)
        
        if value is None:
            return '#0000' + hex(int(255 * (1 - visited_value)))[2:].zfill(2)
        
        value = 0.5 * (value + 1)
        
        WHITE_POWER = 0.6
        red = [1, WHITE_POWER*(1 - visited_value), WHITE_POWER*(1 - visited_value)]
        green = [WHITE_POWER*(1 - visited_value), 1, WHITE_POWER*(1 - visited_value)]
        color = '#'
        for i in range(3):
            color += f"{hex(int(255 * (value*red[i] + (1-value)*green[i])))[2:]}".zfill(2)
        return color
    
    max_nb_visited = max(graph.vs["nb_visited"])

    fig.add_trace(go.Scatter(
        x = Xn,
        y = Yn,
        mode = 'markers',
        name = 'bla',
        marker = dict(
            symbol = 'circle-dot',
            size = 18,
            color = [get_color(nb_visited, max_nb_visited, value) for nb_visited, value in zip(graph.vs["nb_visited"], graph.vs["value"])],
            line = dict(color='rgb(50,50,50)', width=1)
        ),
        text = [
            f"main: {LOCAL_ACTIONS_NAMES[actions[0]]}, other: {LOCAL_ACTIONS_NAMES[actions[1]]}" if actions is not None else ""
                for actions in graph.vs['action']
        ],
        hoverinfo = 'text',
        opacity = 0.8
    ))

    def make_annotations(pos: ValuesView[Pos], values: list[float | None]) -> list[dict]:
        if len(pos) != len(values):
            raise ValueError('The lists pos and values must have the same length')
        
        annotations = []
        for k, (x, y) in enumerate(pos):
            annotations.append(
                dict(
                    text = 'No Winner' if values[k] is None else f'{values[k]:.2f}',  # Convert the value to a string
                    x = x,
                    y = 2*M - y + 0.3,
                    xref = 'x1',
                    yref = 'y1',
                    font = dict(color='rgb(0,0,0)', size=10),
                    showarrow = False
                )
            )
        return annotations


    annotations = make_annotations(position.values(), graph.vs["value"])

    fig.update_layout(
        title='UCT Tree',
        annotations=annotations,
        font_size=12,
        showlegend=False,
        xaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        ),
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode='closest',
        plot_bgcolor='rgb(248,248,248)'
    )

    fig.show()
