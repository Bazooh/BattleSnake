import igraph as ig
from UCT.UCT import Tree
import plotly.graph_objects as go
import math
from Constants import *


def create_igraph_tree(tree: Tree, depth=0, max_depth=5, action=None) -> ig.Graph:
    g = ig.Graph()
    g.add_vertex(name=str(tree), value=tree.get_value(), action=action, is_root=depth == 0, winner=tree.winner, nb_visited=tree.nb_visited)
    
    if max_depth is not None and depth >= max_depth - 1:
        return g

    if isinstance(tree, Tree) and tree.children is not None:
        for action, child in enumerate(tree.children):
            if child.winner is not None:
                continue
            child_graph = create_igraph_tree(child, depth + 1, max_depth, action)
            g = g.union(child_graph)
            g.add_edge(str(tree), str(child))

    return g


def show_tree(tree, max_depth=5):
    graph = create_igraph_tree(tree, max_depth=max_depth)
    lay = graph.layout_reingold_tilford(mode="in", root=[graph.vs.find(is_root=True).index])
    position = {k: lay[k] for k in range(graph.vcount())}
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
    
    def get_color(nb_visited, max_nb_visited, winner, value):
        if winner is not None :
            if winner == MAIN_PLAYER :
                return '#00dd00'
            elif winner == OTHER_PLAYER:
                return '#dd0000'
            return '#dddd00'
        
        visited_value = math.log(1 + nb_visited) / math.log(1 + max_nb_visited)
        WHITE_POWER = 0.6
        red = [1, WHITE_POWER*(1 - visited_value), WHITE_POWER*(1 - visited_value)]
        green = [WHITE_POWER*(1 - visited_value), 1, WHITE_POWER*(1 - visited_value)]
        color = '#'
        for i in range(3) :
            color += f"{hex(int(255 * (value*red[i] + (1-value)*green[i])))[2:]}".zfill(2)
        return color
    
    max_nb_visited = max(graph.vs["nb_visited"])

    fig.add_trace(go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers',
        name='bla',
        marker=dict(
            symbol='circle-dot',
            size=18,
            color=[get_color(nb_visited, max_nb_visited, winner, value) for nb_visited, winner, value in zip(graph.vs["nb_visited"], graph.vs["winner"], graph.vs["value"])],
            line=dict(color='rgb(50,50,50)', width=1)
        ),
        text=[f"visited {text} times" for text in graph.vs['nb_visited']],
        hoverinfo='text',
        opacity=0.8
    ))

    def make_annotations(pos, values):
        if len(pos) != len(values):
            raise ValueError('The lists pos and values must have the same length')
        annotations = []
        pos_list = list(pos)
        for k, (x, y) in enumerate(pos_list):
            annotations.append(
                dict(
                    text=f'{values[k]:.2f}',  # Convert the value to a string
                    x=x,
                    y=2*M - y + 0.3,
                    xref='x1',
                    yref='y1',
                    font=dict(color='rgb(0,0,0)', size=10),
                    showarrow=False
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
