import pickle
import os
from d3l.input_output.dataloaders import CSVDataLoader
import networkx as nx
from d3l.querying.query_engine import QueryEngine
import plotly.graph_objs as go
import plotly.offline as pyo



def average_score(Neighbour_score, threshold=0.5):
    """
    This function calculates the average syntactic similarity scores of attribute header
    and attribute values.
    if all of attribute header/values similarity scores are 0, we will only consider the other
    score, otherwise we will consider the average syntactic similarity.
    :param Neighbour_score: the score tuple of results
    :param threshold: similarity threshold
    returns: the average syntactic similarity
    """
    similarities_value = [similarities[1] for name, similarities in Neighbour_score]
    if all(x == 0 for x in similarities_value):
        return [(name, similarities[0]) for name, similarities in Neighbour_score]
    else:
        return [(name, sum(similarities) / len(similarities)) for name, similarities
                in Neighbour_score if sum(similarities) / len(similarities) > threshold]


def node_in_graph(dataloader: CSVDataLoader, table_dict=None):
    """
    This function is to load all columns in the datasets as node
    in the graph and return the graph
    each node will store basic information like node name {table_name}.{column_name}, column_type, if it is a subject
    column
    :param dataloader: class that is used to load all data
    :param table_dict: the dictionary that contains the column type dict and the Named-entity columns scores of each
    table calculated by tableMiner+
    """
    graph = nx.Graph()
    tables = table_dict.keys()
    for table_name in tables:
        short_name = table_name[:-4]
        table = dataloader.read_table(table_name=short_name)
        column_t = table.columns
        annotation, NE_scores = table_dict[table_name]
        subCol_index = max(NE_scores, key=lambda k: NE_scores[k])
        # print(annotation, NE_scores)
        for index, col in enumerate(column_t):
            graph.add_node(f"{short_name}.{col}", table_name=short_name, column_type=annotation[index])
            if index == subCol_index:
                graph.nodes[f"{short_name}.{col}"]['SubjectColumn'] = True
    return graph


def buildGraph(dataloader, data_path, indexes, target_path, table_dict=None, similarity_threshold=0.5):
    """
    This function is to build the AURUM graph based on the given indexes.
    :param dataloader: class that is used to load all data
    :param data_path: the data path of all datasets
    :param indexes: the LSH indexes
    :param target_path: where the graph will be saved
    :param table_dict: the dictionary that contains the column type dict and the Named-entity columns scores of each
    table calculated by tableMiner+
    """
    if os.path.exists(os.path.join(target_path, "AurumOnto.pkl")):
        with open(os.path.join(target_path, "AurumOnto.pkl"), "rb") as save_file:
            graph = pickle.load(save_file)
    else:
        graph = node_in_graph(dataloader, table_dict)
        print("start ...")
        T = [i for i in os.listdir(data_path) if i.endswith(".csv")]
        columns = []
        for t in T:
            short_name = t[:-4]
            table = dataloader.read_table(table_name=t[:-4])
            column_t = table.columns
            for col in column_t:
                columns.append((f"{short_name}.{col}", table[col]))
        for col_tuple in columns:
            col_name, column = col_tuple
            qe = QueryEngine(*indexes)
            Neighbours = qe.column_query(column, aggregator=None)
            all_neighbours = average_score(Neighbours, threshold=similarity_threshold)
            isSubCol = graph.nodes[col_name].get('SubjectColumn', False)
            type = "Syntactic_Similar" if isSubCol is False else "PK_FK"
            for neighbour_node, score in all_neighbours:
                if graph.has_edge(col_name, neighbour_node) is False:
                    graph.add_edge(col_name, neighbour_node, weight=score, type=type)
        with open(os.path.join(target_path, "AurumOnto.pkl"), "wb") as save_file:
            pickle.dump(graph, save_file)

    return graph


def draw_interactive_network(G):
    # Get the layout of graph G
    pos = nx.spring_layout(G)

    # Prepare node drawing information
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # 添加节点位置和文本
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

        # Add a text label to each node showing all attributes
        node_info = f'{node}<br>' + '<br>'.join([f'{key}: {value}' for key, value in G.nodes[node].items()])
        node_trace['text'] += (node_info,)

    # Prepare edge drawing information
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=[])

    # Add edge start and end locations
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

        # Add a text label to each edge showing all properties
        edge_info = f'{edge}<br>' + '<br>'.join([f'{key}: {value}' for key, value in G.edges[edge].items()])
        edge_trace['text'] += (edge_info,)

    # Create chart
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Subgraph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # show the chart
    pyo.iplot(fig)
