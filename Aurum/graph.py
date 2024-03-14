import pickle
import os
from d3l.input_output.dataloaders import CSVDataLoader
import networkx as nx
from d3l.querying.query_engine import QueryEngine


def average_score(Neighbour_score, threshold=0.5):
    similarities_value = [similarities[1] for name, similarities in Neighbour_score]
    if all(x == 0 for x in similarities_value):
        return [(name, similarities[0]) for name, similarities in Neighbour_score]
    else:
        return [(name, sum(similarities) / len(similarities)) for name, similarities
                in Neighbour_score if sum(similarities) / len(similarities) > threshold]


Data_path = "../Datasets"
Dataloader = CSVDataLoader(
    root_path=Data_path,
    encoding='utf-8'
)
result_path = "../Result"
threshold = 0.5


def build_graph(dataloader: CSVDataLoader, table_dict=None):
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
            graph.add_node(f"{short_name}.{col}", table_name=short_name, column_type=annotation[index],
                           column=table[col])
            if index == subCol_index:
                graph.nodes[f"{short_name}.{col}"]['SubjectColumn'] = True
    return graph


def search_column_neighbours(dataloader, data_path, indexes, table_dict=None, similarity_threshold=0.5):
    graph = build_graph(dataloader, table_dict)
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
    with open("../Result/AurumOnto.pkl", "wb") as save_file:
        pickle.dump(graph, save_file)
    return graph


# G = search_column_neighbours(Dataloader, Data_path, [name_index, value_index], table_dict=subject_Col_dict)

