import os
import pickle
from collections import defaultdict

import pandas as pd

from SemProp.inputoutput import inputoutput as io
from SemProp.ontomatch import matcher_lib as matcherlib
from SemProp.knowledgerepr import fieldnetwork
from SemProp.modelstore.elasticstore import StoreHandler
from SemProp.ontomatch import glove_api
from SemProp.ontomatch.matcher_lib import MatchingType
from SemProp.ontomatch.ss_api import SSAPI
from SemProp.ontomatch.sem_prop_benchmarking import matchings_basic_pipeline_coh_group_cancellation_and_content


def list_from_dict(combined):
    """
        from local_test in sem_prop_benchmark.py
        :param combined:
        :return:
    """
    l = []
    for k, v in combined.items():
        matchings = v.get_matchings()
        for el in matchings:
            l.append(el)
    return l


class MatchColumn():
    def __init__(self, path_to_serialized_model, onto_name, onto_path, path_to_sem_model):
        self.path_to_serialized_model = path_to_serialized_model
        self.onto_name = onto_name
        self.onto_path = onto_path
        self.path_to_sem_model = path_to_sem_model

    def combineMatch(self, sim_threshold_attr=0.2,
                     sim_threshold_rel=0.2,
                     sem_threshold_attr=0.6,
                     sem_threshold_rel=0.7,
                     coh_group_threshold=0.5,
                     coh_group_size_cutoff=2,
                     sensitivity_cancellation_signal=0.3,
                     summary_threshold = 2,
                     cutting_ratio = 0.8):
        # Deserialize model
        network = fieldnetwork.deserialize_network(self.path_to_serialized_model)
        # Create client
        store_client = StoreHandler()
        # Load glove model
        print("Loading language model...")
        glove_api.load_model(self.path_to_sem_model)
        print("Loading language model...OK")

        # Retrieve indexes
        schema_sim_index = io.deserialize_object(self.path_to_serialized_model + 'schema_sim_index.pkl')
        content_sim_index = io.deserialize_object(self.path_to_serialized_model + 'content_sim_index.pkl')

        # Create ontomatch api
        om = SSAPI(network, store_client, schema_sim_index, content_sim_index)
        # Load parsed ontology
        om.add_krs([(self.onto_name, self.onto_path)], parsed=True)
        all_matchings = defaultdict(list)
        l4_01, l5_01, l42_01, l52_01, l1, l7 = matchings_basic_pipeline_coh_group_cancellation_and_content(
            om,
            om.network,
            om.kr_handlers,
            store_client,
            sim_threshold_attr=sim_threshold_attr,
            sim_threshold_rel=sim_threshold_rel,
            sem_threshold_attr=sem_threshold_attr,
            sem_threshold_rel=sem_threshold_rel,
            coh_group_threshold=coh_group_threshold,
            coh_group_size_cutoff=coh_group_size_cutoff,
            sensitivity_cancellation_signal=sensitivity_cancellation_signal)
        all_matchings[MatchingType.L1_CLASSNAME_ATTRVALUE] = l1
        all_matchings[MatchingType.L4_CLASSNAME_RELATIONNAME_SYN] = l4_01
        all_matchings[MatchingType.L5_CLASSNAME_ATTRNAME_SYN] = l5_01
        all_matchings[MatchingType.L42_CLASSNAME_RELATIONNAME_SEM] = l42_01
        all_matchings[MatchingType.L52_CLASSNAME_ATTRNAME_SEM] = l52_01
        all_matchings[MatchingType.L7_CLASSNAME_ATTRNAME_FUZZY] = l7
        combined_01 = matcherlib.combine_matchings(all_matchings)  # (db_name, source_name) -> stuff
        combined_list = list_from_dict(combined_01)
        combined_sum = matcherlib.summarize_matchings_to_ancestor(om, combined_list,threshold_to_summarize=summary_threshold,
                                                              summary_ratio=cutting_ratio)
        print(len(combined_sum))
        for combine in combined_sum:
            print(combine)
        with open('matchings.pickle', 'wb') as f:
            pickle.dump(combined_sum, f)
        return combined_sum







PATH = os.getcwd()
PATH_DATA = os.path.join(PATH, "test/testmodel/")
print(PATH_DATA)
SEMANTIC_MODEL = "{}{}".format(PATH, "/glove.42B.300d.txt")
ONTOLOGY_NAME = "dbpedia"
PATH_ONTOLOGY = "{}{}".format(PATH, "/aurum/cache_onto/")  # "/aurum/ontomatch/cache_onto/dbpedia.owl"
print(PATH_ONTOLOGY)
print(SEMANTIC_MODEL)
matchCol = MatchColumn(PATH_DATA, ONTOLOGY_NAME, PATH_ONTOLOGY, SEMANTIC_MODEL)
matching_dict = matchCol.combineMatch(sim_threshold_attr=0.2,
                      sim_threshold_rel=0.2,
                      sem_threshold_attr=0.6,
                      sem_threshold_rel=0.6,
                      coh_group_threshold=0.5,
                      coh_group_size_cutoff=2,
                      sensitivity_cancellation_signal=0.3)

with open("Result/dict.pkl", "rb") as f:
    table_dict = pickle.load(f)

def find_subcol_matchings(table, subcol, matching):
    for i in matching:
        db_tuple, onto_tuple = i[0], i[1]
        if table == db_tuple[1] and subcol == db_tuple[2]:
            return onto_tuple

find_match_subcol = []
table_names = os.listdir("Datasets")
for table_name in table_names:
    table = pd.read_csv(f"Datasets/{table_name}")
    headers = table.columns
    if len(table_dict[table_name]) > 0:
        NE_indexes = table_dict[table_name][1]
        subCol_index = max(NE_indexes, key=lambda k: NE_indexes[k])
        subcol_header = headers[subCol_index]
        onto_tuple = find_subcol_matchings(table_name, subcol_header,matching_dict )
        if onto_tuple is not None:
            find_match_subcol.append((table_name, subcol_header,onto_tuple))
print(len(find_match_subcol), find_match_subcol)