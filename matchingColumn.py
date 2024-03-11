import pickle
from collections import defaultdict
from aurum.inputoutput import inputoutput as io
from aurum.ontomatch import matcher_lib as matcherlib
from aurum.knowledgerepr import fieldnetwork
from aurum.modelstore.elasticstore import StoreHandler
from aurum.ontomatch import glove_api
from aurum.ontomatch.matcher_lib import MatchingType
from aurum.ontomatch.ss_api import SSAPI
from aurum.ontomatch.sem_prop_benchmarking import matchings_basic_pipeline_coh_group_cancellation_and_content


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






