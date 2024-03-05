import pandas as pd
import TableMiner.LearningPhase.Learning as learn
from TableMiner.SearchOntology import SearchOntology
from TableMiner.Utils import stabilized, def_bow


# from TableMiner.SCDection.TableAnnotation import TableColumnAnnotation as TA


class TableLearning:
    def __init__(self, table: pd.DataFrame, KB="DBPedia"):
        self._table = table
        self._annotation_classes = {}
        # self._NE_Column = TA(table).subcol_Tjs()
        # self._NE_Column = {0: 3.662611942715209, 3: 1.7804230716137477, 4: 1.5887996947126874, 5: 1.4407044164353653} # this is 125
        self._NE_Column = {1: 3.662611942715209, 2: 1.7804230716137477}  # , 3: 1.5887996947126874
        self._domain_representation = {}
        #print(self._NE_Column)
        self.kb = KB
        self._onto = SearchOntology(kb=KB)

    def get_annotation_class(self):
        return self._annotation_classes

    def update_annotation_class(self, column_index, new_learning: learn):
        self._annotation_classes[column_index] = new_learning

    def get_table(self):
        return self._table

    def get_NE_Column(self):
        return self._NE_Column

    def table_learning(self):
        for column_index in self._NE_Column.keys():
            learning = learn.Learning(self._table, kb=self.kb)
            ne_column = self._table.columns[column_index]
            learning.preliminaryColumnClassification(ne_column)
            learning.preliminaryCellDisambiguation()
            self._annotation_classes[column_index] = learning

    def domain_bow(self):
        winning_entities_definitions = set()
        for column_index, learning in self._annotation_classes.items():
            for entity, ids in learning.get_winning_entitiesId().items():
                for entity_id in ids:
                    definition = self._onto.defnition_sentences(entity_id)
                    if definition is not None:
                        winning_entities_definitions.add(definition)
        bow_domain = def_bow(list(winning_entities_definitions))
        return bow_domain


def updatePhase(currentLearnings: TableLearning):
    table = currentLearnings.get_table()
    print("Starting update")
    previousLearnings = None
    i = 0
    while table_stablized(currentLearnings, previousLearnings) is False:
        previousLearnings = currentLearnings
        bow_domain = currentLearnings.domain_bow()
        for column_index in currentLearnings.get_annotation_class().keys():
            learning = currentLearnings.get_annotation_class()[column_index]
            concepts = learning.get_winning_concepts()
            print(f" iteration {i} column_index {column_index} concepts {concepts}")
            for concept in concepts:
                learning.update_conceptScores(concept, table.columns[column_index], bow_domain)
            learning.preliminaryCellDisambiguation()
            currentLearnings.update_annotation_class(column_index, learning)
        i += 1


def table_stablized(currentLearnings, previousLearnings=None):
    if previousLearnings is None:
        return False
    else:
        stablizedTrigger = True
        for column_index in currentLearnings.get_NE_Column().keys():
            currentLearning_index = currentLearnings.get_annotation_class()[column_index]
            previousLearning_index = previousLearnings.get_annotation_class()[column_index]
            winning_entities = currentLearning_index.get_Entities()
            previous_entities = previousLearning_index.get_Entities()
            concepts = currentLearning_index.get_winning_concepts()
            previous_concepts = previousLearning_index.get_winning_concepts()
            if stabilized(winning_entities, previous_entities) is True and stabilized(concepts, previous_concepts) is True:
                stablizedTrigger = True
            else:
                stablizedTrigger = False
                print("Unstabilized! Re-running ...")
        return stablizedTrigger
