import math
from math import sqrt
import pandas as pd
from TableMiner.Utils import I_inf
import TableMiner.LearningPhase.SamplingRanking as Ranking
from TableMiner.Utils import bow, keys_with_max_value, dice_coefficient
import TableMiner.SearchOntology as So
import time


class Learning:
    def __init__(self, dataframe: pd.DataFrame, kb='Wikidata'):
        self._dataframe = dataframe
        self._column = None
        self._winning_entities_dict = {}
        self._rowEntities = {}
        self._conceptScores = {}
        self._onto = So.SearchOntology(kb=kb)
        self._winningConcepts = {}

    def get_winning_concepts(self):
        return keys_with_max_value(self._conceptScores)

    def get_column(self):
        return self._column

    def get_Entities(self):
        return list(self._rowEntities.values())

    def get_cell_annotation(self):
        return pd.Series(self._rowEntities.values(), index=self._rowEntities.keys())

    def get_winning_entitiesId(self):
        winning_dict = {}
        for index, entity in self._rowEntities.items():
            winning_dict[entity] = self._winning_entities_dict[entity]['id']
        return winning_dict

    def update_conceptScores(self, concept, column_name, domain):
        concept_new_score = self.conceptScore(concept, column_name, domain)
        # print(concept, " ", self._conceptScores[concept], "new concept score: ", concept_new_score)
        self._conceptScores[concept] = concept_new_score

    def get_concepts(self):
        return list(self._conceptScores.keys())

    def __sampleRank__(self, column_name):
        self._dataframe = Ranking.reorder_dataframe_rows(self._dataframe, column_name)

    def get_dataframe(self):
        return self._dataframe

    def get_column_with_name(self, column_name):
        if column_name not in self._dataframe.columns:
            raise ValueError("column not exists!")
        else:
            self.__sampleRank__(column_name)
            self._column = self._dataframe[column_name]
            return self._dataframe[column_name]

    """
    The following code is for obtaining In-table context, this includes:
        1. Column header 
        2. row context
        3. column context
    """

    def get_column_content(self, current_row_index=None, current_column_name=None):
        """
        find the column content of cell xi,j
        :param current_row_index: xi,j current row index
        :param current_column_name: xi,j current column name
        :return: column_content
        """
        if current_row_index is not None:
            column_content = self._dataframe.loc[
                self._dataframe.index != current_row_index, current_column_name].tolist()
        else:
            column_content = self._dataframe[current_column_name].tolist()
        return " ".join([str(element) for element in column_content])

    def get_row_content(self, current_row_index, current_column_name):
        """
         find the column content of cell xi,j
         :param current_row_index: xi,j current row index
         :param current_column_name:  xi,j current column name
         :return: row_content
         """
        row_content = self._dataframe.loc[current_row_index, self._dataframe.columns != current_column_name].tolist()
        return " ".join([str(element) for element in row_content])

    @staticmethod
    def coverage(entity_text, cell_context):
        """
        Calculate the coverage score between the bag-of-words of an entity and a context.
        :param entity_text: The text representing the entity.
        :param cell_context: The text representing the context.
        :return: The coverage score.
        """
        bow_entity_text = bow(entity_text)
        bow_context_text = bow(cell_context)
        # Calculate the intersection of the two bags-of-words
        intersection = set(bow_entity_text) & set(bow_context_text)

        # Calculate the sum of frequencies in the context for the intersection words
        sum_freq_intersection = sum(bow_context_text[word] for word in intersection)

        # Calculate the coverage
        coverage_score = sum_freq_intersection / sum(bow_context_text.values())
        return coverage_score

    @staticmethod
    def ec(entity, contexts, overlap, context_weights=None):
        """
        Calculate the entity context score for a candidate entity.

        :param overlap: The overlap function to use (either dice or coverage).
        :param entity: The text related to a candidate entity.
        :param contexts: A list of context texts.
        :param context_weights: A dictionary of weights for each context text, if available.
        :return: The entity context score.
        """
        # If no specific weights are provided, assume equal weight for all contexts
        if context_weights is None:
            context_weights = {context: 1 for context in contexts}

        # Initialize the entity context score
        entity_context_score = 0

        # Iterate over each context
        for context in contexts:
            # Calculate the overlap using the provided function
            overlap_score = overlap(entity, context)

            # Retrieve the weight for this context
            weight = context_weights.get(context, 1)  # Default to 1 if not specified

            # Add the weighted overlap to the entity context score
            entity_context_score += overlap_score * weight

        return entity_context_score

    @staticmethod
    def en(entity, cell_text):
        """
        Calculate the en score using the provided bag-of-words sets.

        Args:
        - entity: entity e.
        - cell_text: table cell content T.

        Returns:
        - float: The calculated en score.
        """
        # Calculate the intersection of the two sets
        bowset_e = bow(entity)
        bowset_T = bow(cell_text)
        intersection = set(bowset_e) & set(bowset_T)
        # Calculate the sum of frequencies in the context for the intersection words
        sum_freq_intersection = sum(bowset_e[word] for word in intersection)
        en_score = sqrt(2 * sum_freq_intersection / (sum(bowset_e.values()) + sum(bowset_T.values())))
        return en_score

    @staticmethod
    def calculate_cf(en_score, ec_score, cell_text):
        """
        Calculate the overall confidence score for an entity.

        :param en_score: The entity name score.
        :param ec_score: The entity context score.
        :param cell_text: T_i,j.
        :return: The overall confidence score.
        """
        # Calculate the number of tokens in bow(T_i,j)
        num_tokens = sum(bow(cell_text).values())
        # Calculate the confidence score cf(e_i,j)
        cf_score = en_score + (ec_score / sqrt(num_tokens))
        return cf_score

    def cellWinningEntity(self, cell, index, column: pd.Series):
        winning_concepts_list = list(self.get_winning_concepts())

        winningEntity = None
        column_name = column.name
        entity_score = {}
        if index in self._rowEntities.keys():
            # start_time = time.perf_counter()
            candidate_entities = self._onto.find_candidate_entities(cell)
            entities = []
            for entity in candidate_entities:
                concepts_entity = self.candidateConceptGeneration(entity)
                if not set(winning_concepts_list).isdisjoint(set(concepts_entity)):
                    entities.append(entity)
            # end_time = time.perf_counter()
            # print(f"existing cell {cell} update time {end_time - start_time} second entity {entities}")

        else:
            # start_time = time.perf_counter()
            entities = self._onto.find_candidate_entities(cell)
            # end_time = time.perf_counter()
            # print(f"new cell {cell} update time {end_time - start_time} second entity {entities}")
        for entity in entities:
            # start_time = time.perf_counter()
            triples = self._onto.find_entity_triple_objects(entity)
            # end_time = time.perf_counter()
            # print(f"triples finding {cell}'s entity {entity} update time {end_time - start_time} second triples {triples}")
            if len(triples) > 0:
                entity_score[entity] = {}
                rowContent = self.get_row_content(index, column_name)
                columnContent = self.get_column_content(index, column_name)
                ec = self.ec(entity, [column_name, rowContent, columnContent], self.coverage)
                en = self.en(entity, cell)
                cf = self.calculate_cf(en, ec, cell)
                entity_score[entity]['score'] = cf
                entity_id = self._onto.get_entity_id(entity)
                entity_score[entity]['id'] = entity_id

        if len(entity_score) > 0:
            winningEntity = max(entity_score, key=lambda k: entity_score[k]['score'])
            self._winning_entities_dict[winningEntity] = {'id': entity_score[winningEntity]['id'],
                                                          'score': entity_score[winningEntity]['score'],
                                                          'concept': []}
        if winningEntity is not None:
            self._rowEntities[index] = winningEntity
        return winningEntity

    def candidateConceptGeneration(self, entity):
        # Placeholder: Replace with actual lookup
        concepts_entity = []
        if entity is None:
            #print("Invalid winning entity!")
            return []
        if entity is not None:
            # for time in range(3):
            concepts_entity = self._onto.findConcepts(entity)
            """
            if entity in self._winning_entities_dict.keys():
                entity_ids = self._winning_entities_dict[entity]['id']
            else:
                entity_ids = self._onto.get_entity_id(entity)
            for eid in entity_ids:
                concepts = self._onto.findConcepts(eid)
                for concept in concepts:
                    if concept not in concepts_entity:
                        concepts_entity.append(concept)"""
            # sleep(1)
            if entity in self._winning_entities_dict.keys():
                self._winning_entities_dict[entity]['concept'] = concepts_entity
        return concepts_entity

    def conceptInstanceScore(self, concept):
        score = 0
        for index, winning_entity in self._rowEntities.items():
            # print("index, winning_entity ",index, winning_entity)
            if winning_entity is not None:
                property_eni = self._winning_entities_dict[winning_entity]
                concept_row = property_eni["concept"]
                if concept_row:
                    if concept in concept_row:
                        score += property_eni['score']
        score = score / len(self._rowEntities)
        return score

    def conceptContextScore(self, concept, column_name):
        concept_context = concept
        # uris = So.SearchWikidata.get_concept_uri(concept)
        uris = self._onto.concept_uris(concept)
        column_content = self.get_column_content(current_column_name=column_name)
        if uris:
            concept_context = concept_context + " " + uris[0]
        ec = self.ec(concept_context, [column_name, column_content], dice_coefficient)
        return ec

    @staticmethod
    def domainConceptScore(concept, domain):
        return sqrt(dice_coefficient(concept, domain))

    def conceptScore(self, concept, column_name, domain=None):
        ce_cj = self.conceptInstanceScore(concept)
        cc_cj = self.conceptContextScore(concept, column_name)
        dc_cj = 0 if domain is None else self.domainConceptScore(concept, domain)
        return ce_cj + cc_cj + dc_cj

    def coldStartDisambiguation(self, cell, index):
        concept_pairs = {}
        if isinstance(cell, float):
            if math.isnan(cell):
                return concept_pairs
        winning_entity = self.cellWinningEntity(cell, index, self._column)
        if index in self._rowEntities.keys() and self._conceptScores:
            return concept_pairs
        else:
            # start_time = time.perf_counter()
            concepts_entity = self.candidateConceptGeneration(winning_entity)
            # end_time = time.perf_counter()
            # print(f"existing cell {cell} candidateConceptGeneration time
            # {end_time - start_time} second concepts_entity {concepts_entity}")
            # if len(concepts_entity) == 0:
            # print("The cell and the entity ", cell, index, winning_entity)
            if concepts_entity:
                for cj in concepts_entity:
                    cf_cj = self.conceptScore(cj, self._column.name)
                    concept_pairs[cj] = cf_cj
                # print("winning_entities_dict", self._winning_entities_dict.keys())
            return concept_pairs

    @staticmethod
    def updateCandidateConcepts(current_pairs, concept_pairs):
        # print("previous pair",current_pairs)
        for concept, score in concept_pairs.items():
            current_pairs[concept] = score
        # print("current pair", current_pairs)
        return current_pairs

    def preliminaryColumnClassification(self, column_name):
        column = self.get_column_with_name(column_name)
        # print(column)
        conceptScores = {}
        self._conceptScores = I_inf(column,
                                    conceptScores,
                                    self.coldStartDisambiguation,
                                    self.updateCandidateConcepts)
        self._winningConcepts = keys_with_max_value(self._conceptScores)

    def preliminaryCellDisambiguation(self):
        start_time = time.perf_counter()
        for index, data_item in enumerate(self._column):
            concept_pairs = self.coldStartDisambiguation(data_item, index)
            self._conceptScores = self.updateCandidateConcepts(self._conceptScores, concept_pairs)
        end_time = time.perf_counter()
        print(f"preliminaryCellDisambiguation time: {end_time - start_time} sec \n")
        # print("The column entities ", pd.Series(self._rowEntities))
        return pd.Series(self._rowEntities)
