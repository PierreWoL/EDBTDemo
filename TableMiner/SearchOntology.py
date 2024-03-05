import requests
from TableMiner.Utils import nltk_tokenize,tokenize_with_number
from SPARQLWrapper import SPARQLWrapper, JSON


class SearchOntology:
    def __init__(self, kb="Wikidata"):
        self._kb = kb
        self._candidates = []
        if self._kb == "Wikidata":
            self._ontology = SearchWikidata
        else:
            self._ontology = SearchDBPedia

    def get_candidates(self):
        return self._candidates

    def get_entity_id(self, entity_name):
        entities = [i["label"] for i in self._candidates]
        if entity_name not in entities:
            # print(entity_name,entities, "Entity not found")
            return []
        else:
            if self._kb != "Wikidata":
                entity_ids = [i["uri"] for i in self._candidates if i["label"] == entity_name]
            else:
                entity_ids = [i["id"] for i in self._candidates if i["label"] == entity_name]
            return entity_ids

    def find_candidate_entities(self, cell_content):
        """
        Filters candidate entities based on overlap with cell content.

        Args:
        - cell_content (str): The text content of the cell.
        - candidate_entities (list of str): A list of candidate entity names.

        Returns:
        - list of str: A filtered list of candidate entity names that overlap with cell content.
        """
        # filtered_candidates = []
        # Convert cell content and candidate names to lower case for case-insensitive matching

        lower_content = cell_content.lower()
        cell_content_token = tokenize_with_number(lower_content).split(" ")# nltk_tokenize(lower_content)
        # print(cell_content, cell_content_token)
        entities = self._ontology.search(cell_content) #cell_content
        for candidate in entities:
            entity = candidate['label']
            candidate_token = nltk_tokenize(entity.lower())
            # Check if there's an overlap between the cell content and candidate name
            if set(candidate_token).intersection(cell_content_token):
                # filtered_candidates.append(candidate)
                self._candidates.append(candidate)
        entities = [i["label"] for i in self._candidates]
        return list(set(entities))

    def find_entity_triple_objects(self, entity_name):

        entity_ids = self.get_entity_id(entity_name)

        candidate_triples = []
        for entity_id in entity_ids:
            triples = self._ontology.retrieve_entity_triples(entity_id)
            if triples is not None:
                for triple in triples:
                    candidate_triples.append(triple["value"])
            """else:
                print("\n", entity_name, entity_ids)
                print(entity_id)"""
        return " ".join(candidate_triples)

    def findConcepts(self, cell_content):
        entity_ids = self.get_entity_id(cell_content)
        concepts_all = []
        for entity_id in entity_ids:
            concepts = self._ontology.retrieve_concepts(entity_id)
            if concepts:
                for concept in concepts:
                    if concept not in concepts_all:
                        concepts_all.append(concept)
        return concepts_all

    def concept_uris(self, cell_content):
        return self._ontology.get_concept_uri(cell_content)

    def defnition_sentences(self, cell_uri):
        return self._ontology.get_definitional_sentence(cell_uri)


class SearchWikidata:
    @staticmethod
    def search(cell_content, limit=5):
        """
        Search for candidate entities in Wikidata based on the cell text.

        Args:
        - cell_text (str): The text content of the table cell to search for.

        Returns:
        - list: A list of candidate entities with their Wikidata IDs and labels.
        """

        # URL for the Wikidata SPARQL endpoint
        SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

        # SPARQL query to search for entities with a matching label
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search "{cell_content}";
                            mwapi:language "en".
            ?item wikibase:apiOutputItem mwapi:item.
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT {limit}
        """

        headers = {
            "User-Agent": "Wikidata Search Python script",
            "Accept": "application/sparql-results+json"
        }

        try:
            # Perform the HTTP request to the SPARQL endpoint
            response = requests.get(SPARQL_ENDPOINT, params={'query': query}, headers=headers, timeout=2)  # , timeout=2
            response.raise_for_status()  # will raise an exception for HTTP error codes

            # Parse the response to JSON
            data = response.json()

            # Extract the candidate entities
            candidates = [{
                'id': binding['item']['value'].split('/')[-1],  # Extract the QID
                'label': binding['itemLabel']['value']
            } for binding in data['results']['bindings']]

            return candidates

        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            return None
        except Exception as err:
            print(f"An error occurred: {err}")
            return None

    @staticmethod
    def retrieve_entity_triples(entity_id):
        sparql_query = f"""
        SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
          BIND(wd:{entity_id} AS ?entity)
          ?entity ?p ?statement .
          ?statement ?ps ?value .
          ?property wikibase:claim ?p.
          ?property wikibase:statementProperty ?ps.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        url = "https://query.wikidata.org/sparql"
        response = requests.get(url, params={'query': sparql_query, 'format': 'json'})  # , timeout=3
        if response.status_code == 200:
            results = response.json()["results"]["bindings"]
            triples = []
            for result in results:
                triples.append({
                    "property": result["propertyLabel"]["value"],
                    "value": result.get("valueLabel", {}).get("value", result["value"]["value"])
                })
                """
                triples.append(
                  f'{result["propertyLabel"]["value"]} {result.get("valueLabel", {}).get("value", result["value"]["value"])}'
                )"""
            return triples
        else:
            return None

    @staticmethod
    def retrieve_concepts(entity_id):
        # wd:%s wdt:P31/wdt:P279* ?concept .
        # wd:%s wdt:P31/wdt:P279?/wdt:P279? ?concept .
        sparql_query = """
        SELECT ?concept ?conceptLabel WHERE {
          wd:%s wdt:P31/wdt:P279? ?concept .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """ % entity_id
        url = "https://query.wikidata.org/sparql"
        response = requests.get(url, params={'query': sparql_query, 'format': 'json'})  # , timeout=3

        if response.status_code == 200:
            results = response.json()["results"]["bindings"]
            concepts = {result['conceptLabel']['value'] for result in results}
            return concepts
        else:
            return []

    @staticmethod
    def get_concept_uri(concept_label):
        """
        Get the URI of a concept in Wikidata by its label using P279 (subclass of).

        Args:
        - concept_label (str): The label of the concept to search for.

        Returns:
        - list: A list of URIs for the concept found in Wikidata.
        """
        # Endpoint URL for the Wikidata SPARQL service
        SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

        # SPARQL query to find concepts with the specified label
        sparql_query = f"""
        SELECT ?concept WHERE {{
          ?concept wdt:P279* wd:{concept_label}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        headers = {
            "User-Agent": "Wikidata SPARQL Python script",
            "Accept": "application/sparql-results+json"
        }

        try:
            # Perform the HTTP request to the SPARQL endpoint
            response = requests.get(SPARQL_ENDPOINT, params={'query': sparql_query}, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP error codes

            # Parse the response to JSON
            data = response.json()

            # Extract the URIs for the concept
            uris = [binding['concept']['value'] for binding in data['results']['bindings']]

            return uris
        except:
            return []

        # except requests.exceptions.HTTPError as err:
        #    print(f"HTTP error occurred: {err}")
        #    return None
        # except Exception as err:
        #    print(f"An error occurred: {err}")
        #    return None

    @staticmethod
    def get_definitional_sentence(wikidata_id):
        # Define the SPARQL query
        query = """
        SELECT ?entityDescription WHERE {
            wd:""" + wikidata_id + """ schema:description ?entityDescription.
            FILTER(LANG(?entityDescription) = "en")
        }
        """

        url = 'https://query.wikidata.org/sparql'
        headers = {
            "User-Agent": "Wikidata SPARQL Python script",
            "Accept": "application/sparql-results+json"
        }

        try:
            response = requests.get(url, headers=headers, params={'query': query, 'format': 'json'})
            response.raise_for_status()  # This will raise an exception for HTTP errors
            data = response.json()

            results = data.get('results', {}).get('bindings', [])
            if results:
                description = results[0]['entityDescription']['value']
                return description
            else:
                return "No description found."
        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"

    # Example usage:


# wikidata_id = 'Q183259'  # Wikidata ID for Earth
# print(SearchWikidata.get_definitional_sentence(wikidata_id))

class SearchDBPedia:
    @staticmethod
    def search(cell_content, limit=3):
        """
        Search for entities in DBpedia related to the given text.

        :param cell_content: The text to search for.
        :param limit: Maximum number of results to return.
        :return: A list of matching DBpedia resources.
        """
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        cell_content_escaped = cell_content.replace("'", r" ")
        # print(cell_content, cell_content_escaped)
        query = """
        SELECT DISTINCT ?resource ?label WHERE {
          ?resource rdfs:label ?label.
          ?label bif:contains "'%s'".
          FILTER (lang(?label) = 'en')
        } LIMIT %d
        """ % (
            cell_content_escaped , limit)  # Simple escaping, more sophisticated escaping may be needed
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(1)
            results = sparql.query().convert()

            entities = []
            for result in results["results"]["bindings"]:
                entities.append({
                    "uri": result["resource"]["value"],
                    "label": result["label"]["value"]
                })

            return entities
        except:
            return []

    @staticmethod
    def retrieve_entity_triples(entity_uri, limit=3):
        """
        Retrieve the triples for a given entity URI from DBpedia.

        :param entity_uri: The URI of the entity in DBpedia.
        :return: A list of triples (subject, predicate, object) associated with the entity.
        """

        def simplify_uri(uri):
            """
            Simplifies a URI to its last component which is usually the most meaningful part.

            :param uri: The full URI as a string.
            :return: The simplified version of the URI.
            """
            # Split the URI by '#' and '/' and return the last part
            return uri.split('#')[-1].split('/')[-1]

        def format_triple(predicate, obj):
            """
            Formats the predicate and object of a triple to a more readable form.

            :param predicate: The predicate URI of the triple.
            :param obj: The object URI or literal of the triple.
            :return: A tuple of simplified predicate and object.
            """
            simplified_predicate = simplify_uri(predicate)
            simplified_object = simplify_uri(obj) if obj.startswith('http') else obj
            return simplified_predicate, simplified_object

        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = """
        SELECT ?predicate ?object WHERE {
          <%s> ?predicate ?object.
        } LIMIT %d
        """ % (entity_uri, limit)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(2)
            results = sparql.query().convert()

            triples = []

            for result in results["results"]["bindings"]:
                property_per, value_per = format_triple(result["predicate"]["value"],
                                                        result["object"]["value"])
                triples.append({
                    "property": property_per,
                    "value": value_per
                })
            return triples
        except:
            return []

    @staticmethod
    def retrieve_concepts(uri, limit=3):
        """
        Retrieve concepts associated with a given DBpedia entity URI.

        :param uri: The URI of the DBpedia entity.
        :param limit: Maximum number of concepts to return.
        :return: A list of concepts associated with the entity.
        """
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = """
        SELECT ?type ?broader WHERE {
          { <%s> rdf:type ?type }
          UNION
          { <%s> skos:broader ?broader }
        } LIMIT %d
        """ % (uri, uri, limit)

        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(1)
            results = sparql.query().convert()

            concepts = []
            for result in results["results"]["bindings"]:
                if 'type' in result:
                    type_uri = result['type']['value']
                    concepts.append(type_uri.split('/')[-1].split('#')[-1])  # Get the last part of the URL or fragment
                if 'broader' in result:
                    broader_uri = result['broader']['value']
                    concepts.append(broader_uri.split('/')[-1].split('#')[-1])  # Same as above
            return concepts
        except:
            return []

    @staticmethod
    def get_concept_uri(concept_name):
        """
        Fetches the URI for a concept in DBpedia using the concept name.

        :param concept_name: The name of the concept (e.g., "Python (programming language)").
        :return: The URI of the concept in DBpedia if found, None otherwise.
        """
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = f"""
        SELECT ?concept WHERE {{
            ?concept rdfs:label "{concept_name}"@en.
        }}
        LIMIT 1
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(1)

        try:
            results = sparql.query().convert()
            uris = []
            for result in results["results"]["bindings"]:
                uris.append(result["concept"]["value"])
            return uris
        except Exception as e:
            print(f"An error occurred: {e}")

        return []

    @staticmethod
    def get_definitional_sentence(entity_uri, language='en'):
        """
        Fetches the definitional sentence (abstract) of a specified entity from DBpedia based on its URI.

        Parameters:
        - entity_uri: The URI of the entity in DBpedia.
        - language: The language of the abstract (default is English, 'en').

        Returns:
        - The abstract (definitional sentence) of the entity in the specified language, or None if not found.
        """
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = f"""
        SELECT ?abstract WHERE {{
          <{entity_uri}> dbo:abstract ?abstract .
          FILTER (lang(?abstract) = '{language}')
        }}
        LIMIT 1
        """
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(1)
            results = sparql.query().convert()
            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["abstract"]["value"]
            else:
                return None
        except:
            return None


"""uri = 'http://dbpedia.org/resource/Pikmin_2'
result = SearchDBPedia.get_definitional_sentence(uri)
print(result)"""
