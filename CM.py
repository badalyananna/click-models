from BaseModel import BaseModel

class CM(BaseModel):
    """
    Cascade model implemented using the MLE approach.
    Parameters:
    - a_qu: a dictionary of document query pairs that contain attractiveness parameter.
    """

    def default_doc_query_counts(self, search_sessions, starting_point = (1, 2)):
        """
        The function creates a dict with default clicks for each document query pair. 
        The parameter starting point indicates the default clicks.
        """
        counts = dict()
        for session in search_sessions:
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                if query_id in counts:
                    for i in range(10):
                        result = query["results"][i]
                        for document in result:
                            if document in counts[query_id]:
                                pass
                        
                            else: # adding new documents to queries that are already in the counts
                                (document, clicks), = result.items()
                                counts[query_id][document] = [starting_point[0], starting_point[1]]                      
                
                else:
                    counts[query_id] = dict()
                    for i in range(10):
                        result = query["results"][i]
                        (document, clicks), = result.items()
                        counts[query_id][document] = [starting_point[0], starting_point[1]]              
        return counts

    def get_document_query_counts(self, search_sessions):
        """
        Calculated the number of times each document query pair was clicked and encountered.
        """
        counts = self.default_doc_query_counts(search_sessions)

        for session in search_sessions:
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
 
                click_encountered = 0
                i = 0 
                while click_encountered == 0 and i < 10:
                    result = query["results"][i]
                    (result_id, clicks), = result.items()
                    
                    counts[query_id][result_id][1] += 1 # adding 1 to encountered count
                    
                    # some documents were clicked more than once, but we will store only the first click, because of the limitations of the model
                    if clicks >= 1:
                        clicks = 1
                        counts[query_id][result_id][0] += 1
                    i+=1
                    
                    click_encountered = clicks
                
        return counts

    def train(self, search_sessions):
        a_qu = self.get_document_query_counts(search_sessions)
        for query in a_qu:
            for document in a_qu[query]:
                values = a_qu[query][document]
                a_qu[query][document] = values[0]/values[1]

        self.params['a_qu'] = a_qu

    def get_click_probs_per_result(self, search_sessions):
        a_qu = self.params['a_qu']

        probs_per_result = dict()
        unseen_attract = 0.5 # attractiveness of documents that have not been seen

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}

                p_exam = 1 # probability of the result to be examined
                for result in query["results"]:
                    (result_id, clicks), = result.items()
                    
                    try:
                        prob = a_qu[query_id][result_id] * p_exam
                    except KeyError:
                        prob = unseen_attract * p_exam
                            
                    query_dict['results'][result_id] = prob

                    p_exam = 1 - prob

                probs_per_result[session].append(query_dict)

        return probs_per_result
