from BaseModel import BaseModel

class DCM(BaseModel):
    """
    Dependant Click Model implemented using MLE approach.
    Parameters:
    - a_qu: a dictionary of document query pairs that contain attractiveness parameter.
    - lambda_r: 1x10 array, a continuation parameter at each rank.
    """

    def find_counts(self, search_sessions):
        """
        The function calculated:
        - last_clicked_dict: the dictionary of last clicked rank for each document query pair
        - click_counts: the click count for each rank as the last clicked rank
        - stop_counts: the clicks on each rank as the last one

        """
        last_clicked_dict = dict()
        click_counts = [0] * 10   

        for session in search_sessions:
            last_clicked_dict[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                last_clicked = 10
                for rank in range(len(query["results"])):
                    result = query["results"][rank]
                    (result_id, clicks), = result.items()
                    if clicks >= 1:
                        last_clicked = rank
                    click_counts[rank] += clicks
                
                query_dict = {'query_id': query_id, 'last_clicked': last_clicked}
                last_clicked_dict[session].append(query_dict)

        stop_counts = [0] * 11
        for session in last_clicked_dict:
            for query_dict in last_clicked_dict[session]:
                stop_counts[query_dict['last_clicked']] += 1

        return last_clicked_dict, click_counts, stop_counts


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
                                counts[query_id][document] = [starting_point[0], starting_point[1]]                      
                
                else:
                    counts[query_id] = dict()
                    for i in range(10):
                        result = query["results"][i]
                        for document in result:
                            counts[query_id][document] = [starting_point[0], starting_point[1]]                 
        return counts

    def get_document_query_counts(self, search_sessions, last_clicked_dict):
        """
        Calculated the number of times each document query pair was clicked and encountered.
        """
        counts = self.default_doc_query_counts(search_sessions)

        for session in search_sessions:
            for j in range(len(search_sessions[session]["queries"])):
                query = search_sessions[session]["queries"][j]
                query_id = query['query']
 
                last_clicked = last_clicked_dict[session][j]['last_clicked']
                i = 0 
                while i <= last_clicked and i < 10:
                    result = query["results"][i]
                    for result_id in result:
                        counts[query_id][result_id][1] += 1
                                                        
                        clicked = result[result_id]
                        if clicked == 1:
                            counts[query_id][result_id][0] += clicked
                            
                        elif clicked > 1:
                            counts[query_id][result_id][0] += clicked
                            counts[query_id][result_id][1] += clicked - 1
                        i+=1
                
        return counts

    def get_click_probs_per_result(self, search_sessions):
        a_qu = self.params['a_qu']
        lambda_r = self.params['lambda_r']

        probs_per_result = dict()
        unseen_attract = 0.5 # attractiveness of documents that have not been seen

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}

                p_exam = 1 # probability of the result to be examined
                for rank in range(len(query["results"])):
                    result = query["results"][rank]
                    for result_id in result:
                        try:
                            prob = a_qu[query_id][result_id] * p_exam
                        except KeyError:
                            prob = unseen_attract * p_exam
                            
                        query_dict['results'][result_id] = prob

                    p_exam = lambda_r[rank] * prob + p_exam - prob

                probs_per_result[session].append(query_dict)

        return probs_per_result
    
    def train(self, search_sessions):
        last_clicked_dict, click_counts, stop_counts = self.find_counts(search_sessions)
        lambda_r = []
        for i in range(10):
            lambda_r.append(1 - (stop_counts[i] / click_counts[i]))
        
        self.params['lambda_r'] = lambda_r

        a_qu = self.get_document_query_counts(search_sessions, last_clicked_dict)
        for query in a_qu:
            for document in a_qu[query]:
                values = a_qu[query][document]
                a_qu[query][document] = values[0]/values[1]
        
        self.params['a_qu'] = a_qu




