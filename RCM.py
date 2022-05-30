from BaseModel import BaseModel

class RCM(BaseModel):
    """
    Basic CRT model. Contains 1 parameter p, which is the probability of a random click.
    """
    
    def train(self, search_sessions):
        """
        Finds parameter p for the RCM model.
        """
        results = 0
        clicks = 0
        for session in search_sessions:
            results += search_sessions[session]['stats']['results']
            clicks += search_sessions[session]['stats']['clicks']
    
        self.params['p'] = clicks/results

    def get_click_probs_per_result(self, search_sessions):
        
        p = self.params['p']

        probs_per_result = dict()

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}
                for result in query["results"]:
                    for result_id in result:
                        query_dict['results'][result_id] = p
                probs_per_result[session].append(query_dict)
                
        return probs_per_result