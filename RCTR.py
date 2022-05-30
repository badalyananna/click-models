from BaseModel import BaseModel
import numpy as np

class RCTR(BaseModel):
    """
    Contains 1 parameter p, which is the probability of a click at each rank.
    """
    def train(self, search_sessions):
        results = 0
        clicks = np.zeros(10)
        for session in search_sessions:
            for query in search_sessions[session]['queries']:
                i = 0
                for result in query['results']:
                    (key, value), = result.items()
                    clicks[i] += value
                    i += 1
            results += search_sessions[session]['stats']['results'] 
        self.params['p'] = clicks * 10 / results


    def get_click_probs_per_result(self, search_sessions):
        
        p = self.params['p']

        probs_per_result = dict()

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}
                for i in range(len(query["results"])):
                    result = query["results"][i]
                    for result_id in result:
                        query_dict['results'][result_id] = p[i]
                probs_per_result[session].append(query_dict)
                
        return probs_per_result