from BaseModel import BaseModel
from RCM import RCM

class DCTR(BaseModel):
    """
    Document Click Throught Rate calculated using MLE approach.
    Parameters:
    p_qu: probability of a click for each document query pair
    """

    def get_document_query_counts(self, search_sessions):
        """
        The function calculates document query counts and outputs a list [total_clicked, total_encountered] for each document query pair.
        """
        counts = dict()
        for session in search_sessions:
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                if query_id in counts:
                    for i in range(10):
                        result = query["results"][i]
                        (document, clicks), = result.items()
                        if document in counts[query_id]: # updating counts of the document query pair
                            counts[query_id][document][1] += 1 # adding 1 to encountered counts

                            if clicks >= 1:
                                clicks = 1
                            counts[query_id][document][0] += clicks
                    
                        else: # adding new documents to queries that are already in the counts
                            clicks += 1
                            # some documents were clicked more than once, but we will store only the first click, because of the limitations of the model
                            if clicks >= 2:
                                clicks = 2
                            counts[query_id][document] = [clicks, 3]

                
                else:
                    counts[query_id] = dict()
                    for i in range(10):
                        result = query["results"][i]
                        (document, clicks), = result.items()
                        # some documents were clicked more than once, but we will store only the first click, because of the limitations of the model
                        clicks += 1
                        if clicks >= 2:
                            clicks = 2
                        counts[query_id][document] = [clicks, 3]                 
        return counts

    def train(self, search_sessions):
        p_qu = self.get_document_query_counts(search_sessions)
        for query in p_qu:
            for document in p_qu[query]:
                values = p_qu[query][document]
                p_qu[query][document] = values[0]/values[1]

        self.params['p_qu'] = p_qu

    def get_click_probs_per_result(self, search_sessions):
        p_qu = self.params['p_qu']
        
        # take the probability for unseen data using the rcm model
        rcm_model = RCM()
        rcm_model.train(search_sessions)
        unseen_qu_prob = rcm_model.params['p']

        probs_per_result = dict()

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}
                for result in query["results"]:
                    for result_id in result:
                        try:
                            prob = p_qu[query_id][result_id]
                        except KeyError:
                            prob = unseen_qu_prob
                        query_dict['results'][result_id] = prob
                probs_per_result[session].append(query_dict)
                
        return probs_per_result


from preprocessing import get_sessions, train_test_split

if __name__ == "__main__":
    sessions = get_sessions("Clicklog.txt", 1000)
    train_sessions, test_sessions = train_test_split(sessions, 0.75)
    model_dctr = DCTR()
    model_dctr.train(train_sessions)
    print(model_dctr.params)