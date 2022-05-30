from calendar import c
from BaseModel import BaseModel

class SDBN(BaseModel):
    """
    Simplified Dynamic Bayesian Network model built using MLE approach.
    Parameters:
    - a_qu: attractiveness parameter for each document query pair.
    - s_qu: satisfaction parameter for each document query pair.
    """

    def get_last_clicked(self, search_sessions):
        """
        Outputs a dictionary of last clicked rank each query in a session.
        """
        last_clicked_dict = dict()  

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

                query_dict = {'query_id': query_id, 'last_clicked': last_clicked}
                last_clicked_dict[session].append(query_dict)
        return last_clicked_dict

    def default_doc_query_counts(self, search_sessions, starting_point = (1, 2)):
        """
        The function creates a dict with default clicks for each document query pair. 
        The parameter starting point indicates the default clicks.s
        """
        counts = dict()
        for session in search_sessions:
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                if query_id in counts:
                    for i in range(10):
                        result = query["results"][i]
                        (document, clicks), = result.items()
                        if document in counts[query_id]:
                            pass
                    
                        else: # adding new documents to queries that are already in the counts
                            counts[query_id][document] = [starting_point[0], starting_point[1]]                      
                
                else:
                    counts[query_id] = dict()
                    for i in range(10):
                        result = query["results"][i]
                        (document, clicks), = result.items()
                        counts[query_id][document] = [starting_point[0], starting_point[1]]                 
        return counts

    def get_doc_query_counts(self, search_sessions, last_clicked_dict):
        """
        Calculated the counts for attractiveness and satisfation for each doc query pair.
        """
        attr_counts = self.default_doc_query_counts(search_sessions)
        sat_counts = self.default_doc_query_counts(search_sessions)

        for session in search_sessions:
            for j in range(len(search_sessions[session]["queries"])):
                query = search_sessions[session]["queries"][j]
                query_id = query['query']
 
                last_clicked = last_clicked_dict[session][j]['last_clicked']
                rank = 0 
                while rank <= last_clicked and rank < 10:
                    result = query["results"][rank]
                    (result_id, clicks), = result.items()
                    
                    attr_counts[query_id][result_id][1] += 1
                                                        
                    clicks = result[result_id]
                    if clicks >= 1:
                        attr_counts[query_id][result_id][0] += clicks
                        attr_counts[query_id][result_id][1] += clicks - 1
                        sat_counts[query_id][result_id][1] += clicks
                        if last_clicked == rank:
                            sat_counts[query_id][result_id][0] += clicks

                    rank+=1
                
        return attr_counts, sat_counts

    def train(self, search_sessions):
        last_clicked_dict = self.get_last_clicked(search_sessions)

        a_qu, s_qu = self.get_doc_query_counts(search_sessions, last_clicked_dict)
        
        for query in a_qu:
            for document in a_qu[query]:
                a = a_qu[query][document]
                s = s_qu[query][document]
                a_qu[query][document] = a[0]/a[1]
                s_qu[query][document] = s[0]/s[1]
                
        
        self.params['a_qu'] = a_qu
        self.params['s_qu'] = s_qu

    def get_click_probs_per_result(self, search_sessions):
        a_qu = self.params['a_qu']
        s_qu = self.params['s_qu']

        probs_per_result = dict()
        unseen_attr = 0.5
        unseen_sat = 0.5

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}
                
                p_exam = 1
                for rank in range(len(query["results"])):
                    result = query["results"][rank]
                    (document_id, clicks), = result.items()
                    try:
                        attr = a_qu[query_id][document_id]
                        sat = a_qu[query_id][document_id]
                    except KeyError:
                        attr = unseen_attr
                        sat = unseen_sat
                    query_dict['results'][document_id] = attr * p_exam
                    p_exam *= (1 - sat) * attr + (1 - attr)
                probs_per_result[session].append(query_dict)
        
        return probs_per_result



from preprocessing import get_sessions, train_test_split
if __name__ == "__main__":
    sessions = get_sessions("Clicklog.txt", 1000)
    train_sessions, test_sessions = train_test_split(sessions, 0.75)

    model_sdbn = SDBN()
    model_sdbn.train(train_sessions)

    a_qu = model_sdbn.params['a_qu']
    s_qu = model_sdbn.params['s_qu']
    i = 0
    for query in a_qu:
        for document in a_qu[query]:
            print(f"{query, document}: attr {a_qu[query][document]}, sat: {s_qu[query][document]}")
        i += 1
        if i > 5:
            break


    print(f"Log-likelihood on train sessions: {model_sdbn.log_likelihood(train_sessions)}")
    print(f"Log-likelihood on  test sessions: {model_sdbn.log_likelihood(test_sessions)}")

    print(f"Perplexity on train sessions: {model_sdbn.perplexity(train_sessions)}")
    print(f"Perplexity on  test sessions: {model_sdbn.perplexity(test_sessions)}")



