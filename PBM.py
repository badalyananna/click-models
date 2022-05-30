from BaseModel import BaseModel

class PBM(BaseModel):
    """
    Position Based Model implemented using the EM algorithm.
    Parameters:
    - a_qu: a dictionary of document query pairs that contain attractiveness parameter.
    - Gammar_r: 1x10 array, the examination parameter for each rank. 
    """

    def default_doc_query_counts(self, search_sessions, starting_point = 0.5):
        """
        Creates a dictionary with a starting parameter alpha for each doc query pair and the list of sessions where this document query pair was encountered.
        In the list of sessions, each session contains information on the rank of the document in a given session and the clicks statistics.
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
                                session_stats = {'id': session, 'rank': i, 'clicks': result[document]}
                                counts[query_id][document]['sessions'].append(session_stats)
                        
                            else: # adding new documents to queries that are already in the counts
                                session_stats = {'id': session, 'rank': i, 'clicks': result[document]}
                                counts[query_id][document] = {'alpha': starting_point, 'sessions': [session_stats]}                   
                
                else:
                    counts[query_id] = dict()
                    for i in range(10):
                        result = query["results"][i]
                        for document in result:
                            session_stats = {'id': session, 'rank': i, 'clicks': result[document]}
                            counts[query_id][document] = {'alpha': starting_point, 'sessions': [session_stats]}              
        return counts

    def train(self, search_sessions):
        alphas = self.default_doc_query_counts(search_sessions)
        gammas = [0.5] * 10

        iterations = 50
        for _ in range(iterations):
            new_alphas = dict()
            for query in alphas:
                new_alphas[query] = dict()
                for document in alphas[query]:
                    q_u = alphas[query][document]
                    alpha_new = 1
                    for session in q_u['sessions']:
                        if session['clicks'] >= 1:
                            alpha_new += 1
                        else:
                            alpha_new += (1 - gammas[session['rank']]) * q_u['alpha'] / (1 - gammas[session['rank']] * q_u['alpha'])
                    new_alphas[query][document] = alpha_new / (len(q_u['sessions']) + 2)
            
            new_gammas = [1] * 10
            total_queries = 2
            for session in search_sessions:
                for query in search_sessions[session]["queries"]:
                    total_queries += 1
                    for rank in range(10):
                        result = query["results"][rank]
                        (document, clicks), = result.items()
                        if clicks >= 1:
                            new_gammas[rank] += 1
                        else:
                            new_gammas[rank] += (1 - alphas[query['query']][document]['alpha']) * gammas[rank] / (1 - gammas[rank] * alphas[query['query']][document]['alpha'])
            
            # updating the parameters
            for i in range(10):
                gammas[i] = new_gammas[i] / total_queries

            for query in alphas:
                for document in alphas[query]:
                    alphas[query][document]['alpha'] = new_alphas[query][document]

        self.params['a_qu'] = new_alphas
        self.params['gamma_r'] = gammas

    def get_click_probs_per_result(self, search_sessions):
        a_qu = self.params['a_qu']
        gamma_r = self.params['gamma_r']

        probs_per_result = dict()
        unseen_attract = 0.5

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}

                for rank in range(len(query["results"])):
                    p_exam = gamma_r[rank]
                    result = query["results"][rank]
                    (result_id, clicks), = result.items()
                    try:
                        prob = a_qu[query_id][result_id] * p_exam
                    except KeyError:
                        prob = unseen_attract * p_exam

                    query_dict['results'][result_id] = prob
                probs_per_result[session].append(query_dict)
        
        return probs_per_result

from preprocessing import get_sessions, train_test_split
if __name__ == "__main__":
    sessions = get_sessions("Clicklog.txt", 1000)
    train_sessions, test_sessions = train_test_split(sessions, 0.75)

    model_pbm = PBM()
    model_pbm.train(train_sessions)
    print(model_pbm.params['gamma_r'])

    print(f"Log-likelihood on train sessions: {model_pbm.log_likelihood(train_sessions)}")
    print(f"Log-likelihood on  test sessions: {model_pbm.log_likelihood(test_sessions)}")
    print(f"Perplexity on train sessions: {model_pbm.perplexity(train_sessions)}")
    print(f"Perplexity on  test sessions: {model_pbm.perplexity(test_sessions)}")
