from BaseModel import BaseModel

class UBM(BaseModel):
    """
    User Brousing Model implemented using the EM algorithm.
    Parameters:
    - a_qu: a dictionary of document query pairs that contain attractiveness parameter.
    - Gammar_rr: 10x10 array of all possible currect_rank and previously_clicked_rank pairs, where the last rank (9) is used to indicate that no document has been clicked before.    
    """
    def default_doc_query_counts(self, search_sessions, starting_point = 0.5):
        """
        Creates a dictionary with a starting parameter alpha for each doc query pair and the list of sessions where this document query pair was encountered.
        In the list of sessions, each session contains information on the rank of the document and the rank of a previouslcy clicked  document in a given session and the clicks statistics.
        """
        counts = dict()
        for session in search_sessions:
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                if query_id in counts:
                    last_clicked = 9 # as tha last document cannot be the previously clicked, we use 9 to decode that no document was clicked before
                    for i in range(10):
                        result = query["results"][i]
                        for document in result:
                            if document in counts[query_id]:
                                session_stats = {'id': session, 'rank': i, 'clicks': result[document], 'last_clicked': last_clicked}
                                counts[query_id][document]['sessions'].append(session_stats)
                        
                            else: # adding new documents to queries that are already in the counts
                                session_stats = {'id': session, 'rank': i, 'clicks': result[document], 'last_clicked': last_clicked}
                                counts[query_id][document] = {'alpha': starting_point, 'sessions': [session_stats]}  
                            # chacked if the document was clicked; if yes, change the last clicked rank number
                            if result[document] >= 1:
                                last_clicked = i                 
                
                else:
                    # if the query has not been encountered, create a dictionary which will contain the docs
                    counts[query_id] = dict()
                    last_clicked = 9
                    for i in range(10):
                        result = query["results"][i]
                        (document, clicks), = result.items()
                        session_stats = {'id': session, 'rank': i, 'clicks': result[document], 'last_clicked': last_clicked}
                        counts[query_id][document] = {'alpha': starting_point, 'sessions': [session_stats]}
                        if clicks >= 1:
                            last_clicked = i                 
        return counts

    def train(self, search_sessions):
        alphas = self.default_doc_query_counts(search_sessions)
        gammas = [[0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10]

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
                            alpha_new += (1 - gammas[session['last_clicked']][session['rank']]) * q_u['alpha'] / (1 - gammas[session['last_clicked']][session['rank']] * q_u['alpha'])
                    new_alphas[query][document] = alpha_new / (len(q_u['sessions']) + 2)

            new_gammas = [[0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10, [0.5] * 10]
            total_queries = [[2] * 10, [2] * 10, [2] * 10, [2] * 10, [2] * 10, [2] * 10, [2] * 10, [2] * 10, [2] * 10, [2] * 10]

            for session in search_sessions:
                for query in search_sessions[session]["queries"]:
                    last_clicked = 9
                    for rank in range(10):
                        result = query['results'][rank]
                        (document, clicks), = result.items()
                        total_queries[last_clicked][rank] += 1
                        if clicks >= 1:
                            new_gammas[last_clicked][rank] += 1
                            last_clicked = rank
                        else:
                            new_gammas[last_clicked][rank] += (1 - alphas[query['query']][document]['alpha']) * gammas[last_clicked][rank] / (1 - gammas[last_clicked][rank] * alphas[query['query']][document]['alpha'])           
            
            # updating the parameters
            for j in range(10):
                for i in range(10):
                    gammas[j][i] = new_gammas[j][i] / total_queries[j][i]

            for query in alphas:
                for document in alphas[query]:
                    alphas[query][document]['alpha'] = new_alphas[query][document]

        self.params['a_qu'] = new_alphas
        self.params['gamma_rr'] = gammas

    def get_click_probs_per_result(self, search_sessions):
        a_qu = self.params['a_qu']
        gamma_rr = self.params['gamma_rr']

        probs_per_result = dict()
        unseen_attract = 0.5

        for session in search_sessions:
            probs_per_result[session] = []
            for query in search_sessions[session]["queries"]:
                query_id = query['query']
                query_dict = {'query_id': query_id, 'results': {}}
                
                last_clicked = 9
                for rank in range(len(query["results"])):
                    p_exam = gamma_rr[last_clicked][rank]
                    result = query["results"][rank]
                    (document_id, clicks), = result.items()
                    try:
                        prob = a_qu[query_id][document_id] * p_exam
                    except KeyError:
                        prob = unseen_attract * p_exam
                    query_dict['results'][document_id] = prob

                    if clicks >= 1:
                        last_clicked = rank

                probs_per_result[session].append(query_dict)
        
        return probs_per_result


from preprocessing import get_sessions, train_test_split
if __name__ == "__main__":
    sessions = get_sessions("Clicklog.txt", 1000)
    train_sessions, test_sessions = train_test_split(sessions, 0.75)

    model_ubm = UBM()
    model_ubm.train(train_sessions)
    print(f"Log-likelihood on train sessions: {model_ubm.log_likelihood(train_sessions)}")
    print(f"Log-likelihood on  test sessions: {model_ubm.log_likelihood(test_sessions)}")

    print(f"Perplexity on train sessions: {model_ubm.perplexity(train_sessions)}")
    print(f"Perplexity on  test sessions: {model_ubm.perplexity(test_sessions)}")
