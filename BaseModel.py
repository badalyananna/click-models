from abc import abstractmethod
import math
import numpy as np

class BaseModel():
    """
    Base class for a click model. Contains methods:
    train: calculates model parameters for the given the search sessions (implemented later for each model).
    get_click_probs_per_result: finds the full probabilities for given sessions using the already calculated model parameters.
        (implemented later for each model)
    log_likelihood: calculates log_likelihood using the click probabilities per result
    perplexity: calculates perplexity using the click probabilities per result
    """

    def __init__(self):
        self.params = dict()

    @abstractmethod
    def train(self, search_sessions):
        """
        The function calculates the model parameters.
        """
        pass

    @abstractmethod
    def get_click_probs_per_result(self, search_sessions):
        """
        The function outputs a dictionary of full click probabilities for each session, query and result.
        """
        pass

    def log_likelihood(self, search_sessions):
        """
        Calculates log_likelihood for a model with trained parameters.
        Input:
        search_sessions used to evaluate the parameter
        """
        probs_per_result = self.get_click_probs_per_result(search_sessions)
        log_like = 0
        results = 0
        for session in search_sessions:
            p_session = 1
            for i in range(len(search_sessions[session]["queries"])):
                query = search_sessions[session]["queries"][i]
                results += 10
                for result in query["results"]:
                    for result_id in result:
                        clicked = result[result_id]
                        
                        p = probs_per_result[session][i]['results'][result_id]

                        if clicked >= 1:
                            p_session *= (p ** clicked)
                        else:
                            p_session *= (1 - p)
            log_like += math.log(p_session) 
        return log_like / results


    def perplexity_per_rank(self, search_sessions):
        """
        Calculates perplexity per rank for a model with trained parameters.
        Input:
        search_sessions used to evaluate the parameter
        Output:
        list with a perplexity for each rank
        """
        probs_per_result = self.get_click_probs_per_result(search_sessions)

        powers = np.zeros(10)
        queries = 0
        for session in search_sessions:
            for i in range(len(search_sessions[session]["queries"])):
                query = search_sessions[session]["queries"][i]
                results = query['results']
                queries += 1
                for rank in range(len(results)):
                    for result_id in results[rank]:
                        clicked = results[rank][result_id]

                        p = probs_per_result[session][i]['results'][result_id]

                        if clicked >= 1:
                            powers[rank] += math.log2(p**clicked)
                        else:
                            powers[rank] += math.log2(1-p)

        probabilities = 2 ** (-powers / queries)
        return probabilities

    def perplexity(self, search_sessions):
        """
        Calculates the overall perplexity.
        """
        probabilities = self.perplexity_per_rank(search_sessions)
        return probabilities.mean()

