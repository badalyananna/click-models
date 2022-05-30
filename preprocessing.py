import itertools

def get_sessions(filename, max_sessions, starting_session=0):
    """ 
    The function reads the clicklogs file and outputs the dictionary of sessions: 
    filename - link to the clicklogs file
    max_sessions - number of sessions to retrieve
    starting_sessions - number of sessions to skip in the beginning
    """
    lines_to_skip=5
    sessions = dict()
    with open(filename) as f:
        i = 0
        n_sessions = 0
        for line in f:
            # we pass the first session because it has missing data in our file
            if i < lines_to_skip:
                pass
                i += 1
            else:
                if n_sessions > max_sessions:
                    sessions.pop(session_id) # dropping the last session as it is not complete
                    break
                else:
                    line = line.split()
                    session_id = int(line[0])
                    if session_id in sessions and n_sessions >= starting_session:
                        if line[2] == 'Q':
                            res = [{line[5:][idx]: 0} for idx in range(0, len(line[5:]))]
                            sessions[session_id]['queries'].append({'query': int(line[3]), 'results': res})
                            sessions[session_id]['stats']['results'] +=10
                        elif line[2] == 'C':
                            clicks_count = 0
                            for j in range(3, len(line)):
                                my_dicts = sessions[session_id]['queries'][-1]['results']
                                for d in my_dicts:
                                    d.update((k, v+1) for k, v in d.items() if k == line[j])
                                clicks_count += 1
                            sessions[session_id]['stats']['clicks'] += clicks_count
                    else:
                        if line[2] == 'Q':
                            n_sessions += 1
                            if n_sessions >= starting_session:
                                res = [{line[5:][idx]: 0} for idx in range(0, len(line[5:]))]
                                data = {'queries': [{'query': int(line[3]), 'results': res}],
                                    'stats': {'results': 10,
                                                'clicks': 0}} 
                                sessions[session_id] = data
    return sessions

def train_test_split(sessions, train_size, filtered=True):
    """
    The function takes the sessions dictionary and splits them into train_sessions and test_sessions.
    
    Input:
    sessions - the dictionary of sessions
    train_size - the size of the final train_sessions file, takes values from 0 to 1
    filtered - if true, the filtered parameter filters the train set to contain only the sessions with document query pairs 
               that exist in the train set, otherwise the sessions are split without filtering.
    """
    if filtered:
        train_query_docs = dict()
        train_sessions = dict()
        test_sessions = dict()
        max_test_entries = int(len(sessions)*(1-train_size))
        for session in sessions:
            if len(test_sessions) < max_test_entries:
                all_query_docs_in_train = True
                for query in sessions[session]["queries"]:
                    query_id = query['query']
                    if query_id in train_query_docs:
                        for result in query['results']:
                            (result_id, clicks), = result.items()
                            if result_id in train_query_docs[query_id]:
                                pass
                            else:
                                all_query_docs_in_train = False
                                train_query_docs[query_id].add(result_id)

                    else:
                        all_query_docs_in_train = False
                        train_query_docs[query_id] = set()
                        for result in query['results']:
                            (result_id, clicks), = result.items()
                            train_query_docs[query_id].add(result_id)
            
                if all_query_docs_in_train:
                    test_sessions[session] = sessions[session]
                else:
                    train_sessions[session] = sessions[session]
            else:
                train_sessions[session] = sessions[session]

    else: 
        total = len(sessions)
        i = iter(sessions.items())
        train_sessions = dict(itertools.islice(i, int(total * train_size)))
        test_sessions = dict(i)
    return train_sessions, test_sessions

def n_unique_queries(sessions):
    """
    Calculates the number of unique queries in the sessions.
    """
    unique_sessions = set()
    for session in sessions:
        for query in sessions[session]["queries"]:
            query_id = query['query']
            unique_sessions.add(query_id)
    return len(unique_sessions)

def total_queries(sessions):
    """
    Calculates the total number of queries in the sessions.
    """
    queries=0
    for session in sessions:
        for query in sessions[session]["queries"]:
            queries+=1
    return queries

if __name__ == "__main__":
    sessions = get_sessions("dataset/Clicklog.txt", 100000)
    print(total_queries(sessions))
    print(n_unique_queries(sessions))