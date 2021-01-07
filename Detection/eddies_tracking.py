# -*- coding: utf-8 -*-

"""classes.py: This file defines defines the functions used for tracking eddies
position over several days."""

__author__     = "G. Ghienne, A. Lefebvre, A. Lerosey, L. Menard, A. Perier"
__date__       = "December 2020"
__version__    = "1.0"
__maintainer__ = "Not maintened"
__email__      = ["guillaume.ghienne@imt-atlantique.net",
                  "alexandre.lefebvre@imt-atlantique.net",
                  "antoine.lerosey@imt-atlantique.net",
                  "luc.menard@imt-atlantique.net",
                  "alexandre.perier@imt-atlantique.net"]


import numpy as np


def eddies_tracker(initial_date, list_days, metric_name='Jaccard',nb_prev_day=5,min_detection=5):
    """ Identify which eddies are the same over several days.

    Args:
        initial_date (int) : Date of the first day in list_days.
        list_days (list of list of classes.Eddy) : The first index is a date
            relative to the parameter 'date'. The elements are the list of
            detected eddies at the corresping data.
        metric_name (str, default='Jaccard') : The metric to use for matching
            eddies position on consecutive days.
        nb_prev_day (int, default=2): The maximum number of days to look
            backward in already classified eddies for matching previous
            positions. It must be strictly positive. As some eddies might not be
            detected for several consecutive days, an eddy can be matched with a
            previous position several days before. The default value has been
            choosen with the assumption that an eddy that last several days will
            not be not detected for more that one day.
        min_detection (int, default=5) : Minimum number of observation of an
            eddy. Eddies with too view observations are discarded 

    Returns:
        eddies_path (dict(int : dict(int : classes.Eddy))) : The path of eddies.
        A unique id is assigned to each eddy so that 2 object 'classes.Eddy'
        share the same id if and only if they represent the same eddy at
        different dates. The first key is a date, the second key is the eddy
        identifier.

    """

    eddies_path    = {}
    current_max_id = -1
    nb_days        = len(list_days)

    # Select the metric
    if metric_name=='Jaccard':
        from metrics import eddies_jaccard_index as metric
    else:
        print("eddies_tracker: invalid metric")
        return {}

    # Init the paths
    if nb_days == 0:
        print("eddies_tracker: empty list_days")
        return {}
    eddies_path[initial_date] = {}
    for eddy in list_days[0]:
        current_max_id += 1
        eddies_path[initial_date][current_max_id] = eddy

    # Handle the first days separatly (as part of the initialisation)
    for day in range(1,nb_prev_day):
        if nb_days == day:
            return eddies_path
        current_max_id += track_one_day(initial_date+day,eddies_path,list_days[day],
                                    current_max_id,metric,nb_prev_day=day)

    # Match positions day by day
    for day in range(nb_prev_day,nb_days):
        current_max_id += track_one_day(initial_date+day,eddies_path,
                                        list_days[day],current_max_id,
                                        metric,nb_prev_day=nb_prev_day)

    # Delete eddies observed too view times
    nb_observation = np.zeros((current_max_id+1))
    for day in eddies_path.keys():
        for eddy_id in eddies_path[day].keys():
            nb_observation[eddy_id] += 1
    remove_eddy = nb_observation<min_detection
    for day in eddies_path.keys():
        list_keys = list(eddies_path[day].keys())
        for eddy_id in list_keys:
            if remove_eddy[eddy_id]:
                del eddies_path[day][eddy_id]

    return eddies_path


def track_one_day(date,current_eddies_path,eddies_in_day,current_max_id,metric,nb_prev_day=2):
    """ Assign an id to the eddies of a day using the classification of previous
    days.

    The eddies position of a given day are matched with the positions of the
    previous days. The eddies of the day are added to the current paths.

    Args:
        date (int) : Date of the day to be added in the paths.
        current_eddies_path (dict(int : dict(int : classes.Eddy))) : The paths
            of eddies during the previous days. The first key is the date, the
            second key is the eddy identifier. The paths are modified
            'inplace' and the results are assigned to a new date entry.
        eddies_in_day (list of classes.Eddy) : the list of detected eddies at
            the date 'data'.
        current_max_id (int) : The current maximum of identifier used for the
            eddies. This parameter is used when a new eddy is detected. An eddy
            is considered to be new if it has not been matched with any previous
            eddy position.
        metric (function(e1, e2 :classes.Eddy) = score: float) : The metric
            function to use for matching eddies positions. The score must
            increase with the similarity of the 2 eddies.
        nb_prev_day (int, default=2): The maximum number of days to look
            backward in already classified eddies for matching previous
            positions. It must be strictly positive. As some eddies might not be
            detected for several consecutive days, an eddy can be matched with a
            previous position several days before. The default value has been
            choosen with the assumption that an eddy that last several days will
            not be not detected for more that one day. 

    Returns:
        nb_new_eddies (int) : The number of new eddies.

    """

    current_eddies_path[date] = {}
    nb_eddies =  len(eddies_in_day)
    nb_new_eddies = 0


    # Compute the best distance between current eddies and the eddies of
    # previous days.
    distances = np.zeros((nb_eddies,nb_prev_day,2))

    for eddy_id in range(nb_eddies):
        for date_offset in range(0,nb_prev_day):
            best_score = 0
            best_id    = -1
            for key in current_eddies_path[date-1-date_offset].keys():
                score = metric(eddies_in_day[eddy_id],
                               current_eddies_path[date-1 - date_offset][key])
                if score > best_score:
                    best_score = score
                    best_id    = key
            distances[eddy_id,date_offset,:] = np.array([best_id,best_score])

    # Set of id of eddies observed during the previous day(s).
    previous_eddies = set()
    for date_offset in range(1,nb_prev_day+1):
        prev_eddies_day = set(current_eddies_path[date - date_offset].keys())
        previous_eddies = previous_eddies.union(prev_eddies_day)
    previous_eddies = list(previous_eddies)


    is_matched = np.zeros(nb_eddies, dtype=bool)
    # Try to match current eddies with those previously observed.
    for previous_eddy in previous_eddies:
        for date_offset in range(0,nb_prev_day):
            # Get the id of all current eddies which have their heighest score
            # with previous_eddy.
            match = np.argwhere(distances[:,date_offset,0]==previous_eddy)[:,0]
            if len(match)!=0:
                # If several current eddies match this previous eddy, the one
                # with the highest score is selected.
                match_id = match[np.argmax(distances[match,date_offset,1])]
                if not is_matched[match_id]:
                    is_matched[match_id] = True
                    current_eddies_path[date][previous_eddy]=eddies_in_day[match_id]
                    break # End the first 'for' loop.

    # The remaining not matched eddues are new eddies
    new_eddies_id = np.argwhere(~is_matched)[:,0]
    for new_eddy_id in new_eddies_id:
        nb_new_eddies  += 1
        current_max_id += 1
        current_eddies_path[date][current_max_id] = eddies_in_day[new_eddy_id]

    return nb_new_eddies