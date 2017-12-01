""" tracker/metrics.py

Author: Daniel Zurawski
Organization: Fermilab
Grammar: Python 3.6.1
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from . import extractor as ext, utils
import math


def number_of_hits(
        frame : pd.DataFrame,
        ) -> int:
    """ Get the number of hits.
    :param frame:
        A pd.DataFrame
    :return:
        The number of hits within this "frame". Padding does not count as a
        valid hit, but noisy hits do count.
    """
    if {"padding"}.issubset(frame.columns):
        return len(utils.remove_padding(frame))
    else:
        return len(frame)


def number_of_tracks(
        frame : pd.DataFrame,
        noise_counts_as_a_track : bool = False,
        ) -> int:
    """ Get the number of tracks.
    :param frame:
        A pd.DataFrame
    :param noise_counts_as_a_track:
        If True, then noise counts as a track.
    :return:
        The number of tracks within this "frame". Padding does not count as a
        track. Noise may or may not count as a track.
    """
    if {"padding"}.issubset(frame.columns):
        frame = utils.remove_padding(frame)
    # noinspection PyTypeChecker
    if {"noise"}.issubset(frame.columns) and not noise_counts_as_a_track:
        frame = utils.remove_noise(frame)
    return len(frame.groupby(["event_id", "cluster_id"]))


def number_of_crossings(
        frame : pd.DataFrame,
        ) -> int:
    """ Return the number of times two separate tracks cross each other within
    this event.
    :param frame: A pd.DataFrame representing an event of many tracks.
    :return: The number of times two separate tracks cross each other within
    this frame's event.
    """
    frame   = utils.remove_noise(utils.remove_padding(frame))
    tracks  = utils.list_of_groups(frame, "cluster_id")
    crosses = 0
    for i in range(len(tracks)):
        for j in range((i + 1), len(tracks)):
            crosses += tracks_crossed(tracks[i], tracks[j])
    return crosses


def distributions(
        frame : pd.DataFrame,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Return unique number of tracks per event and number of occurrences.
    :param frame:
        A pd.DataFrame.
    :return:
        A tuple of two np.ndarrays. The first np.ndarray contains a sorted
        array of the unique number of tracks within an event. The second
        np.ndarray contains an array of the number of events with that
        unique number of tracks.
        Example output: np.array(2, 5, 6), np.array(5, 4, 1)
        Translates to:
        (5 events with 2 tracks),
        (4 events with 5 tracks),
        (6 tracks with 1 track)
    """
    if {"padding"}.issubset(frame.columns):
        frame = utils.remove_noise(utils.remove_padding(frame))
    events = utils.list_of_groups(frame, group="event_id")
    sizes  = np.array([number_of_tracks(event) for event in events])
    return tuple(np.unique(sizes, return_counts=True))


def discrete(
        matrix : np.ndarray
        ) -> np.ndarray:
    """ Get a probability matrix of discrete (0 or 1) values.
    :param matrix:
        The np.ndarray that will be made discrete.
    :return:
        A discrete np.ndarray. All values are either 0 or 1. All rows
        add up to exactly 1.
    """
    return utils.to_categorical(utils.from_categorical(matrix))


def threshold(
        matrix : np.ndarray,
        thresh : float
        ) -> np.ndarray:
    """ Get a discrete matrix such that values above threshold are 1'd.
    :param matrix:
        The input matrix
    :param thresh:
        A threshold value. Values below this are 0'd. Values greater than or
        equal to this are made into 1's.
    :return:
        The threshold matrix.
    """
    threshold_matrix = np.copy(matrix)
    threshold_matrix[thresh >  matrix] = 0
    threshold_matrix[thresh <= matrix] = 1
    return threshold_matrix


def percent_of_hits_assigned_correctly(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : Union[np.ndarray, List[np.ndarray]],
        order   : List[str],
        do_not_factor_in_padding : bool = True,
        do_not_factor_in_noise   : bool = False,
        ) -> float:
    """ Get the percent of hits that were assigned to the correct track.
    :param frames:
        A list of frames.
    :param guesses:
        A list of prediction matrices
    :param order:
        A permutation of ("phi", "r", "z")
    :param do_not_factor_in_padding:
        If True, then do not factor in the correct or incorrect categorization
        of padding.
    :param do_not_factor_in_noise:
        If True, then do not factor in the correct or incorrect categorization
        of noise.
    :return:
        The percent of hits that were assigned to the correct track.
    """
    if isinstance(frames, pd.DataFrame):
        return percent_of_hits_assigned_correctly(
                [frames], [guesses], order,
                do_not_factor_in_padding, do_not_factor_in_noise)
    n_hits, n_correct = 0, 0
    for i, guess in enumerate(guesses):
        guess = utils.from_categorical(guess)
        frame = frames[i]
        if do_not_factor_in_padding:
            guess  = utils.remove_padding(frame, guess, order)
            frame  = utils.remove_padding(frame)
        if do_not_factor_in_noise:
            guess  = utils.remove_noise(frame, guess, order)
            frame  = utils.remove_noise(frame)
        target = ext.extract_output(frame, order, categorical=False)
        n_correct += np.equal(guess, target).sum()
        n_hits    += len(guess)
    return n_correct / n_hits


def percent_of_events_with_correct_number_of_tracks(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : List[np.ndarray],
        order   : List[str],
        ) -> float:
    """ Return the percent of events with the correct number of tracks.
    :param frames:
        A list of data frames, each corresponding to an event.
    :param guesses:
        A list of probability matrices
    :param order:
        A permutation of ("phi", "r", "z")
    :return:
        The percent of events such that the number of tracks within that event's
        corresponding guess is equal to the number of tracks that this event
        truly has.
    """
    if isinstance(frames, pd.DataFrame):
        return percent_of_tracks_assigned_correctly(
                utils.list_of_groups(frames, "event_id"), guesses, order)
    n_correct = 0
    for i in range(len(frames)):
        guess  = utils.from_categorical(guesses[i])
        target = ext.extract_output(frames[i], order, categorical=False)
        n_correct += (len(np.unique(guess)) == len(np.unique(target)))
    return n_correct / len(frames)


def percent_of_tracks_assigned_correctly(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : Union[np.ndarray, List[np.ndarray]],
        order   : List[str],
        percent : float = 1.0,
        do_not_factor_in_padding : bool = True,
        do_not_factor_in_noise   : bool = False,
        ) -> float:
    """ Get the percent of tracks that were assigned the correct hits.
    :param frames:
        A set of frames
    :param guesses:
        The set of guess matrices.
    :param order:
        How to order the input data (phi, r, z)
    :param percent:
        For each track, the percent of hits that were correctly assigned to the
        track in order for that track to be considered as correct.
    :param do_not_factor_in_padding:
        If true, the padding column is ignored.
    :param do_not_factor_in_noise:
        If true, the noise column is ignored.
    :return:
        The percent of tracks among all tracks that were classified correctly.
    """
    if isinstance(frames, pd.DataFrame):
        return percent_of_tracks_assigned_correctly(
                [frames], [guesses], order, percent,
                do_not_factor_in_padding, do_not_factor_in_noise)
    n_tracks, n_correct = 0, 0
    for i, guess in enumerate(guesses):
        frame = frames[i]
        guess  = discrete(guess)
        target = ext.extract_output(frame, order)
        if do_not_factor_in_padding:
            guess  = utils.remove_padding(frame, guess,  order)
            target = utils.remove_padding(frame, target, order)
            frame  = utils.remove_padding(frame)
        if do_not_factor_in_noise:
            guess  = utils.remove_noise(frame, guess,  order)
            target = utils.remove_noise(frame, target, order)
        target = target.transpose()
        guess  = guess.transpose()
        for r in range(len(target)):
            if target[r].sum() > 0:
                track, length = 0, 0
                for c in range(len(target[r])):
                    track  += (target[r, c] and guess[r, c])
                    length += target[r, c]
                n_correct += ((percent * length) <= track)
                n_tracks  += 1
    return n_correct / n_tracks


def threshold_metrics(
        frame  : pd.DataFrame,
        guess  : np.ndarray,
        thresh : float,
        order  : List[str],
        ) -> np.ndarray:
    """ Return metrics corresponding to the threshold matrix.
        Returns a np.ndarray consisting of four floats.
        float 0: The probability that a hit was assigned to a correct track
            by the threshold matrix.
        float 1: The probability that a hit was assigned to an incorrect track
            by the threshold matrix.
        float 2: The probability that a hit was assigned to more than 1 track
            by the threshold matrix.
        float 3: The probability that a hit was assigned to no track by the
            threshold matrix.
    """
    # Remove the padding column, if necessary.
    out    = ext.extract_output(frame, order)
    target = utils.remove_padding(frame, out, order)
    guess  = utils.remove_padding(frame, guess, order)
    n_hits = number_of_hits(frame)
    matrix = threshold(guess, thresh)
    stack  = np.dstack((target, matrix)).transpose((0, 2, 1))
    rights = np.sum([pair[1, np.argmax(pair[0])] == 1 for pair in stack])
    wrongs = np.sum((stack[:, 0] - stack[:, 1] < 0).any(axis=1))
    multi  = np.sum(np.sum(matrix, axis=1) > 1)  # Hits assigned to multiple.
    no_tks = np.sum(np.sum(matrix, axis=1) < 1)  # Hits unassigned to any.
    return np.array([rights, wrongs, multi, no_tks]) / n_hits


def accuracy_vs_tracks(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : List[np.ndarray],
        order   : List[str],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Return (x, y), where y is the accuracy and x is the number of tracks.
    :param frames:
        A set of frames
    :param guesses:
        A set of guess matrices.
    :param order:
        How to order (phi, r, z)
    :return: A tuple consisting of two numpy arrays that can be used to plot
        this relationship.
    """
    if isinstance(frames, pd.DataFrame):
        groups = utils.list_of_groups(frames, "event_id")
        return accuracy_vs_tracks(groups, guesses, order)
    n_tracks = [number_of_tracks(frame) for frame in frames]
    accuracy = [percent_of_hits_assigned_correctly(frames[i], guesses[i], order)
                for i in range(len(guesses))]
    return np.array(n_tracks), np.array(accuracy)


def accuracy_vs_thresholds(
        frames   : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses  : List[np.ndarray],
        order    : List[str],
        threshes : List[float],
        mode     : str = "correct",  # ("correct", "incorrect", "many", "none")
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Return (x, y) such that y is the accuracy and x is the thresholds.
    :param frames:
        A set of frames
    :param guesses:
        A set of guesses.
    :param order:
        How to order input (phi, r, z)
    :param threshes:
        A list of floats representing the thresholds necessary for a hit to be
        classified as correct.
    :param mode:
        "correct": Probability that a hit was assigned to the correct track
            with a probability of at least threshold.
        "incorrect": Probability that a hit was assigned to at least one
            incorrect track with a probability of at least threshold.
        "many": Probability that a hit was assigned to multiple tracks,
            each with a probability of at least threshold.
        "none": Probability that a hit had no probability equal to or greater
            than threshold in any of the tracks.
    """
    mode  = mode.lower()
    modes = ("correct", "incorrect", "many", "none")
    if mode not in modes:
        print("Error: the 'variation' variable was not found in function:")
        return np.ndarray([]), np.ndarray([])
    if isinstance(frames, pd.DataFrame):
        groups = utils.list_of_groups(frames, "event_id")
        return accuracy_vs_thresholds(groups, guesses, order, threshes, mode)
    index = modes.index(mode)
    accuracy = [[threshold_metrics(frames[i], guesses[i], thresh, order)
                 for i in range(len(frames))] for thresh in threshes]
    return np.array(threshes), np.array(accuracy).transpose()[index]


def accuracy_vs_bend(
        frames  : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses : List[np.ndarray],
        order   : List[str],
        bends   : List[float],
        ) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(frames, pd.DataFrame):
        groups = utils.list_of_groups(frames, "event_id")
        return accuracy_vs_bend(groups, guesses, order, bends)
    bends = sorted(bends)
    accuracies = [[] for _ in range(len(bends) + 1)]
    for f, frame in enumerate(frames):
        tracks = utils.list_of_groups(frame, "cluster_id")
        guess  = discrete(guesses[f])
        matrix = ext.extract_output(frame, order)
        common = np.logical_and(guess, matrix).sum(axis=0)
        track_lengths = matrix.sum(axis=0)
        for t, track in enumerate(tracks):
            if track["noise"].any() or track["padding"].any():
                continue
            low_phi   = track[track["r"] == track["r"].min()]["phi"].min()
            high_phi  = track[track["r"] == track["r"].max()]["phi"].min()
            phi_delta = change_in_phi(low_phi, high_phi)
            low_r     = track["r"].min()
            high_r    = track["r"].max()
            r_delta   = np.abs(high_r - low_r)
            bend      = ((phi_delta / r_delta) if r_delta != 0 else 0)
            bend      = np.round(bend * 1000000, 0).astype(int)
            accuracy  = common[t] / track_lengths[t]
            index     = np.searchsorted(bends, bend)
            accuracies[index].append(accuracy)
    return np.array(bends), np.array(accuracies)


def change_in_phi(
        phi_1 : float,
        phi_2 : float,
        ) -> float:
    """ Calculate the change in phi between phi_1 and phi_2.
    :param phi_1: An angle in radians.
    :param phi_2: An angle in radians.
    :return: The shortest angle in radians between phi_1 and phi_2.
    """
    angle = np.abs(phi_1 % (2 * np.pi) - phi_2 % (2 * np.pi))
    return angle if (angle < np.pi) else (2 * np.pi - angle)


def phi_is_between(
        phi_low  : float,
        phi_high : float,
        phi_mid  : float,
        ) -> bool:
    """ Return True if phi_mid is between phi_low and phi_high.
    :param phi_low: An angle in radians.
    :param phi_high: An angle in radians.
    :param phi_mid: An angle in radians.
    :return: True if phi_mid lies on the shortest arc between phi_low and
        phi_high. False otherwise.
    """
    low_to_mid  = change_in_phi(phi_low, phi_mid)
    mid_to_high = change_in_phi(phi_mid, phi_high)
    low_to_high = change_in_phi(phi_low, phi_high)
    return math.isclose(low_to_mid + mid_to_high, low_to_high)


def tracks_crossed(
        track_1: pd.DataFrame,
        track_2: pd.DataFrame,
        ) -> bool:
    """ Return True if these two tracks cross each other.
    :param track_1: A pd.DataFrame representing a track.
    :param track_2: A pd.DataFrame representing a track.
    :return: True if these two tracks cross each other. False otherwise.
    """
    if len(track_1) <= 1 or len(track_2) <= 1:
        return False

    # Grab the unique radiuses that are used by each tracks' hits.
    track_1_r = np.sort(track_1["r"].unique())
    track_2_r = np.sort(track_2["r"].unique())

    # Find a starting points for the max and min r values.
    max_r = np.min([track_1_r.max(), track_2_r.max()])
    min_r = np.max([track_1_r.min(), track_2_r.min()])

    # Define the minimum and maximum radiuses to compare phi values on.
    # Accounts for case when a track is not full.
    low_r_1  = track_1_r[min([np.searchsorted(track_1_r, min_r),
                              len(track_1_r) - 1])]
    low_r_2  = track_2_r[min([np.searchsorted(track_2_r, min_r),
                              len(track_2_r) - 1])]
    high_r_1 = track_1_r[min([np.searchsorted(track_1_r, max_r),
                              len(track_1_r) - 1])]
    high_r_2 = track_2_r[min([np.searchsorted(track_2_r, max_r),
                              len(track_2_r) - 1])]

    # Find the phis used for comparing if a track crosses another track.
    low_phi_1  = track_1[track_1["r"] ==  low_r_1]["phi"].min() % (2 * np.pi)
    low_phi_2  = track_2[track_2["r"] ==  low_r_2]["phi"].min() % (2 * np.pi)
    high_phi_1 = track_1[track_1["r"] == high_r_1]["phi"].max() % (2 * np.pi)
    high_phi_2 = track_2[track_2["r"] == high_r_2]["phi"].max() % (2 * np.pi)
    answer     = (low_phi_1  <= low_phi_2) != (high_phi_1 <= high_phi_2)

    # Adjust for lines that cross the 0 radians line.
    answer *= not ((2 * low_phi_1  < np.pi) and (2 * high_phi_1 > 3 * np.pi))
    answer *= not ((2 * high_phi_1 < np.pi) and (2 * low_phi_1  > 3 * np.pi))
    answer *= not ((2 * low_phi_2  < np.pi) and (2 * high_phi_2 > 3 * np.pi))
    answer *= not ((2 * high_phi_2 < np.pi) and (2 * low_phi_2  > 3 * np.pi))

    return answer


def accuracy_vs_momentum(
        frames    : Union[pd.DataFrame, List[pd.DataFrame]],
        guesses   : List[np.ndarray],
        order     : List[str],
        momentums : List[float],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ Return the accuracy vs momentum.

    :param frames:
        A set of frames.
    :param guesses:
        A set of matrices.
    :param order:
        How to order the input (phi, r, z)
    :param momentums:
        A set of momentum steps to bin the data in.
    """
    if isinstance(frames, pd.DataFrame):
        groups = utils.list_of_groups(frames, "event_id")
        return accuracy_vs_momentum(groups, guesses, order, momentums)
    momentums = sorted(momentums)
    accuracies = [[] for _ in range(len(momentums) + 1)]
    for f, frame in enumerate(frames):
        tracks = utils.list_of_groups(frame, "cluster_id")
        guess  = discrete(guesses[f])
        matrix = ext.extract_output(frame, order)
        common = np.logical_and(guess, matrix).sum(axis=0)
        track_lengths = matrix.sum(axis=0)
        for t, track in enumerate(tracks):
            if track["noise"].any() or track["padding"].any():
                continue
            momentum  = track["momentum"].min()
            accuracy  = common[t] / track_lengths[t]
            index     = np.searchsorted(momentums, momentum)
            accuracies[index].append(accuracy)
    return np.array(momentums), np.array(accuracies)


def closeness_of_tracks(
        track_1: pd.DataFrame,
        track_2: pd.DataFrame,
        ) -> Tuple[float, float, float]:
    """
    Typical way to measure closeness of track:
    Look at DeltaPhi, DeltaEta and DeltaR of the momentum at the origin.
    eta    = -ln[tan(theta/2)]
    theta  = atan2(pT, pz)
    pT     = sqrt(px^2 + py^2)
    DeltaR = sqrt(DeltaPhi^2 + DeltaEta^2)
    """
    # TODO
    h1 = track_1.nsmallest(1, "r")  # Get first layer hit.
    h2 = track_2.nsmallest(1, "r")  # Get first layer hit.
    
    p1, r1, z1 = h1["phi"], h1["r"], h1["z"]
    p2, r2, z2 = h2["phi"], h2["r"], h2["z"]
    
    x1, y1 = r1 * np.cos(p1), r1 * np.sin(p1)
    x2, y2 = r2 * np.cos(p2), r2 * np.sin(p2)
    
    th1 = np.arctan2(np.sqrt(x1**2 + y1**2), z1)
    th2 = np.arctan2(np.sqrt(x2**2 + y2**2), z2)
    
    eta1 = 0 - np.log(np.tan(th1 / 2))
    eta2 = 0 - np.log(np.tan(th2 / 2))
    
    delta_phi = np.sqrt(p1**2 + p2**2)
    delta_eta = np.sqrt(eta1**2 + eta2**2)
    delta_r   = np.sqrt(delta_phi**2 + delta_eta**2)
    
    # Unsure what to return here.
    return delta_phi, delta_eta, delta_r
