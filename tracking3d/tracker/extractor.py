""" extractor.py/extractor
"""

import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Union, Any
from . import utils, metrics


def extract_input(
        frame   : Union[pd.DataFrame, List[pd.DataFrame]],
        order   : List[str],
        nan_to  : Any = 0,
        ) -> np.ndarray:
    """ Extract a model input array from a frame.
    :param frame:
        A pd.DataFrame to extract from. It can have multiple event_ids or can
        be a list of pd.DataFrames.
    :param order:
        A permutation of ["phi", "r", "z"] that will be used to sort the input.
    :param nan_to:
        What value should NaN values be transformed into.
    :return:
        If there is a single event within the frame, return a 2D matrix
        consisting of input data. Input data is an array of [phi, r, z]
        position information. Thus, it has shape (None, 3).
        If there are multiple events or a list of frames was passed into
        "frame", return a cube of data such that each depth consists of a
        2D matrix of input data corresponding to each event.
    """
    if isinstance(frame, pd.DataFrame):
        if len(pd.unique(frame["event_id"])) > 1:
            groups = utils.list_of_groups(frame, "event_id")
            return extract_input(groups, order, nan_to)
        else:
            return frame.sort_values(order).fillna(nan_to)[order].get_values()
    else:
        return np.array([extract_input(e, order, nan_to) for e in frame])


def extract_output(
        frame       : Union[pd.DataFrame, List[pd.DataFrame]],
        order       : List[str],
        column      : str = "cluster_id",
        categorical : bool = True,
        ) -> np.ndarray:
    """ Extract a model output array from a frame.
    :param frame:
        A pd.DataFrame to extract from. It can have multiple event_ids or can
        be a list of pd.DataFrames.
    :param order:
        A permutation of ["phi", "r", "z"] that will be used to sort the input.
    :param column:
        The name of the column to extract the output from.
    :param categorical:
        True if the output should be a categorical probability matrix.
        False if the output should be a list of categories.
    :return:
        If "categorical" is True:
        If there is a single event within the frame, return a 2D matrix
        consisting of a probability target matrix.
        Thus, it has shape (None, *number of categories in cluster_id*).
        If there are multiple events or a list of frames was passed into
        "frame", return a cube of data such that each depth consists of a
        2D matrix of input data corresponding to each event.
        If "categorical" is False:
        Return a list of such that the i'th hit belongs to the i'th category
        in the list.
        """
    if isinstance(frame, pd.DataFrame):
        if len(pd.unique(frame["event_id"])) > 1:
            groups = utils.list_of_groups(frame, "event_id")
            return extract_output(groups, order, column, categorical)
        sort = frame.sort_values(order)[column].get_values()
        return utils.to_categorical(sort) if categorical else sort
    else:
        return np.array([extract_output(e, order, column, categorical)
                        for e in frame])


def input_output_generator(
        events  : List[pd.DataFrame],
        batch   : int,
        order   : List[str],
        shuffle : bool = True,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ A input-output generator for a model.
    :param events:
        A list of pd.DataFrames to get input-output from.
    :param batch:
        How many instances of input-output to output per yield.
    :param order:
        A permutation of ["phi", "r", "z"]
    :param shuffle:
        True if the function should shuffle the list of events after each
        epoch. False if no shuffling should occur.
    :return:
        Yields a tuple of input-output data.
    """
    while True:
        if shuffle:
            random.shuffle(events)
        for i in range(0, len(events), batch):
            b_idx  = i + batch
            sample = events[i:b_idx] if b_idx < len(events) else events[i:]
            yield (np.array([extract_input(event,  order) for event in sample]),
                   np.array([extract_output(event, order) for event in sample]))


def prepare_frame(
        frame    : pd.DataFrame,
        n_rows   : int = -1,
        n_tracks : int = -1,
        n_noise  : int = 0,
        ) -> pd.DataFrame:
    """ Prepare a data set.
    :param frame:
        A pd.DataFrame
    :param n_rows:
        The number of rows that the frame should have after including padding
        and noise.
    :param n_tracks:
        The number of tracks that the frame.
    :param n_noise:
        The number of noise hits to add per event.
    :return:
        The prepared pd.DataFrame such that each event has padding and noise.
    """
    events   = utils.list_of_groups(frame, group="event_id")
    cleans   = []  # The prepared events go in here.
    if n_tracks < 0:
        n_tracks = max([metrics.number_of_tracks(event) for event in events])
    if n_rows < 0:
        n_rows = n_noise + max([metrics.number_of_hits(event)
                            for event in events])
    for event_id, event in enumerate(events):
        cluster_ids = event["cluster_id"].unique()[:n_tracks]
        event = event[event["cluster_id"].isin(cluster_ids)]
        event = event[:n_rows - n_noise]
        # Map track ids to indices within a probability matrix.
        idx    = event.groupby("cluster_id")["r"].transform(min) == event["r"]
        event  = event.assign(phi=event["phi"] % (2 * np.pi))
        sort   = event[idx].sort_values(["phi", "r", "z"])
        lows   = pd.unique(sort["cluster_id"])
        id2idx = dict((id_, i) for i, id_ in enumerate(lows))
        clean  = event.assign(
            event_id   = event_id,
            cluster_id = event["cluster_id"].map(id2idx),
            noise      = False,
            padding    = False,
        )

        n_padding = n_rows - len(clean) - n_noise
        cleans.append(make_noise(clean, n_tracks, event_id, n_noise))
        cleans.append(make_padding(n_tracks + 1, event_id, n_padding))
        cleans.append(clean)
    # noinspection PyTypeChecker
    return pd.concat(cleans)


def make_noise(
        frame      : pd.DataFrame,
        cluster_id : int,
        event_id   : int,
        n_noise    : int,
        ) -> pd.DataFrame:
    """ Create noise.
    :param frame:
        A frame to base the noise on.
    :param cluster_id:
        The cluster_id that the noise should have.
    :param event_id:
        The event_id that the noise should have.
    :param n_noise:
        The number of noise hits to generate
    :return:
        A pd.DataFrame consisting of noisy hits.
    """
    min_z = frame["z"].min()
    max_z = frame["z"].max()
    columns = ["event_id", "cluster_id", "r", "phi", "z", "noise" , "padding"]
    data = np.stack([
        np.full(n_noise, event_id),
        np.full(n_noise, cluster_id),
        np.random.choice(frame["r"].unique(), n_noise),
        np.random.uniform(0, 2 * np.pi, n_noise),
        np.random.uniform(min_z, max_z, n_noise),
        np.full(n_noise, True),
        np.full(n_noise, False),
    ], axis=1)
    return pd.DataFrame(data=data, columns=columns)


def make_padding(
        cluster_id : int,
        event_id   : int,
        n_padding  : int,
        ) -> pd.DataFrame:
    """ Create padding.
    :param cluster_id:
        The cluster_id that each padding should have.
    :param event_id:
        The event_id that each padding should have.
    :param n_padding:
        The number of padding rows to generate.
    :return:
        A pd.DataFrame consisting of the padding rows.
    """
    columns = ["event_id", "cluster_id", "noise", "padding"]
    data = np.stack([
        np.full(n_padding, event_id),
        np.full(n_padding, cluster_id),
        np.full(n_padding, False),
        np.full(n_padding, True),
    ], axis=1)
    return pd.DataFrame(data=data, columns=columns)
