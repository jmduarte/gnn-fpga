""" tracker/utils.py

Author: Daniel Zurawski
Organization: Fermilab
Grammar: Python 3.6.1
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional
from . import extractor


def list_of_groups(
        frame : pd.DataFrame,
        group : str = "event_id",
        ) -> List[pd.DataFrame]:
    """ Return a list of pd.DataFrame groups.
    :param frame:
        A pd.DataFrame with a column name equal to "group".
    :param group:
        A column name to group the "frame" by.
    :return:
        A list of pd.DataFrames such that each frame is a group of rows
        with the same value in the "group" column.
    """
    return [event for (_, event) in frame.groupby(group)]


def is_prepared(
        frame : pd.DataFrame,
        ) -> bool:
    """ Return True if the "frame" has been prepared.
    :param frame:
        A pd.DataFrame.
    :return:
        True if the "frame" is the output of the extractor.prepare_frame()
        function. A prepared frame differs from an unprepared frame in
        that the prepared frame has boolean columns, "padding" and "noise".
    """
    columns = {"event_id", "cluster_id", "r", "phi", "z", "padding", "noise"}
    return columns.issubset(frame.columns)


def remove_padding(
        frame  : pd.DataFrame,
        matrix : Optional[np.ndarray] = None,
        order  : Optional[List[str]] = None,
        ) -> Union[pd.DataFrame, np.ndarray]:
    """ Remove the padding from the "frame" or "matrix".
    :param frame:
        A pd.DataFrame.
    :param matrix:
        A probability matrix.
    :param order:
        An ordering (permutation of ["phi", "r", "z"])
    :return:
        If matrix is None, then return the frame with its padding rows
        removed. If matrix was not None and an ordering was specified,
        return the matrix with its padding removed.
    """
    if matrix is None or order is None:
        return frame[frame["padding"] == 0].copy()
    else:
        return matrix[frame.sort_values(order)["padding"] == 0].copy()


def add_padding(
        frame    : pd.DataFrame,
        n_rows   : int,
        n_tracks : int,
        ) -> pd.DataFrame:
    """ Add padding to the "frame".
    :param frame:
        A pd.DataFrame to add padding to.
    :param n_rows:
        The number of rows that the frame should end up having after padding.
    :param n_tracks:
        The number of tracks that the frame has. Noise and padding do not
        count as tracks for the intent of this parameter.
    :return:
        A pd.DataFrame with padding added such that it has "n_rows" rows
        and its noise and padding rows had their cluster ids properly adjusted.
    """
    frame = remove_padding(frame)
    # noinspection PyUnresolvedReferences
    frame.loc[frame["noise"] == 1, "cluster_id"] = n_tracks
    events = list_of_groups(frame, "event_id")
    for i in range(len(events)):
        n_padding = n_rows - len(events[i])
        event_id  = events[i].iloc[0]["event_id"]
        padding   = extractor.make_padding(n_tracks + 1, event_id, n_padding)
        events[i] = pd.concat([events[i], padding])
    # noinspection PyTypeChecker
    return pd.concat(events)


def remove_noise(
        frame  : pd.DataFrame,
        matrix : Optional[np.ndarray] = None,
        order  : Optional[List[str]] = None,
        ) -> Union[pd.DataFrame, np.ndarray]:
    """ Remove noise from the "frame".
    :param frame:
        A pd.DataFrame to remove noise from or to use as reference if you want
        to remove padding from the matrix.
    :param matrix:
        A matrix to remove noise from.
    :param order:
        An ordering (permutation of ["phi", "r", "z"])
    :return:
        If matrix is None, then return the frame with its noise rows
        removed. If matrix was not None and an ordering was specified,
        return the matrix with its noise removed.
    """
    if matrix is None or order is None:
        return frame[frame["noise"] != True].copy()
    else:
        return matrix[frame.sort_values(order)["noise"] != True].copy()


def to_categorical(
        categories : np.ndarray,
        n_columns  : int = -1,
        ) -> np.ndarray:
    """  Change a sequence of track categories to a 1-hot probability matrix.
    :param categories:
        A sequence of track categories.
    :param n_columns:
        The number of columns (potential categories) that this probability
        matrix should have.
    :return:
        A 1-hot probability matrix.
    """
    categories = [int(category) for category in categories]
    n_columns = max(categories) + 1 if n_columns < 0 else n_columns
    matrix    = np.zeros((len(categories), n_columns))
    for i, category in enumerate(categories):
        if category < n_columns:
            matrix[i, category] = 1
    return matrix


def from_categorical(
        matrix : np.ndarray,
        ) -> np.ndarray:
    """ Change a 1-hot probability matrix to a sequence of track categories.
    :param matrix:
        A probability matrix.
    :return:
        A list of categories that each row is most likely to be a member of.
    """
    return np.argmax(matrix, axis=1)
