import numpy as np
import pandas as pd
import IPython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
from . import extractor as ext, utils, metrics


def display_matrices(
        data : np.ndarray,
        target : np.ndarray,
        decimal : int = 2,
        order   : Optional[List[str]] = None,
        noise : bool = True,
        padding : bool = True,
        ) -> None:
    """ Display the data matrix and the target matrix side-by-side.
    :param data: The data input matrix.
    :param target: The target output matrix.
    :param decimal: How many decimals to round the matrices to.
    :param order: How to order the data matrix (phi, r, z).
    :param noise: Whether or not there is noise in the data.
    :param padding: Whether or not there is padding in the data.
    """
    table = pd.DataFrame(data, columns=order)
    if target.shape[1] > 1:
        column  = [chr(65+i) for i in range(target.shape[1] - 2)]
        column.append("noise" if noise else chr(65 + target.shape[1] - 2))
        column.append("padding" if padding else chr(65 + target.shape[1] - 1))
    else:
        column = [chr(65)]
    out_table = pd.DataFrame(data=target, columns=column).replace(0, "")
    table = pd.concat([table, out_table], axis=1)
    with pd.option_context('display.max_columns', None, "display.max_rows", None):
        IPython.display.display(table)

def display(
        frame   : pd.DataFrame,
        order   : List[str],
        guess   : Optional[np.ndarray] = None,
        mode    : str = "default",
        decimal : int = 2,
        ) -> None:
    """ Display the frame or guess through IPython.
    :param frame:
        A pd.DataFrame
    :param order:
        A permutation of ["phi", "r", "z"]
    :param guess:
        A prediction probability matrix.
    :param mode:
        One of ["default", "pairs"]
        If "pairs", then the answer is displayed in the same cell as the
        "guess" prediction. Format: "`ANSWER`[PREDICTION]"
    :param decimal:
        How many decimals places to round the guesses to.
    :return:
        None.
    """
    table  = pd.DataFrame(ext.extract_input(frame, order), columns=order)
    target = ext.extract_output(frame, order).round(0)
    if target.shape[1] > 1:
        column  = [chr(65+i) for i in range(target.shape[1] - 2)]
        noise   = frame["noise"].any()
        padding = frame["padding"].any()
        column.append("noise" if noise else chr(65 + target.shape[1] - 2))
        column.append("padding" if padding else chr(65 + target.shape[1] - 1))
    else:
        column = [chr(65)]
    if mode == "guess":
        out_table = pd.DataFrame(data=guess, columns=column).replace(0, "")
        table = pd.concat([table, out_table], axis=1)
    elif mode == "discrete pairs":
        guess = metrics.discrete(guess).round(0)
        data  = []
        for x in range(len(guess)):
            row = []
            for y in range(len(guess[x])):
                if target[x, y] == 0 and guess[x, y] == 0:
                    row.append("")
                else:
                    t, g = int(target[x, y]), int(guess[x, y])
                    row.append("`{0}`[{1}]".format(t, g))
            data.append(row)
        out_table = pd.DataFrame(data=data, columns=column)
        table = pd.concat([table, out_table], axis=1)
    elif mode == "pairs" and guess is not None:
        guess = guess.round(decimal)
        data   = []
        for x in range(len(guess)):
            row = []
            for y in range(len(guess[x])):
                if target[x, y] == 0 and guess[x, y] == 0:
                    row.append("")
                else:
                    t, g = int(target[x, y]), np.round(guess[x, y], 2)
                    row.append("`{0}`[{1}]".format(t, g))
            data.append(row)
        out_table = pd.DataFrame(data=data, columns=column)
        table = pd.concat([table, out_table], axis=1)
    else:
        out_table = pd.DataFrame(data=target, columns=column).replace(0, "")
        table = pd.concat([table, out_table], axis=1)
    with pd.option_context('display.max_columns', 0):
        IPython.display.display(table)


def boxplot(
        data   : List[np.ndarray],
        title  : str  = "Box Plot",
        xlabel : str  = "X",
        ylabel : str  = "Y",
        fliers : bool = False,
        xticks : Optional[List] = None,
        ) -> None:
    """ Create a box plot diagram from the provided data.
    :param data:
        A list of 1D arrays such that each array contains the floating point
        data for one of the box plots.
    :param title:
        The title of the box plot.
    :param xlabel:
        The x axis label of the box plot.
    :param ylabel:
        The y axis label of the box plot.
    :param fliers:
        True if outlier points should be displayed.
    :param xticks:
        The x axis ticks that label each box plot.
    :return:
        None
    """
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.boxplot(data, showfliers=fliers)
    if xticks is not None:
        plt.xticks([i for i in range(1, len(data) + 1)], xticks)


class Plot2D:
    """ A plot of the data. """
    def __init__(
            self,
            frame : pd.DataFrame,
            order : List[str],
            guess : Optional[np.ndarray] = None,
            ) -> None:
        """ Initialize a 2D plot.
        :param frame:
            The frame containing a single event's data.
        :param order:
            The ordering of the frame.
        :param guess:
            The prediction assignment for the hits to a track. If this is
            None, then the hits will be assigned to tracks based on their
            correct assignment.
        :return:
            None
        """
        guess    = ext.extract_output(frame, order) if guess is None else guess
        guess_id = utils.from_categorical(guess)
        frame    = frame.sort_values(order)
        frame    = frame.assign(guess_id=guess_id, index=frame.index)

        self.frame = utils.remove_padding(frame)
        self.order = order
        self.fig   = plt.figure()
        self.leg   = None
        self.ax    = plt.subplot(111)

    def plot(
            self,
            mode  : str,
            title : str = "",
            ) -> None:
        """ Plot this plot.
        :param mode:
            The axes to plot. One of {"xy", "xz", "yz", "zr"}.
            "xy" projects the data onto the XY plane.
            "xz" projects the data onto the XZ plane.
            "yz" projects the data onto the YZ plane.
            "zr" projects the data onto the ZR plane, where r = sqrt(x*x+y*y)
        :param title:
            The title of the plot.
        :return:
            None
        """
        mode   = mode.lower()
        tracks = utils.list_of_groups(self.frame, group="guess_id")
        for i, track in enumerate(tracks):
            guess_id = track.iloc[0]["guess_id"]
            label   = chr(65 + int(guess_id))
            extract = ext.extract_input(track, self.order)
            values  = self.get_values(extract, mode)
            self.ax.scatter(
                    x=values[0, :, 0],
                    y=values[0, :, 1],
                    label=label,
                    picker=True,
                    s=100,
                    linewidth=1,
                    edgecolor='black')
            for t in range(len(track)):
                self.ax.text(values[0, t, 0], values[0, t, 1],
                             label, size=8, zorder=10, color="white",
                             horizontalalignment="center",
                             verticalalignment="center")
        self.ax.set_title(title)
        if mode == "xy":
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            for r in pd.unique(self.frame["r"]):
                self.ax.add_artist(
                        plt.Circle((0, 0), r, color='black', fill=False,
                                   linestyle='-', alpha=0.1))
        elif mode == "yz":
            self.ax.set_xlabel("Y")
            self.ax.set_ylabel("Z")
        elif mode == "xz":
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Z")
        elif mode == "zr":
            self.ax.set_xlabel("Z")
            self.ax.set_ylabel("R")
            min_z = self.frame["z"].min()
            max_z = self.frame["z"].max()
            for r in pd.unique(self.frame["r"]):
                self.ax.plot([min_z, max_z], [r, r], alpha=0.1, color="black")
        self.leg = self.ax.legend(loc='upper right', fancybox=True)
        plt.show()

    def get_values(
            self,
            values : np.ndarray,
            mode : str,
            ) -> np.ndarray:
        """ Retrieve the appropriate x and y coordinates for each point.
        :param values:
            An 2D array of shape (None, 3) such that each row contains
            phi, r and z values.
        :param mode:
            The way in which values should be extracted. For this parameter,
            just put in the mode that was specified upon this object's
            construction.
        :return:
            A 2D array of shape (None, 2) such that the first column
            contains all the x values and the second column contains all the
            y values necessary for plotting.
        """
        mode = mode.lower()
        ps   = values[:, self.order.index("phi")]
        zs   = values[:, self.order.index("z")]
        rs   = values[:, self.order.index("r")]
        xs   = np.cos(ps) * rs
        ys   = np.sin(ps) * rs
        if mode == "xy":
            return np.dstack((xs, ys))
        elif mode == "xz":
            return np.dstack((xs, zs))
        elif mode == "yz":
            return np.dstack((ys, zs))
        else:
            return np.dstack((zs, np.sqrt(xs ** 2 + ys ** 2)))


class Plot3D:
    """ A plot of the data. """
    def __init__(
            self,
            frame : pd.DataFrame,
            order : List[str],
            guess : Optional[np.ndarray] = None,
            ) -> None:
        """ Initialize a 3D plot.
        :param frame:
            The frame containing a single event's data.
        :param order:
            The ordering of the frame.
        :param guess:
            The prediction assignment for the hits to a track. If this is
            None, then the hits will be assigned to tracks based on their
            correct assignment.
        :return:
            None
        """
        guess    = ext.extract_output(frame, order) if guess is None else guess
        guess_id = utils.from_categorical(guess)
        frame    = frame.sort_values(order)
        frame    = frame.assign(guess_id=guess_id, index=frame.index)

        self.frame = utils.remove_padding(frame)
        self.order = order
        self.fig   = plt.figure()
        self.leg   = None
        self.ax    = Axes3D(self.fig)

    def plot(
            self,
            title    : str = "",
            x_limits : Tuple[int, int] = (-1000, 1000),
            y_limits : Tuple[int, int] = (-1000, 1000),
            z_limits : Tuple[int, int] = (-200,   200),
            ) -> None:
        """ Plot this plot.
        :param title:
            The title of the plot.
        :param x_limits:
            The maximum x boundaries
        :param y_limits:
            The maximum y boundaries
        :param z_limits:
            The maximum z boundaries
        :return:
            None
        """
        tracks = utils.list_of_groups(self.frame, group="guess_id")
        for i, track in enumerate(tracks):
            guess_id = int(track.iloc[0]["guess_id"])
            label  = chr(65 + guess_id)
            values = self.cartesian(ext.extract_input(track, self.order))
            self.ax.scatter3D(
                    xs=values[0, :, 0],
                    ys=values[0, :, 2],
                    zs=values[0, :, 1],
                    label=label,
                    picker=True,
                    s=100,
                    linewidth=1,
                    edgecolor='black',
                    depthshade=True)
            for t in range(len(track)):
                self.ax.text(values[0, t, 0], values[0, t, 2], values[0, t, 1],
                             label, size=8, zorder=10, color="white",
                             horizontalalignment="center",
                             verticalalignment="center")
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")
        self.ax.set_xlim3d(x_limits[0], x_limits[1])
        self.ax.set_ylim3d(z_limits[0], z_limits[1])
        self.ax.set_zlim3d(y_limits[0], y_limits[1])
        self.leg = self.ax.legend(loc='upper right', fancybox=True)
        plt.show()

    def cartesian(
            self,
            values : np.ndarray,
            ) -> np.ndarray:
        """  Transform 'phi', 'z', 'r' coordinates to cartesian coordinates.
        :param values:
            An array of shape (None, 3) that contain the array of
            phi, z, r values.
        :return:
            An array of shape (None, 3) such that the first column contains all
            the x values, the second column contains all the y values and the
            third column contains all the z values.
        """
        ps = values[:, self.order.index("phi")]
        zs = values[:, self.order.index("z")]
        rs = values[:, self.order.index("r")]
        xs = np.cos(ps) * rs
        ys = np.sin(ps) * rs
        return np.dstack((xs, ys, zs))
