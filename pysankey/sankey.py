# -*- coding: utf-8 -*-
"""
Produces simple Sankey Diagrams with matplotlib.
@author: Anneya Golob & marcomanz & pierre-sassoulas & jorwoods
                      .-.
                 .--.(   ).--.
      <-.  .-.-.(.->          )_  .--.
       `-`(     )-'             `)    )
         (o  o  )                `)`-'
        (      )                ,)
        ( ()  )                 )
         `---"\    ,    ,    ,/`
               `--' `--' `--'
                |  |   |   |
                |  |   |   |
                '  |   '   |
"""

# Standard Library
import copy
import functools
import operator
from collections import defaultdict, namedtuple

# Third Party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# So that multiple images are retained when exporting to pdf or pgf, important only when using the "gradient" option
matplotlib.rcParams["image.composite_image"] = False


class PySankeyException(Exception):
    pass


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass


# A class that returns itself when copied, necessary as nan != nan
class _Constant:
    def __init__(self, name=None):
        self.name = name

    def __deepcopy__(self, *args):
        # This is important so I can keep looking this up when I copy the widths dict

        return self

    def __repr__(self):
        return self.name or super().__repr__()


_EMPTY = _Constant("_EMPTY")
_SKIP = _Constant("_SKIP")
_INVALIDS = (_EMPTY, _SKIP)


def check_data_matches_labels(labels, data, side):
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)

        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())

        if isinstance(labels, list):
            labels = set(labels)

        if labels != data:
            msg = "\n"

            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"

            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch(
                "{0} labels and data do not match.{1}".format(side, msg)
            )


# make sure labels for _SKIP and _EMPTY aren't in the middle of everything
def pull_invalids(array):
    for invalid in _INVALIDS:
        has_invalid = invalid in array

        if has_invalid:
            index = np.argwhere(array == invalid)
            array = np.delete(array, index)
            array = np.append(array, invalid)

    return array


rgbColor = namedtuple("RGB", ["red", "green", "blue"])


def hex_to_rgb(hex_color):
    """Transform a hex color representation into a ``rgbColor`` named tuple.

    :param hex_color: the hex color string

    :returns: RGB colors
    """
    hex_color = hex_color.lstrip("#")
    # As hex colors may also be of the form #fff
    step = len(hex_color) // 3

    rgb_color = rgbColor(
        *(
            int(hex_color[i : i + step], 16) / 256.0
            for i in range(0, len(hex_color), step)
        )
    )

    return rgb_color


def sankey(
    values,
    weights=None,
    labels=None,
    color_dict=None,
    steps=None,
    aspect=4,
    color="left",
    fontsize=14,
    font_family="serif",
    figure_name=None,
    formats=["png"],
    bar_distance=0.02,
    bar_width=0.02,
    close_plot=False,
    vertical_align=True,
    text_on_boxes=False,
    none_value=_EMPTY,
    skip_value=_SKIP,
    weights_are_relative=False,
    alpha=0.65,
):
    """
    Make Sankey Diagram showing flow from left-->right

    Inputs:
    :param values: scalar of NumPy arrays of object labels for each step or a Pandas DataFrame
    :param weights: scalar of NumPy array of weights for each strip for each step, if only one array is provided the same weights will be used for all other labels
    :param color_dict: dictionary of colors to use for each label {'label':'color'}
    :param labels: scalar of scalars giving the order of the labels in the diagram for each step
    :param aspect: vertical extent of the diagram in units of horizontal extent
    :param color: if "right", each strip in the diagram will be be colored according to its left label, if 'gradient', a gradient from the left label color to the right label color will be created
    :param steps: scalar giving the horizontal extents of each gap, so its length should be that of values minus one
    :param fontsize: the font size to use for the labels
    :param font_family: the font family to use for the labels
    :param figure_name: if given, the figure will be saved under that name
    :param formats: if ``figure_name`` is given a figure will be saved for each format specified here
    :param bar_distance: the distance the label bars should have to each other
    :param bar_width: the horizontal width of the bars
    :param close_plot: if True the plot will be closed
    :param vertical_align: if False, no height correction will be done to align each step vertically
    :param text_on_boxes: if True, labels will be displayed on their boxes
    :param none_value: a value that should be used for empty values, no strip will be painted to or from an empty value
    :param skip_value: a value that represents a skipped step, strips will be drawn that directly go to the next non-skip step
    :param weights_are_relative: if True, weights will be transformed to sum up to one
    :param alpha: the alpha value of the strips

    :returns: None
    """

    if isinstance(values, pd.DataFrame):
        values = [values[i] for i in values.columns]

    weights = weights if weights is not None else [pd.Series([])] * len(values)

    labels = labels if labels is not None else [[]] * len(values)

    steps = steps if steps is not None else np.ones(len(values) - 1)
    assert len(steps) == len(values) - 1

    total_steps = sum(steps)

    # Check weights
    weights[0] = (
        np.ones(len(values[0]))
        if weights[0] is None or weights[0].empty
        else weights[0]
    )
    weights[1:] = [weights[0] if w is None or w.empty else w for w in weights[1:]]

    if weights_are_relative:
        weights = [(w / w.min()) / (w / w.min()).sum() for w in weights]

    plt.figure()
    plt.rc("text", usetex=False)
    plt.rc("font", family=font_family)

    # Create Dataframe

    values = [
        v.reset_index(drop=True) if isinstance(v, pd.Series) else v for v in values
    ]
    value_indices = list(range(len(values)))

    values_dict = {f"values_{i}": v for i, v in enumerate(values)}
    weights_dict = {f"weights_{i}": weights[i] for i, v in enumerate(values)}

    dataframe = pd.DataFrame(
        {**values_dict, **weights_dict}, index=range(len(values_dict["values_0"]))
    )

    dataframe.fillna(_EMPTY, inplace=True)

    if none_value is not None:
        dataframe.replace(none_value, _EMPTY, inplace=True)

    dataframe.replace(skip_value, _SKIP, inplace=True)
    dataframe[f"values_0"].replace(_SKIP, _EMPTY, inplace=True)
    dataframe[f"values_{value_indices[-1]}"].replace(_SKIP, _EMPTY, inplace=True)

    skipping_lines = defaultdict(set)

    waiting = object()

    for __, row in dataframe.iterrows():
        waiting_for_match = waiting

        for i, value in enumerate(row):
            if value is _SKIP:
                if waiting_for_match is waiting:
                    waiting_for_match = (row[i - 1], i - 1, [i])
                else:
                    waiting_for_match[2].append(i)

            elif waiting_for_match is not waiting:
                skipping_lines[waiting_for_match[0:2]].add(
                    (value, i, tuple(waiting_for_match[2]))
                )
                waiting_for_match = waiting
            else:
                waiting_for_match = waiting

    # Identify all labels that appear 'left' or 'right'
    all_labels = list(
        pd.Series(
            np.r_[tuple(dataframe[f"values_{i}"].unique() for i in value_indices)]
        ).unique()
    )

    # Identify left labels

    labels = [
        check_data_matches_labels(l)
        if l
        else pull_invalids(pd.Series(dataframe[f"values_{i}"].unique()).unique())
        for i, l in enumerate(labels)
    ]

    # If no color_dict given, make one

    if color_dict is None:
        color_dict = {}
        palette = "hls"
        colorPalette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, label in enumerate(all_labels):
            color_dict[label] = colorPalette[i % len(colorPalette)]
    else:
        missing = [
            label
            for label in all_labels
            if label not in color_dict.keys() and label not in _INVALIDS
        ]

        if missing:
            msg = (
                "The color_dict parameter is missing values for the following labels : "
            )
            msg += "{}".format(", ".join(map(str, missing)))
            raise ValueError(msg)

    def get_cmap(left_label, right_label):
        colors = [color_dict[label] for label in (left_label, right_label)]
        n = len(colors)

        colors = list(map(hex_to_rgb, colors))
        colors = [[(i * 1 / (n - 1), v, v) for v in colors[i]] for i in range(n)]

        colors = list(zip(*colors))

        c_dict = dict(zip(("red", "green", "blue"), colors))

        cmap = matplotlib.colors.LinearSegmentedColormap(
            f"{left_label} > {right_label}", segmentdata=c_dict, N=256
        )

        return cmap

    # Determine widths of individual strips
    ns = defaultdict(lambda: defaultdict(defaultdict))

    def get_widths(i, r_i, left_label, right_label, *skips):
        conditions = [
            (dataframe[f"values_{i}"] == left_label),
            (dataframe[f"values_{r_i}"] == right_label),
        ]
        conditions[1:1] = [(dataframe[f"values_{s}"] == _SKIP) for s in skips]
        condition = functools.reduce(operator.and_, conditions)

        left = dataframe[condition]
        left_width = left[f"weights_{i}"].sum()

        right = dataframe[condition]
        right_width = right[f"weights_{r_i}"].sum()

        return left_width, right_width

    for i, left_labels in enumerate(labels[:-1]):

        for left_label in left_labels:
            left_dict = defaultdict(lambda: 0)
            right_dict = defaultdict(lambda: 0)

            for right_label, r_i, skips in skipping_lines[left_label, i]:
                left_width, right_width = get_widths(
                    i, r_i, left_label, right_label, *skips
                )

                left_dict[right_label, r_i] += left_width
                left_dict[right_label] += left_width
                right_dict[right_label, r_i] += right_width
                right_dict[right_label] += left_width

            r_i = i + 1

            for right_label in labels[i + 1]:
                left_width, right_width = get_widths(i, r_i, left_label, right_label)
                left_dict[right_label, r_i] += left_width
                right_dict[right_label, r_i] += right_width

            ns[i]["left"][left_label] = left_dict
            ns[i]["right"][left_label] = right_dict

    # Determine positions of left label patches and total widths
    widths = defaultdict(defaultdict)
    top_edge = 0

    for i, labels_ in enumerate(labels):

        for j, label in enumerate(labels_):
            my_d = {}

            my_d["left"] = dataframe[dataframe[f"values_{i}"] == label][
                f"weights_{i}"
            ].sum()

            if j == 0:
                my_d["bottom"] = 0
            else:
                my_d["bottom"] = widths[i][labels_[j - 1]]["top"]

            if my_d["left"] and j:
                my_d["bottom"] += bar_distance * dataframe[f"weights_{i}"].sum()

            my_d["top"] = my_d["bottom"] + my_d["left"]

            if j == 0:
                my_d["height"] = my_d["left"]
            else:
                my_d["height"] = my_d["top"] - widths[i][labels_[j - 1]]["top"]

            widths[i][label] = my_d

            top_edge = max(top_edge, my_d["top"])

    heights = {
        n: sum(v["height"] for k, v in step.items() if k not in _INVALIDS)
        for n, step in widths.items()
    }
    max_height = max(heights.values())

    # Total vertical extent of diagram
    x_max = top_edge / aspect
    x_step = x_max / (len(labels) - 1)

    def get_x(i):
        return (sum(steps[:i]) / total_steps) * x_max

    # Plot strips

    if vertical_align:
        height_corrections = {n: (max_height - h) / 2 for n, h in heights.items()}
    else:
        height_corrections = {n: 0 for n in value_indices}

    def draw_line(i, r_i, right_label, left_label, right_pair=None):
        if any(label in _INVALIDS for label in (left_label, right_label)):
            return

        right_pair = right_pair or i
        label_color = left_label

        if color == "right":
            label_color = right_label

        hc_l = height_corrections[i]
        hc_r = height_corrections[r_i]
        # Create array of y values for each strip, half at left value,
        # half at right, convolve
        ys_d = np.array(
            50 * [widths[i][i][left_label]["bottom"] + hc_l]
            + 50 * [widths[right_pair][r_i][right_label]["bottom"] + hc_r]
        )
        ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
        ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode="valid")
        ys_u = np.array(
            50
            * [
                widths[i][i][left_label]["bottom"]
                + ns[i]["left"][left_label][right_label, r_i]
                + hc_l
            ]
            + 50
            * [
                widths[right_pair][r_i][right_label]["bottom"]
                + ns[i]["right"][left_label][right_label, r_i]
                + hc_r
            ]
        )
        ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")
        ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode="valid")

        # Update bottom edges at each label so next strip starts at the right place
        widths[i][i][left_label]["bottom"] += ns[i]["left"][left_label][
            right_label, r_i
        ]
        widths[right_pair][r_i][right_label]["bottom"] += ns[i]["right"][left_label][
            right_label, r_i
        ]

        x_pos_1 = get_x(i)
        x_pos_2 = get_x(r_i)

        if i != 0:
            x_pos_1 = x_pos_1 + ((bar_width / 2) * x_max)

        if r_i != (len(labels) - 1):
            x_pos_2 = x_pos_2 - ((bar_width / 2) * x_max)

        points = np.linspace(x_pos_1, x_pos_2, len(ys_d))

        if color == "gradient":
            extent = [points[0], points[-1], min(ys_d), max(ys_u)]

            fill_between = plt.fill_between(points, ys_d, ys_u, color="none", lw=0)
            fill_path, = fill_between.get_paths()
            fill_mask = matplotlib.patches.PathPatch(
                fill_path, fc="none", ec="none", lw=0
            )
            plt.gca().add_patch(fill_mask)
            gradient = np.linspace(0, 1, 256)
            fill_gradient = np.vstack((gradient, gradient))

            cmap = get_cmap(left_label, right_label)
            fill_im = plt.imshow(
                fill_gradient, cmap=cmap, extent=extent, aspect="auto", alpha=alpha
            )
            fill_im.set_clip_path(fill_mask)

        else:
            plt.fill_between(
                points, ys_d, ys_u, alpha=alpha, color=color_dict[label_color]
            )

    widths_org = widths

    widths = defaultdict(lambda: copy.deepcopy(widths_org))

    for i, left_labels in enumerate(labels[:-1]):
        left_height = heights[i]
        left_height_correction = (max_height - left_height) / 2

        right_height = heights[i + 1]
        right_height_correction = (max_height - right_height) / 2
        hc_r = right_height_correction

        for left_label in left_labels:
            for right_label, r_i, __ in skipping_lines[left_label, i]:
                draw_line(i, r_i, right_label, left_label, right_pair=r_i - 1)

            for right_label in labels[i + 1]:
                r_i = i + 1

                if (
                    len(
                        dataframe[
                            (dataframe[f"values_{i}"] == left_label)
                            & (dataframe[f"values_{i+1}"] == right_label)
                        ]
                    )
                    > 0
                ):
                    draw_line(i, r_i, right_label, left_label)

    # Draw vertical bars on left and right of each  label's section & print label

    widths = widths_org

    for i, labels_ in enumerate(labels):
        hc = 0

        if vertical_align:
            height_correction = (max_height - heights[i]) / 2
            hc = height_correction

        for label in labels_:
            if label in _INVALIDS or not widths[i][label]["left"]:
                continue

            x_pos_1 = get_x(i) - bar_width / 2 * x_max
            x_pos_2 = get_x(i) + bar_width / 2 * x_max

            if i == 0:
                x_pos_1 = -bar_width * x_max
                x_pos_2 = 0 * x_max
            elif i == (len(labels) - 1):
                x_pos_1 = 1 * x_max
                x_pos_2 = (1 + bar_width) * x_max

            plt.fill_between(
                [x_pos_1, x_pos_2],
                2 * [widths[i][label]["bottom"] + hc],
                2 * [widths[i][label]["bottom"] + widths[i][label]["left"] + hc],
                color=color_dict[label],
                alpha=0.99,
            )
            x_pos = get_x(i) + (bar_width / 2 + 0.03) * x_max * int(not text_on_boxes)

            horizontal_align_ = "center" if text_on_boxes else "left"

            if i == 0:
                x_pos = (-bar_width / 2 - 0.03 * int(not text_on_boxes)) * x_max
                horizontal_align_ = "center" if text_on_boxes else "right"

            elif i == (len(labels) - 1):
                x_pos = (1 + bar_width / 2 + 0.03 * int(not text_on_boxes)) * x_max

            plt.text(
                x_pos,
                hc + widths[i][label]["bottom"] + 0.5 * widths[i][label]["left"],
                label,
                {"ha": horizontal_align_, "va": "center"},
                fontsize=fontsize,
            )

    plt.gca().axis("off")
    plt.gcf().set_size_inches(6, 6)

    if figure_name != None:
        for format_ in formats:
            plt.savefig(
                "{}.{}".format(figure_name, format_), bbox_inches="tight", dpi=150
            )

    if close_plot:
        plt.close()
