# %%
# %matplotlib qt
import numpy as np
import sleap


# from . import MODEL_FILES_DIR
from sleap.nn.inference import Tracker
import matplotlib

matplotlib.use("Agg")
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import h5py
from scipy.signal import savgol_filter
import os, sys
from syn_tracker.processes.Wormstats import wormstats
import tables
import warnings
import tqdm
import seaborn as sns

# from IPython.display import clear_output
from matplotlib.colors import LogNorm
import tensorflow as tf
from syn_tracker.processes.helper import get_events
from matplotlib.patches import Patch

# from syn_tracker import MODEL_FILES_DIR

warnings.filterwarnings("ignore")


# def blockPrint():
#     sys.stdout = open(os.devnull, "w")


# blockPrint()
wormstats_header = wormstats()

# %%


def _process_skeleton(video_file, predictor):
    RAW_DIR = video_file

    # File paths
    RESULTS_DIR = RAW_DIR.replace("RawVideos", "Results")
    AUX_DIR = Path(
        str(Path(RAW_DIR).parent).replace("RawVideos", "AuxiliaryFiles")
    ).joinpath("metadata_source.xlsx")
    meta_df = pd.read_excel(AUX_DIR)
    # Load video using SLEAP
    video = sleap.Video.from_filename(RAW_DIR)

    # # Prepare save folder
    save_folder = Path(RESULTS_DIR).parent.joinpath(Path(RESULTS_DIR).stem)
    save_folder.mkdir(exist_ok=True, parents=True)

    label = predictor.predict(video, make_labels=True)

    label.export(save_folder.joinpath("metadata_skeletons.hdf5"))

    return meta_df, save_folder


# return save_folder.joinpath('metadata_skeletons.h5'


def _open_skeleton_file(filename):

    with h5py.File(filename, "r") as f:

        locations = f["tracks"][:].T

    return locations


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def instance_node_velocities(instance_idx, locations, frame_count, node_count):
    velocity_threshold = 0.01
    fly_node_locations = locations[:, :, :, instance_idx]
    fly_node_velocities = np.zeros((frame_count, node_count))
    fly_node_angular_velocities = np.zeros((frame_count, node_count))

    for n in range(0, node_count):
        fly_node_velocities[:, n] = smooth_diff(fly_node_locations[:, n, :])
        avg_velocity = np.nanmean(fly_node_velocities[:, n])
        # print(avg_velocity)
        if avg_velocity < velocity_threshold:  # Fly is classified as dead
            fly_node_angular_velocities[:, n] = 0
        else:
            for t in range(1, frame_count - 1):
                vec1 = fly_node_locations[t + 1, n, :] - fly_node_locations[t, n, :]
                vec2 = fly_node_locations[t, n, :] - fly_node_locations[t - 1, n, :]
                angle1 = np.arctan2(vec1[1], vec1[0])
                angle2 = np.arctan2(vec2[1], vec2[0])
                angular_velocity_node = angle1 - angle2
                fly_node_angular_velocities[t, n] = angular_velocity_node
            fly_node_angular_velocities[0, n] = np.nan
            fly_node_angular_velocities[frame_count - 1, n] = np.nan

    return fly_node_velocities, fly_node_angular_velocities

    # return fly_node_velocities


def _return_time_series_feat_summ(
    save_folder, locations, meta_df, MM_TO_PX=1 / 115, FPS=19
):

    TIME_STEP = 1 / FPS
    plate_name = str(Path(save_folder).name).split(".")[0].split("_")[:6]
    del plate_name[3]
    data = []
    frame_count, node_count, _, instance_count = locations.shape

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)

    # Loop through each fly_ID
    for fly_ID in range(instance_count):
        velocities, angular_velocity = instance_node_velocities(
            fly_ID, locations, frame_count, node_count
        )
        velocities = velocities * (MM_TO_PX / TIME_STEP)
        angular_velocity = angular_velocity / TIME_STEP
        frames = np.arange(frame_count)

        # Loop through each frame to create rows
        for i in range(frame_count):
            # Create a row with the velocities, fly_ID, and frame_number
            row = (
                list(velocities[i])
                + list(angular_velocity[i])
                + [fly_ID, frames[i]]
                + plate_name
                + meta_df.iloc[int(plate_name[3]) - 1, 2:].tolist()
            )
            data.append(row)
    # Initialize columns and create DataFrame
    columns_time_series = wormstats_header.init_col_time_series
    columns_feat_summ = wormstats_header.init_col_fet_summ
    df_time_series = pd.DataFrame(data, columns=columns_time_series)

    # Add motion_mode, replacing NaNs where necessary
    df_time_series["motion_mode"] = get_events(df_time_series, FPS, MM_TO_PX)
    df_time_series["motion_mode"] = np.where(
        df_time_series["thorax_vel"].isna(), np.nan, df_time_series["motion_mode"]
    )

    # Define columns for velocity and motion mode
    feat_columns = columns_time_series[:10] + ["motion_mode"]

    # Calculate medians for all except 'motion_mode' and mean for 'motion_mode'
    feat_summ = df_time_series[feat_columns[:-1]].apply(
        lambda x: np.nanmedian(np.abs(x))
    ).tolist() + [df_time_series["motion_mode"].mean()]

    # Calculate for motion_mode == 0
    feat_summ += (
        df_time_series[df_time_series["motion_mode"] == 0][feat_columns[:-1]]
        .apply(lambda x: np.nanmedian(np.abs(x)))
        .tolist()
    )

    # Combine with the first row string values
    final_result = feat_summ + df_time_series[columns_time_series[10:]].iloc[0].tolist()

    # Create final DataFrame and drop 'frame_number'
    result_feat_summ = pd.DataFrame(
        [final_result], columns=columns_feat_summ + columns_time_series[10:]
    )
    result_feat_summ.drop(columns="frame_number", inplace=True)

    return df_time_series, result_feat_summ


def main_process(Raw_vids_path, MODEL_FILES_DIR, FPS=19, mm_2_px=1 / 115):
    # Constants
    FPS = FPS
    MM_TO_PX = mm_2_px

    RawVideos_path = Path(Raw_vids_path)
    Results_path = RawVideos_path.parent / "Results"
    Figures = RawVideos_path.parent / "Figures"
    Figures.mkdir(exist_ok=True, parents=True)

    # Model paths
    MODEL_CENTROID = os.path.join(MODEL_FILES_DIR, "240903_172126.centroid.n=160")
    MODEL_INSTANCE = os.path.join(
        MODEL_FILES_DIR, "240903_174724.centered_instance.n=160"
    )

    predictor = sleap.load_model([MODEL_CENTROID, MODEL_INSTANCE], batch_size=2)

    predictor.tracker = Tracker.make_tracker_by_name(tracker="flow")

    # Get all mp4 files in RawVideos_path
    mp4_files = list(RawVideos_path.rglob("*.mp4"))

    # Initialize wormstats header

    # Define the filters for PyTables
    TABLE_FILTERS = tables.Filters(
        complevel=5, complib="zlib", shuffle=True, fletcher32=True
    )

    for file_id in tqdm.tqdm(range(len(mp4_files)), total=len(mp4_files)):
        tf.keras.backend.clear_session()

        # try:

        video_file = str(mp4_files[file_id])
        save_folder = Results_path / Path(video_file).stem
        meta_df, save_folder = _process_skeleton(video_file, predictor)

        # Open skeleton file
        locations = _open_skeleton_file(save_folder / "metadata_skeletons.hdf5")

        # Create and write to metadata_featuresN.h5
        with tables.File(
            save_folder / "metadata_featuresN.hdf5", "w"
        ) as f, tables.File(Results_path / "feature_summary.hdf5", "a") as f_sum:
            tab_time_series = f.create_table(
                "/",
                "timeseries_data",
                wormstats_header.traj_timeseries,
                filters=TABLE_FILTERS,
            )
            if "/features_summary_data" in f_sum:
                tab_feat_summary = f_sum.get_node("/features_summary_data")
            else:
                tab_feat_summary = f_sum.create_table(
                    "/",
                    "features_summary_data",
                    wormstats_header.feat_summary,
                    filters=TABLE_FILTERS,
                )

            time_series, feat_summary = _return_time_series_feat_summ(
                save_folder, locations, meta_df, MM_TO_PX, FPS=FPS
            )

            tab_time_series.append(
                np.array(
                    time_series.to_records(index=False),
                    dtype=wormstats_header.traj_timeseries,
                )
            )
            tab_feat_summary.append(
                np.array(
                    feat_summary.to_records(index=False),
                    dtype=wormstats_header.feat_summary,
                )
            )
            # clear_output(wait=True)

            # # Process and write feature summary data
            # feat_summary['fly_type_concentration (ppm)'] = (
            #     feat_summary['fly_type'] + '_' + feat_summary['concentration (ppm)'].astype(str)

            # )
    # except Exception as e:
    #    continue

    def plot_velocity_boxplots(data, CSN_name, clustermap=True, Figures="figures"):
        # Ensure the figures directory exists
        Figures.mkdir(exist_ok=True, parents=True)

        # Clean up column names

        # Create new columns for fly type and concentration
        data["fly_concentration"] = (
            data["fly_type"] + "_" + data["concentration (ppm)"].astype(str)
        )
        data["fly_concentration_hour"] = (
            data["fly_type"]
            + "_"
            + data["concentration (ppm)"].astype(str)
            + "_"
            + data["hour"].astype(str)
        )

        # Define the body parts for velocity plotting
        velocity_columns = df.columns[:10].tolist()

        # Melt the DataFrame for easier plotting
        melted_df = data.melt(
            id_vars=["fly_concentration", "hour"],
            value_vars=velocity_columns,
            var_name="velocity_type",
            value_name="velocity",
        )

        # Drop NaN values
        melted_df.dropna(inplace=True)

        # Get the list of unique hours, sorted
        hours_list = sorted(melted_df["hour"].unique())

        # Plotting loop for each velocity type
        for velocity_column in velocity_columns:
            # Filter data for the current velocity type
            plot_data = melted_df[melted_df["velocity_type"] == velocity_column]

            # Initialize the matplotlib figure
            plt.figure(figsize=(30, 8))

            # Create the swarmplot first
            ax = sns.swarmplot(
                x="fly_concentration",
                y="velocity",
                hue="hour",
                hue_order=hours_list,
                dodge=True,
                data=plot_data,
                palette="Set1",
            )

            # Overlay the boxplot
            sns.boxplot(
                x="fly_concentration",
                y="velocity",
                hue="hour",
                hue_order=hours_list,
                data=plot_data,
                dodge=True,
                showfliers=False,
                linewidth=1,
                color="black",
                fill=False,
                ax=ax,
                legend=False,
            )

            # Customize the plot with clear titles and labels
            plt.title(
                f"{velocity_column}_{CSN_name}", fontsize=16
            )  # Increase title font size
            plt.xlabel(
                "Fly_type and Concentration (ppm)", fontsize=14
            )  # Increase x-label font size
            ylabel = (
                "Angular velocity (rad/s)"
                if "angular" in velocity_column
                else "Velocity (mm/s)"
            )

            plt.ylabel(ylabel, fontsize=14)

            # Increase the font size of the tick labels
            plt.xticks(fontsize=12)  # Increase x-tick label font size
            plt.yticks(fontsize=12)  # Increase y-tick label font size
            legend = ax.legend(
                title="Hour", fontsize=12, title_fontsize=14
            )  # Increase legend font size
            legend.set_title("Hour")  # Set the title of the legend

            # Save the plot to the specified folder
            plt.savefig(
                os.path.join(Figures, f"{velocity_column}_{CSN_name}_plot.png"),
                bbox_inches="tight",
            )

            # Close the plot to free memory
            plt.close()

        # Calculate mean velocities for the cluster map

    def _plot_clustermap(data, Figures):
        data["fly_CSN_concentration_hour"] = (
            data["fly_type"]
            + "_"
            + data["CSN"].astype(str)
            + "_"
            + data["concentration (ppm)"].astype(str)
            + "_"
            + data["hour"].astype(str)
        )

        velocity_column_all = data.columns[:11].tolist()
        data[velocity_column_all] = (
            data[velocity_column_all] - data[velocity_column_all].mean()
        ) / data[velocity_column_all].std()
        mean_velocities = data.groupby("fly_CSN_concentration_hour")[
            velocity_column_all
        ].mean()
        mean_velocities["motion_mode_fraction"] = data.groupby(
            "fly_CSN_concentration_hour"
        )["motion_mode_fraction"].mean()
        # Drop NaN values
        mean_velocities.dropna(inplace=True)
        label = [i[1] for i in mean_velocities.index.str.split("_")]

        # Convert list to DataFrame
        label = pd.DataFrame(label, columns=["CSN"])

        # Flatten the tuple (if necessary) - removing commas from the labels
        label["CSN"] = label["CSN"].apply(lambda x: x if isinstance(x, str) else x[0])

        # Get unique labels and set up color palette
        unique_labels = np.unique(label["CSN"])
        palette = sns.color_palette("tab10", unique_labels.shape[0])

        # Create lookup table for colors
        lut = dict(zip(unique_labels, palette))

        # Map labels to colors
        row_colors = label["CSN"].map(lut)

        # Set index for row_colors to match mean_velocities index
        row_colors.index = mean_velocities.index

        # Plot clustermap with normalized data and mapped row_colors
        g = sns.clustermap(
            mean_velocities,
            metric="euclidean",
            method="average",
            cmap="jet",
            cbar_kws={"label": "znorm [-]"},
            row_colors=row_colors,
            linewidths=0,
            figsize=(30, 10),
            yticklabels=True,
            cbar_pos=(0, 0.8, 0.03, 0.2),  # Position of the colorbar
        )
        g.ax_heatmap.set_xticklabels(
            g.ax_heatmap.get_xticklabels(), fontsize=10, rotation=90
        )
        g.ax_heatmap.set_yticklabels(
            g.ax_heatmap.get_yticklabels(), fontsize=10, rotation=0
        )
        cbar = g.ax_heatmap.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(14)  # Set the font size of the colorbar label
        cbar.ax.tick_params(labelsize=14)

        handles = [Patch(facecolor=color, label=label) for label, color in lut.items()]

        # Place the legend on the clustermap
        plt.legend(
            handles=handles,
            title="CSN Labels",  # Adjust this to your label
            bbox_to_anchor=(1.5, -2.5),  # Adjust position based on figure
            loc="upper left",  # Position of the legend
            borderaxespad=0.0,
        )
        plt.savefig(os.path.join(Figures, "velocity_clustermap.png"))

    df = pd.read_hdf(Results_path / "feature_summary.hdf5", "/features_summary_data")
    df.columns = df.columns.str.replace(r"\[mms\]", "", regex=True)

    for CSN_file in tqdm.tqdm(df["CSN"].unique()):

        data_csn = df[df["CSN"] == CSN_file]
        plot_velocity_boxplots(
            data_csn, CSN_file, clustermap=True, Figures=Figures.joinpath(f"{CSN_file}")
        )
    try:
        _plot_clustermap(df, Figures)
    except Exception as e:
        raise ValueError("You don't have enough points to plot a clustermap") from e


# %%
