
#%%
%matplotlib qt
import numpy 
import sleap
from sleap.nn.inference import Tracker
from pathlib import Path 
import subprocess
import numpy as np
import matplotlib.pyplot as plt 
import h5py
from scipy.signal import savgol_filter
import pandas as pd
from debug_helper import get_events

# %%
Raw_dir = 'Final_code/Project_1/RawVideos/CTRL - untreated/time 72h/GBJH006_01_08_1169872637_01_2023-03-16_11-14-46-258_001_Off_Centre.mp4'
#Raw_dir = 'Final_code/Project_1/RawVideos/IRAC_9B/time 72h/GBJH006_02_02_1169872705_06_2024-08-15_14-02-24-739_001_Off_Centre.mp4'
Results_dir = Raw_dir.replace('RawVideos','Results')
mm_2_px = (1/115)   #  mm to pixel

time = 1/19    #  seconds 
video = sleap.Video.from_filename(Raw_dir)
model_1_centroid = 'Final_code/model/final_model/240903_172126.centroid.n=160'
model1_instance = 'Final_code/model/final_model/240903_174724.centered_instance.n=160'



dir_folder = Path(Results_dir).parent.joinpath(str(Path(Results_dir).name).split('.')[0])
dir_folder.mkdir(exist_ok=True, parents=True)
print(video.shape, dir_folder)
# %%

predictor = sleap.load_model([model_1_centroid, model1_instance], batch_size=4)

predictor.tracker = Tracker.make_tracker_by_name(tracker="flow")
#sleap-track video  --frames 0-20  --tracking.tracker simple  -m model_1_centroid -m model1_instance 

# %%
label = predictor.predict(video, make_labels=True)


print(label)


label.export(dir_folder.joinpath('featuresN.h5'))
# %%
filename = dir_folder.joinpath('featuresN.h5')
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()


#%%


from scipy.interpolate import interp1d


#locations = fill_missing(locations)
frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)


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
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel


def smooth_angular_vel(node_ang_vel, win=25, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    angular_vel_smooth = np.zeros_like(node_ang_vel)
    
    for c in range(node_ang_vel.shape[-1]):
        angular_vel_smooth[:, c] = savgol_filter(node_ang_vel[:, c], win, poly)
    
    #node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return angular_vel_smooth


def instance_node_velocities(instance_idx, locations, frame_count, node_count):
    fly_node_locations = locations[:, :, :, instance_idx]
    fly_node_velocities = np.zeros((frame_count, node_count))
    fly_node_angular_velocities = np.zeros((frame_count, node_count))

    for n in range(0, node_count):
        fly_node_velocities[:, n] = smooth_diff(fly_node_locations[:, n, :])
        for t in range(1, frame_count - 1):
            vec1 = fly_node_locations[t + 1, n, :] - fly_node_locations[t, n, :]
            vec2 = fly_node_locations[t, n, :] - fly_node_locations[t - 1, n, :]
            angle1 = np.arctan2(vec1[1], vec1[0])
            angle2 = np.arctan2(vec2[1], vec2[0])
            angular_velocity_node = angle1 - angle2
            fly_node_angular_velocities[t, n] = angular_velocity_node
        fly_node_angular_velocities[0, n] = np.nan
        fly_node_angular_velocities[frame_count-1, n] = np.nan

    return fly_node_velocities, fly_node_angular_velocities



def plot_instance_node_velocities(instance_idx, node_velocities):
    plt.figure(figsize=(20,8))
    plt.imshow(node_velocities.T, aspect='auto',interpolation="nearest", cmap='jet')

    # Add color bar
    cbar = plt.colorbar()
    cbar.set_label('Velocity [mm/s]', rotation=90, labelpad=15) 

    plt.xlabel('frames')
    plt.ylabel('nodes')
    plt.yticks(np.arange(node_count), node_names, rotation=20)
    plt.title(f'Fly {instance_idx} node velocities')



#%%
fly_ID = 2
fly_node_locations = locations[:, :, :, fly_ID]

velocities, angular_velocity = instance_node_velocities(fly_ID, locations, frame_count, node_count)
velocities = velocities * (mm_2_px / time)
angular_velocity = angular_velocity /time
motion_mode =get_events( velocities[:,1], 19,mm_2_px)
angular_velocity_pd = pd.DataFrame(angular_velocity) 
print(angular_velocity_pd)
#%%

lf = label[0]
sleap.nn.viz.plot_img(lf.image, scale=0.6)
sleap.nn.viz.plot_instances(lf, color_by_track=True, tracks=label.tracks)
for i in range(node_count):
    plt.scatter(fly_node_locations[:,i,0], fly_node_locations[:,i,1], s=20, c = np.arange(frame_count),cmap = 'plasma')
plt.show()
plt.savefig(dir_folder.joinpath(f'Image_fly_id_{fly_ID}.jpeg'), dpi=300,bbox_inches='tight', pad_inches=0)

plt.figure()
plt.plot(velocities[:,1])
plt.plot(motion_mode)

#%%
plot_instance_node_velocities(fly_ID, angular_velocity)

plt.savefig(dir_folder.joinpath(f'velocity_profile_fly_id_{fly_ID}.jpeg'), dpi=300,bbox_inches='tight', pad_inches=0)


# %%


""""
Save skeletons
"""


for i in range(0,10,1):

    lf = label[i]
    sleap.nn.viz.plot_img(lf.image, scale=0.6)
    sleap.nn.viz.plot_instances(lf, color_by_track=True, tracks=label.tracks)
    figname = dir_folder.joinpath(f"{i:04d}.png")
    plt.axis('off')
    plt.savefig(figname, pad_inches=0, bbox_inches="tight", dpi=400)
    plt.close()



#%%

cmd = f"ffmpeg -framerate 10 -y  -hide_banner -loglevel error -pattern_type glob -i '{str(dir_folder)}/*.png' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p '{str(dir_folder)}/DT_C_spline.mp4' "
subprocess.run(cmd, shell=True)