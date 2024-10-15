import pickle

import gtsam
from BundleAdjustment import BundleAdjustment
from PoseGraph import PoseGraph
from tracking_database import TrackingDB
from BundleWindow import BundleWindow
from gtsam.utils import plot
from gtsam import symbol
import numpy as np
import matplotlib.pyplot as plt
# from algorithms_library import plot_pose_graph_poses
NUM_FRAMES = 3360

def get_relative_pose_and_cov_mat_last_kf(window, numForSymbolLastFrame = 10):
    #assume window optimized
    first_camera = window.get_optimized_first_camera()
    last_camera = window.get_optimized_last_camera()
    marginals = window.marginals()
    keys = gtsam.KeyVector()
    sym_c0 = symbol('c', 0)
    sym_ck = window.get_last_frame_symbol()
    # sym_ck = symbol('c', numForSymbolLastFrame)
    keys.append(sym_c0)
    keys.append(sym_ck)
    infoCovMatrix = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    covMatrix = np.linalg.inv(infoCovMatrix)
    relative_pose = first_camera.between(last_camera)
    return relative_pose, covMatrix




def q1(db):
    #todo: check avout the problem with the bundle window size - 20 - maybe our constraints not well enough (exercise 4)
    first_window = BundleWindow(0, 10, db=db)
    # first_window.create_factor_graph()

    current_Rt_gtsam = gtsam.Pose3()
    prev = current_Rt_gtsam

    first_window.add_trans(current_Rt_gtsam)

    first_window.cam_insert(0, current_Rt_gtsam)
    sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])

    first_window.add_cam_factor( pose_uncertainty=gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas))
    first_window.optimize()
    check = True
    first_window.save("bundle_windows/first_window3")
    new_window = BundleWindow(db)
    new_window.load("bundle_windows/first_window3")
    #check if window loaded well print the values
    print(new_window.get_optimized_values())
    if check == True:
        return
    pose_c0 = first_window.get_optimized_first_camera()
    pose_ck = first_window.get_optimized_last_camera()

    marginals = first_window.marginals()
    values = first_window.get_optimized_values()

    gtsam.utils.plot.plot_trajectory(fignum=0, values=values,marginals=marginals, scale=1, title="Covariance poses first bundle")
    plt.savefig("poses_first_bundle_with_covariance.png")
    keys = gtsam.KeyVector()
    sym_c0 = symbol('c', 0)
    sym_ck = symbol('c', 10)
    keys.append(sym_c0)
    keys.append(sym_ck)
    # covMatrix = marginals.jointMarginalCovariance(keys).fullMatrix()
    infoCovMatrix = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    covMatrix = np.linalg.inv(infoCovMatrix)
    print(covMatrix)
    relative_pose = pose_c0.between(pose_ck)
    print(f"Relative pose between c0 and ck: {relative_pose}")
    #save the window in the folder "bundle_windows"

    # Save the window
    first_window.save("bundle_windows/first_window3")
    # with open("bundle_windows/first_window2.pkl", 'wb') as file:
    #     pickle.dump(first_window, file)
def q2(db):
    bundle = BundleAdjustment(db)
    bundle.load("bundle_data_window_size_20_witohut_bad_matches_ver2/", "bundle with window_size_20_witohut_bad_matches_ver2", 5)

    relative_poses, cov_matrices = [], []
    for i, window in enumerate(bundle.get_windows()):
        print(f"try to get relative pose and cov mat for window {i}")
        relative_pose, cov_matrix = get_relative_pose_and_cov_mat_last_kf(window, 5)
        relative_poses.append(relative_pose)
        cov_matrices.append(cov_matrix)
    poseGraph = PoseGraph(db, bundle, relative_poses, cov_matrices, bundle.get_key_frames())
    poseGraph.create_factor_graph()
    gtsam.utils.plot.plot_trajectory(fignum=0, values=poseGraph.get_initial_estimate(), scale=1, title="Initial estimate")
    plt.savefig("initial_estimate_for_pose_graph.png")
    plt.close()
    poseGraph.optimize()
    gtsam.utils.plot.plot_trajectory(fignum=0, values=poseGraph.get_optimized_values(), scale=1, title="Optimized values", d2)
    plt.savefig("optimized_values_for_pose_graph.png")
    plt.close()
    marginals = poseGraph.marginals()
    gtsam.utils.plot.plot_trajectory(fignum=0, values=poseGraph.get_optimized_values(), marginals=marginals, scale=1, title="Optimized values with marginals", d2_view=True)
    plt.savefig("optimized_values_with_marginals_for_pose_graph.png")
    plt.close()


    # plot_pose_graph_poses(poseGraph.get_initial_estimate())
    # poseGraph.optimize()
    # plot_pose_graph_poses(poseGraph.get_optimized_values())
    # plot_pose_graph_poses(poseGraph.get_optimized_values(), covariances=poseGraph.cov_matrices)

if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    # q1(db)
    q2(db)