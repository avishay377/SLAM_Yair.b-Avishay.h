import gtsam
from BundleAdjustment import BundleAdjustment
from tracking_database import TrackingDB
from BundleWindow import BundleWindow
from gtsam.utils import plot
from gtsam import symbol
import numpy as np
import matplotlib.pyplot as plt
NUM_FRAMES = 3360
def q1(db):

    #todo: check avout the problem with the bundle window size - 20 - maybe our constraints not well enough (exercise 4)
    first_window = BundleWindow(0, 10, db=db)
    first_window.create_factor_graph()
    first_window.optimize()
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


def q2(db):



if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    q1(db)