Stereo Visual SLAM Pipeline (KITTI)
A modular Visual SLAM system implementing the full pipeline from raw stereo imagery to a globally optimized pose graph using GTSAM.



📍 Project Roadmap & Status:

This project was developed through a series of incremental engineering milestones, moving from geometric foundations to probabilistic optimization.


Phase 1: Feature Engineering & Stereo Geometry

	•Exercise 1: Feature Tracking — Robust ORB/SIFT detection and descriptor matching.
	
	•Exercise 2: Geometric Outlier Rejection — Implementation of a vertical deviation filter for rectified stereo pairs (rejection of ~16.5% erroneous
	matches) and Linear Least Squares triangulation
	
		•Result: Generated 3D point clouds with a median distance of $0.8 \times 10^{-6}$ units compared to OpenCV’s implementation.
		
		
Phase 2: Visual Odometry (VO)
	
	• Exercise 3: PnP & RANSAC — Estimation of relative motion between consecutive stereo pairs. Implemented a custom RANSAC framework to identify 
	"supporting" 3D points.
	
		•Result: Successfully estimated vehicle trajectory over 3,000+ frames, closely matching KITTI ground truth.
	
	•Exercise 4: Multi-Frame Tracking Database — Developed a scalable database to manage 580,000+ feature tracks with a mean track length of 3.4 frames.
	

Phase 3: Probabilistic Optimization (The "Brain")
	
	•Exercise 5: Sliding Window Bundle Adjustment (BA) — [Current Focus] Implementation of local factor graph optimization using GTSAM.
	
		•Result: Reduced total factor graph error from 16.40 to 7.32 in local windows, significantly refining point projections.
	
	•Exercise 6: Global Pose Graph Optimization — Summary of trajectory via keyframes and relative pose constraints.
	
		•Result: Optimized global trajectory with marginal covariances, achieving high consistency with ground truth.
		


🛠️ Tech Stack
	
	•Core: Python, NumPy, OpenCV,
	
	•Optimization: Feature-Tracking, PnP, RANSAC, GTSAM (Factor Graphs)
	
	•Dataset: KITTI Odometry Benchmark (Sequence 00) - https://www.cvlibs.net/datasets/kitti/eval_odometry.php



