#pragma once
#include <opencv2/core.hpp>
#include "enums.hpp"

namespace kmeans {
	// Segment a frame using k-means clustering with the specified backend
	//
	// Args:
	//   frame: input image (cv::Mat, 3-channel BGR)
	//   k: number of clusters for K-means
	//   backend: which backend to use (sequential, CUDA, threaded, MPI)
	//   sample_size: maximum number of points in the coreset
	//   color_scale: scaling factor for the color dimensions in the feature vectors
	//   spatial_scale: scaling factor for the spatial dimensions in the feature vectors
	//   flatCenters: precomputed cluster centers(flattened, size k * 5). Used for MPI.
	//   totalRows: total number of rows in the full frame (used for MPI spatial calculations).
	//   totalCols: total number of columns in the full frame (used for MPI spatial calculations).

	//
	// Returns:
	//   Segmented image (cv::Mat, 3-channel BGR) where each pixel is colored by its cluster center
	cv::Mat segmentFrameWithKMeans(
		Algorithm algorithm,
		const cv::Mat& frame,
		int k,
		int sample_size = 0,
		float color_scale = 1.0f,
		float spatial_scale = 0.5f
	);
}