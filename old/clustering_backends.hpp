#pragma once
#include <opencv2/core.hpp>
#include "common/enums.hpp"

// Segment a frame using K-means clustering with different backends (sequential, threaded, MPI, CUDA)
// See: clustering_seq.cpp, clustering_thr.cpp, clustering_mpi.cpp, clustering_cuda.cu, clustering_thrpool.cpp 
// for implementations, and include/clustering_backends.hpp for declarations/argument meanings

namespace kmeans
{
	cv::Mat segmentFrameWithKMeans_regular(
		const cv::Mat& frame,
		int k,
		Initialization initialization,
	);

	cv::Mat segmentFrameWithKMeans_quantum(
		const cv::Mat& frame,
		int k,
	);
}