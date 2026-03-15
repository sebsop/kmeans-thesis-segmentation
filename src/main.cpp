#include "video_io.hpp"
#include "coreset.hpp"
#include <opencv2/opencv.hpp>

// Entry point
int main(int argc, char** argv) 
{
	// Prevent OpenCV from inserting logs in the console
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	std::cout << "Starting application..." << std::endl << std::endl;
	std::cout << "Press 'Esc' to quit." << std::endl;
	std::cout << "Press '1' to use the regular k-means algorithm." << std::endl;
	std::cout << "Press '2' to use the k-means++ algorithm" << std::endl;
	std::cout << "Press '3' to use the quantum k-means algorithm" << std::endl << std::endl;

	std::cout << "Hint: use the 'k' slider to adjust the number of segments.";

	// Display the webcam feed using OpenCV
	showWebcamFeed();

	std::cout << std::endl << std::endl << std::endl << "Shutting down..." << std::endl;

    return 0;
}