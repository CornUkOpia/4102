// Written in C++ because I (Gabriel) find it a bit easier.

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <iostream>
#include <cstdlib>

#define CONF_FILENAME "./calibOptions.xml"
#define IN_DIR "./calibPics/"
#define OUT_DIR "./calibPics_out/"
#define EXTENSION ".jpg"
#define CALIB_FILENAME "./calibMatrix.xml"

int main(int argc, char** argv)
{
	// Variables and their default values.
	int checkerboardWidth = 7;
	int checkerboardHeight = 7;
	int squareSizeMM = 20;	// I didn't have a ruler to measure.
	int numPics = 16;
	int cornerSubPix_winSize = 11;

	// Load settings.
	// UNDONE: It's not too inconvenient to hardcode the parameters and recompile.
	/*cv::FileStorage confFile(CONF_FILENAME, cv::FileStorage::READ);
	if ( confFile.isOpened() )
	{
		confFile["checkerboardWidth"] >> checkerboardWidth;
		confFile["checkerboardHeight"] >> checkerboardHeight;
		confFile["squareSizeMM"] >> squareSizeMM;
		confFile["numPics"] >> numPics;
		confFile["cornerSubPix_winSize"] >> cornerSubPix_winSize;
		confFile.release();
	}
	else
	{
		cv::FileStorage confFileOut(CONF_FILENAME, cv::FileStorage::WRITE);
		confFileOut << "checkerboardWidth" << checkerboardWidth;
		confFileOut << "checkerboardHeight" << checkerboardHeight;
		confFileOut << "squareSizeMM" << squareSizeMM;
		confFileOut << "numPics" << numPics;
		confFileOut << "cornerSubPix_winSize" << cornerSubPix_winSize;
		confFileOut.release();
	}*/

	int i;
	cv::Mat curSrc, curSrcWithPoints;
	std::vector<std::vector<cv::Point2f>> boardPoints;
	std::vector<cv::Point2f> curPointBuffer;
	char filepath[49], filename[33], filenumber[3];
	bool found;
	cv::Size boardSize(checkerboardWidth, checkerboardHeight);
	cv::Size imageSize;
	for ( i = 0; i < numPics; i++ )
	{
		// NOTE: Probably not great from a security perspective, but that's fine here.
		memset(filepath, '\0', 49);
		memset(filename, '\0', 33);
		memset(filenumber, '\0', 3);
		strcpy(filepath, IN_DIR);
		sprintf(filenumber, "%d", i);
		strcat(filename, filenumber);
		strcat(filename, EXTENSION);
		strcat(filepath, filename);
		std::cout << filepath << "\n";
		curSrc = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
		imageSize = curSrc.size();

		found = cv::findChessboardCorners(curSrc, boardSize, curPointBuffer,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK);
		std::cout << "\tCheckerboard corner detection run\n";
		if ( found )
		{
			cv::cornerSubPix( curSrc, curPointBuffer, cv::Size(cornerSubPix_winSize, cornerSubPix_winSize), cv::Size(-1,-1),
					cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.0001) );
			std::cout << "\tCorners refined\n";
			boardPoints.push_back(curPointBuffer);
			cv::cvtColor(curSrc, curSrcWithPoints, CV_32F);
			cv::drawChessboardCorners( curSrcWithPoints, boardSize, curPointBuffer, found );
			strcpy(filepath, OUT_DIR);
			strcat(filepath, filename);
			cv::imwrite(filepath, curSrcWithPoints);
			std::cout << "\tCheckerboard written\n";
		}
		else
		{
			std::cout << "\tNO CORNERS FOUND\n";
		}
	}

	if ( boardPoints.size() > 0 )
	{
		// Taken from OpenCV sample code file "camera-calibration.cpp".
		// Hate to admit it, but I'm not really sure what the objectPoints and newObjPoints parameters are for.
		std::vector<std::vector<cv::Point3f>> objectPoints(1);
		int j;
		for (j = 0; j < checkerboardHeight; j++)
			for (i = 0; i < checkerboardWidth; i++)
				objectPoints[0].push_back(cv::Point3f(i*squareSizeMM, j*squareSizeMM, 0));
		objectPoints[0][checkerboardWidth - 1].x = objectPoints[0][0].x + (squareSizeMM * (checkerboardWidth - 1));
		std::vector<cv::Point3f> newObjPoints = objectPoints[0];
		objectPoints.resize(boardPoints.size(), objectPoints[0]);

		// Calibrate the camera.
		cv::Mat cameraMatrix, distortionCoeffs;
		std::vector<cv::Mat> rvecs, tvecs;
		double rmsReprojError = cv::calibrateCameraRO(objectPoints, boardPoints, imageSize, checkerboardWidth - 1, cameraMatrix,
								distortionCoeffs, rvecs, tvecs, newObjPoints, 0);

		// Print the camera's intrinsic parameter matrix and the distortion coefficients, then write it to a file.
		std::cout << "Camera matrix:\nK = ";
		for (j = 0; j < cameraMatrix.rows; j++)
		{
			for (i = 0; i < cameraMatrix.cols; i++)
			{
				std::cout << "\t" << cameraMatrix.at<double>(j, i);
			}
			std::cout << "\n";
		}
		std::cout << "Distortion coefficients:\ndistCoeffs = ";
		for (j = 0; j < distortionCoeffs.rows; j++)
		{
			for (i = 0; i < cameraMatrix.cols; i++)
			{
				std::cout << "\t" << distortionCoeffs.at<double>(j, i);
			}
			std::cout << "\n";
		}
		std::cout << "RMS reprojection error: " << rmsReprojError << "\n";
		cv::FileStorage calibFile(CALIB_FILENAME, cv::FileStorage::WRITE);
		calibFile << "camera_K" << cameraMatrix;
		calibFile << "camera_distCoeffs" << distortionCoeffs;
		calibFile.release();
	}
	
	return 0;
}
