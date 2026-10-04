#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <numbers>
#include <thread>
#include <chrono>

using namespace std;

vector<vector<float>> createGaussianKernel(int size, float sigma) {
	vector<vector<float>> kernel(
		size,
		vector<float>(size)
	);

	int center = size / 2;
	float sum = 0.0f;

	for (int x = -center; x <= center; x++) {
		for (int y = -center; y <= center; y++) {

			float value =
				exp(-(x * x + y * y) / (2.0f * sigma * sigma))
				/ (2.0f * std::numbers::pi_v<float> *sigma * sigma);

			kernel[x + center][y + center] = value;

			sum += value;
		}
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			kernel[i][j] /= sum;
		}
	}

	return kernel;
}

float kernelSum(vector<vector<float>> kernel) {
	float kernelSum = 0.0f;

	for (const auto& row : kernel) {
		for (float value : row) {
			kernelSum += value;
		}
	}
	return kernelSum;
}

void convolutionSeq(cv::Mat& img, vector<vector<float>>& kernel, cv::Mat& res) {
	int rows = img.rows; int cols = img.cols;
	int stepSize = kernel.size() / 2;
	float ks = kernelSum(kernel);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float total = 0.0f;
			for (int x = -stepSize; x <= stepSize; x++) {
				for (int y = -stepSize; y <= stepSize; y++) {
					
					if (i - x < 0 || j - y < 0 || i - x > rows - 1 || j - y > cols - 1) {
						continue;
					}
					float pixel = static_cast<float>(img.at<uint8_t>(i - x, j - y));
					total += pixel * kernel[stepSize + x][stepSize + y];
				}
			}
			res.at<uint8_t>(i, j) = cv::saturate_cast<uint8_t>(total);
		}
	}
}
void convolutionSeq(cv::Mat& img, vector<vector<float>>& kernel, cv::Mat& res, string& paddingStyle) {
	if (paddingStyle == "valid") {
		int rows = img.rows;
		int cols = img.cols;
		int stepSize = kernel.size() / 2;
		float ks = kernelSum(kernel);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {

				float total = 0.0f;
				bool outOfBounds = false;

				for (int x = -stepSize; x <= stepSize; x++) {
					for (int y = -stepSize; y <= stepSize; y++) {

						int row = i - x;
						int col = j - y;

						if (row < 0 || col < 0 ||
							row >= rows || col >= cols) {

							outOfBounds = true;
							break;
						}

						float pixel =
							static_cast<float>(img.at<uint8_t>(row, col));

						total += pixel *
							kernel[stepSize + x][stepSize + y];
					}
					if (outOfBounds) {
						break;
					}
				}
				if (!outOfBounds) {
					res.at<uint8_t>(
						i - stepSize,
						j - stepSize
					) = cv::saturate_cast<uint8_t>(total);
				}
			}
		}
		return;
	}
	else {
		int paddingValue = stoi(paddingStyle);

		int rows = img.rows; int cols = img.cols;
		int stepSize = kernel.size() / 2;
		float ks = kernelSum(kernel);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				float total = 0.0f;
				for (int x = -stepSize; x <= stepSize; x++) {
					for (int y = -stepSize; y <= stepSize; y++) {
						float pixel;
						if (i - x < 0 || j - y < 0 || i - x > rows - 1|| j - y > cols - 1) {
							pixel = paddingValue;
						}
						else {
							pixel = static_cast<float>(img.at<uint8_t>(i - x, j - y));
						}
						total += pixel * kernel[stepSize + x][stepSize + y];
					}
				}
				res.at<uint8_t>(i, j) = cv::saturate_cast<uint8_t>(total/ks);
			}
		}
	}
}

void convolutionPar(cv::Mat& img, vector<vector<float>>& kernel, cv::Mat& res, int widthStart, int heightStart, int gridSizeX, int gridSizeY, float ks) {
	int stepSize = kernel.size() / 2;
	
	for (int i = heightStart; i < heightStart+gridSizeY; i++) {
		for (int j = widthStart; j < widthStart+gridSizeX; j++) {
			float total = 0.0f;
			for (int x = -stepSize; x <= stepSize; x++) {
				for (int y = -stepSize; y <= stepSize; y++) {
					if (i - x < 0 || j - y < 0 || i - x > img.rows - 1 || j - y > img.cols - 1) {
						continue;
					}
					float pixel = static_cast<float>(img.at<uint8_t>(i - x, j - y));
					total += pixel * kernel[stepSize + x][stepSize + y];
				}
			}
			res.at<uint8_t>(i, j) = cv::saturate_cast<uint8_t>(total);
		}
	}
}

int main(int argc, char** argv) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	cv::Mat img = cv::imread("lena_gray.png", cv::IMREAD_GRAYSCALE);
	// 512x512 grayscale img
	//cv::resize(img, img, cv::Size(10, 10));

	vector<vector<float>> kernel = createGaussianKernel(7, 1.0f);

	cv::Mat res(img.rows, img.cols, CV_8UC1);

	cv::imshow("Original Image", img);

	//string pad = "valid";
	//convolutionSeq(img, kernel, res, pad);
	
	// Parallel start point
	int numThr = atoi(argv[1]);

	int gridRows = 1;
	int gridCols = numThr;

	for (int r = 1; r * r <= numThr; r++) {
		if (numThr % r == 0) {
			gridRows = r;
			gridCols = numThr / r;
		}
	}

	int baseWidth = img.cols / gridCols;
	int widthRem = img.cols % gridCols;

	int baseHeight = img.rows / gridRows;
	int heightRem = img.rows % gridRows;

	vector<thread> thrs(numThr);

	int threadIndex = 0;
	int heightStart = 0;
	float ks = kernelSum(kernel);

	auto startTime = chrono::high_resolution_clock::now();

	for (int gy = 0; gy < gridRows; gy++) {
		int gridSizeY =	baseHeight + (gy < heightRem ? 1 : 0);
		int widthStart = 0;

		for (int gx = 0; gx < gridCols; gx++) {
			int gridSizeX =
				baseWidth + (gx < widthRem ? 1 : 0);

			thrs[threadIndex] = thread(convolutionPar, ref(img), ref(kernel), ref(res), widthStart, heightStart, gridSizeX, gridSizeY, ks);
			widthStart += gridSizeX;
			threadIndex++;
		}
		heightStart += gridSizeY;
	}

	for (int i = 0; i < numThr; i++) {
		thrs[i].join();
	}

	auto endTime = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
	double millisecond = duration / 1000.0;

	cout << duration << ' ' << millisecond;

	cv::imshow("After filter", res);
	cv::waitKey(0);
	cv::destroyAllWindows();


	return 0;
}