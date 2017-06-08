/*==================================
Author: B-one
Date  : 2017/5/30
Task  : dsst tracker tools
====================================*/


#include "dsst.hpp"
#include <cmath>


/**************************
Function: get a 2D gaussian shape(labels) for translation filter
@sigma : standard deviation
@ sz   : the size(rows,cols) of 2D gaussian 
@return: 2D gaussian shape 
***************************/
cv::Mat DSST::TranGaussianLabels(float sigma, cv::Size sz){
	cv::Mat_<float> labels(sz.height, sz.width);
	int h2 = std::floor(float(sz.height)/2.);
	int w2 = std::floor(float(sz.width) / 2.);
	float mult = -0.5 / (sigma*sigma);
	for (int i = 0; i < sz.height; i++){
		for (int j = 0; j < sz.width; j++){
			int ih = i - h2+1;
			int jw = j - w2+1;
			labels.at<float>(i, j) = std::exp(mult*float(ih*ih + jw*jw));
		}
	}
	return labels;
}

/**************************
Function: get a 1D gaussian shape(labels) for scale filter
@sigma : standard deviation
@ n   : the size(cols) of 1D gaussian
@return: 1D gaussian shape
***************************/
cv::Mat DSST::ScaleGaussianLabels(float sigma, int n) {
	cv::Mat labels(1, n, CV_32F);

	float sigma2 = sigma*sigma;
	int x = 0;
	float tmp = 0;
	for (int i = 0; i < n; i++){
		x = i - ceil(float(n) / 2.) + 1;
		tmp = exp(-0.5* (x*x) / sigma2);
		labels.at<float>(0, i) = tmp;
	}

	return labels;
}

/**************************
Function: creat a hann windows for translation filter
@ sz : the dimention of translation input
@return: 2D Hann in the same size
***************************/
cv::Mat DSST::CreatTransHann(cv::Size sz){ 
	cv::Mat tmp1(sz.height, 1, CV_32F);
	cv::Mat tmp2(1,sz.width, CV_32F);

	for (int i = 0; i < sz.height; i++){
		tmp1.at<float>(i, 0) = 0.5*(1 - std::cos(2 * CV_PI*i / (sz.height - 1)));
	}
	for (int i = 0; i < sz.width; i++){
		tmp2.at<float>(0,i) = 0.5*(1 - std::cos(2 * CV_PI*i / (sz.width - 1)));
	}
	return tmp1*tmp2;
}

/**************************
Function: creat a hann windows for scale filter
@ sz    : # of scales 
@return : 1D Hann in the same size, (col vector)
***************************/
cv::Mat DSST::CreatScaleHann(int n){
	cv::Mat tmp1(n, 1, CV_32F);
	for (int i = 0; i < n; i++){
		tmp1.at<float>(i, 0) = 0.5*(1 - cos(2 * CV_PI*i / (n - 1)));
	}
	return tmp1;
}

/**************************
Function: resize the old size image to new size
@ src    : source image
@ dst    : dst image
@ old_sz : the old size of iamge
@ new_sz : the new size of image
@return : void
***************************/
void DSST::DsstResize(cv::Mat &src, cv::Mat &dst, cv::Size old_sz, cv::Size new_sz){
	int interpolation;
	if (new_sz.width > old_sz.height) interpolation = CV_INTER_LINEAR;
	else interpolation = CV_INTER_AREA;
	cv::resize(src, dst, new_sz, 0, 0, interpolation);
}

