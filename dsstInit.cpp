/*==================================
Author: B-one
Date  : 2017/5/30
Task  : Init a dsst tracker
====================================*/


#include "dsst.hpp"
#include <cmath>


void DSST::Init(cv::Mat image, cv::Rect rect){
	pos_ = cv::Point(rect.x + cvFloor((float(rect.width) / 2.)),
									rect.y + cvFloor((float(rect.height)) / 2.));
	current_target_sz = init_target_sz = rect.size();
	padding = 0.5f;
	window_sz = FloorSizeScale(init_target_sz, 1 + padding);

  //----------Create a 2D Gaussian(fft) labels of translation--------------
	float output_sigma = sqrt(init_target_sz.area())*output_sigma_factor;
	cv::dft(TranGaussianLabels(output_sigma, window_sz), trans_yf_,CV_HAL_DFT_COMPLEX_OUTPUT);
	trans_hann_window = CreatTransHann(trans_yf_.size());//create a 2D hann windows

  //----------Create a 1D Gaussian(fft) labels of Scale--------------
	float scale_sigma = float(number_of_scales) / sqrt(33) *scale_sigma_factor;
	cv::dft(ScaleGaussianLabels(scale_sigma, number_of_scales), scale_yf_,CV_HAL_DFT_COMPLEX_OUTPUT); //return a row vector
	scale_hann_window = CreatScaleHann(scale_yf_.size().width);//return a col vector

  //generate a list of scale factors

	for (int i = 0; i < number_of_scales; i++){
		float tmp = pow(scale_step, ceil(float(number_of_scales) / 2.) - i - 1);
		scaleFactors.push_back(tmp);
	}
	//for (auto i : scaleFactors) cout << i << " ";
	//cout << endl;

  //get a norm_sz of scale patch
	float scale_model_factor = 1;
	if (init_target_sz.area()>scale_model_max_area){
		scale_model_factor = sqrt(float(scale_model_max_area) / (init_target_sz.area()));
	}
	scale_model_sz = FloorSizeScale(init_target_sz, scale_model_factor);

  //------------calculate min and max scale factor-----
	float minscale_tmp = max(float(5) / window_sz.height, float(5) / window_sz.width);
	float maxscale_tmp = min(float(image.size().height) / init_target_sz.height,
								float(image.size().width) / init_target_sz.width);
	min_scale_factor = std::pow(scale_step, std::ceil(std::log(minscale_tmp) / std::log(scale_step)));
	max_scale_factor = std::pow(scale_step, std::floor(std::log(maxscale_tmp) / std::log(scale_step)));

  //------Learning-----------------

	TransLearn(image);
	ScaleLearn(image);
}