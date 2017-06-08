/*==================================
Author: B-one
Date  : 2017/5/30
Task  : the translation filter learning of DSST tracker 
====================================*/

#include "dsst.hpp"
#include <cmath>

extern int g_trackState;

void DSST::TransLearn(cv::Mat &im){
	vector<cv::Mat> trans_xl = GetTransSample(im); //这个耗时为0.02，需要改进
	vector<cv::Mat> trans_xlf(trans_xl.size());

	vector<cv::Mat> new_hf_num(trans_xl.size()); 
	cv::Mat new_hf_den;

	for (int i = 0; i < trans_xl.size(); i++){
		cv::dft(trans_xl[i], trans_xlf[i], CV_HAL_DFT_COMPLEX_OUTPUT);//take a fft2 
		cv::mulSpectrums(trans_yf_, trans_xlf[i], new_hf_num[i], 0, true);
		cv::Mat tmp_den;
		cv::mulSpectrums(trans_xlf[i], trans_xlf[i], tmp_den, 0, true);
		if (i == 0) new_hf_den = tmp_den;
		else new_hf_den += tmp_den;
	}

	new_hf_den = GetRealMatrix(new_hf_den);//get a real part

  //-----------Update the model-------------
	if (g_trackState == -1){//first frame
		_hf_num = new_hf_num;
		_hf_den = new_hf_den;
	}else if (g_trackState == 1){//other frame
		for (int i = 0; i < _hf_num.size(); i++){
			_hf_num[i] = (1 - learning_rate)*_hf_num[i] + learning_rate*new_hf_num[i];
		}
		_hf_den = (1 - learning_rate)*_hf_den + learning_rate*new_hf_den;
	}else{
		cout << "something of hf is wrong!" << endl;
	}
}

/**************************
Function: get a patch from image and extract fhog of the patch
@ im : image
@return: a fhog map
***************************/
vector<cv::Mat> DSST::GetTransSample(cv::Mat &im){
	cv::Mat sample;
	cv::Size sample_sz = FloorSizeScale(window_sz, currentScaleFactor);//get the size of ROI

  //in case of the size of sample < 2*2
	if (sample_sz.width <= 1) sample_sz.width = 2;
	if (sample_sz.height <= 1) sample_sz.height = 2;

  //get left-top coordinate and right-buttom coordinate
	cv::Point lefttop(pos_.x+1- cvFloor(float(sample_sz.width) / 2.0), 
						pos_.y+1 - cvFloor(float(sample_sz.height) / 2.0));
	cv::Point rightbottom(lefttop.x + sample_sz.width, lefttop.y + sample_sz.height);
	cv::Point lefttop_limit(max(lefttop.x, 0), max(lefttop.y, 0));
	cv::Point rightbottom_limit(min(rightbottom.x, im.cols - 1), min(rightbottom.y, im.rows - 1));
	cv::Rect roiRect(lefttop_limit, rightbottom_limit);//get a ideal rect
	im(roiRect).copyTo(sample);

  //padding a border, if sample is out the size of image  
	cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
		max(rightbottom.x - im.cols + 1, 0), max(rightbottom.y - im.rows + 1, 0));
	if (border != cv::Rect(0, 0, 0, 0)){
		cv::copyMakeBorder(sample, sample, border.y, border.height, border.x, border.width, cv::BORDER_REFLECT);
	}

  //resize sample size to window_sz
	int interpolation;
	if (window_sz.width > sample_sz.height) interpolation = CV_INTER_LINEAR;
	else interpolation = CV_INTER_AREA;
	cv::resize(sample, sample, window_sz, 0, 0, interpolation);


	vector<cv::Mat> out = GetTransFeatures(sample); //0.018s
	return out;
}

/**************************
Function: get fhog feature of translation sample
@ im_patch : the search window
@ cell_size: a parameter of fhog
@return: a fhog map
***************************/
std::vector<cv::Mat> DSST::GetTransFeatures(cv::Mat im_patch, int cell_size){
	cv::Mat x;
	vector<cv::Mat> x_vector(28);
	vector<cv::Mat> tmp_vector;

	if (im_patch.channels() == 3) cv::cvtColor(im_patch, im_patch, CV_BGR2GRAY);
	im_patch.convertTo(im_patch, CV_32FC1);

	x_vector[0] = im_patch / 255.0 - 0.5;
	tmp_vector = trans_fhog.extract(im_patch, 2, cell_size);//31 channels = 27 gradiens orientation + 4 textures


	for (int i = 0; i < 27; i++){
		x_vector[i + 1] = tmp_vector[i];
	}
	for (int i = 0; i < 28; i++){
		x_vector[i] = x_vector[i].mul(trans_hann_window);
	}
	return x_vector;
}