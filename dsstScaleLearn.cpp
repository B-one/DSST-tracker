/*==================================
Author: B-one
Date  : 2017/5/30
Task  : the translation filter learning of DSST tracker
====================================*/

#include "dsst.hpp"
#include <cmath>

extern int g_trackState;


void DSST::ScaleLearn(cv::Mat &image){
	
	cv::Mat scale_xs = GetScaleSample(image);
	cv::Mat scale_xsf = cv::Mat(cv::Size(scale_xs.cols, scale_xs.rows), CV_32F, float(0));
	
	cv::Mat new_sf_num; 
	cv::Mat new_sf_den;

	if (g_trackState == -1){//first frame
		scale_yf_ = cv::repeat(scale_yf_, scale_xs.rows, 1);
	}

	cv::dft(scale_xs, scale_xsf, CV_HAL_DFT_COMPLEX_OUTPUT | CV_HAL_DFT_ROWS);//fft1 for every row 
	cv::mulSpectrums(scale_yf_, scale_xsf, new_sf_num, 0, true);
	cv::mulSpectrums(scale_xsf, scale_xsf, new_sf_den, 0, true);
	cv::reduce(new_sf_den, new_sf_den, 0, CV_REDUCE_SUM);

	new_sf_den = GetRealMatrix(new_sf_den);//get a real part

	if (g_trackState == -1){//first frame
		_sf_num = new_sf_num;
		_sf_den = new_sf_den;
	}else if (g_trackState == 1){//other frame
		cv::addWeighted(_sf_num, (1 - learning_rate), new_sf_num, learning_rate, 0, _sf_num);
		cv::addWeighted(_sf_den, (1 - learning_rate), new_sf_den, learning_rate, 0, _sf_den);
	}else{
		cout << "something of sf is wrong!" << endl;
	}
}

/**************************
Function: get a fhog scale sample 
@ im : image
@return: a fhog map
***************************/
cv::Mat DSST::GetScaleSample(cv::Mat &im){
	vector<float> cScaleFactors;
	for (int i = 0; i < number_of_scales; i++){//get current scale factors
		cScaleFactors.push_back(scaleFactors[i] * currentScaleFactor);
	}

	cv::Mat out;

	for (int i = 0; i < number_of_scales; i++){//遍历所有尺度
		cv::Mat sample;
		cv::Size sample_sz = FloorSizeScale(init_target_sz, cScaleFactors[i]);
		cv::Point lefttop(pos_.x + 1 - cvFloor(float(sample_sz.width) / 2.0), 
							pos_.y + 1 - cvFloor(float(sample_sz.height) / 2.0));
		cv::Point rightbottom(lefttop.x + sample_sz.width, lefttop.y + sample_sz.height);
		cv::Point lefttop_limit(max(lefttop.x, 0), max(lefttop.y, 0));
		cv::Point rightbottom_limit(min(rightbottom.x, im.cols - 1), min(rightbottom.y, im.rows - 1));
		cv::Rect roiRect(lefttop_limit, rightbottom_limit);
		im(roiRect).copyTo(sample);

		cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
			max(rightbottom.x - im.cols + 1, 0), max(rightbottom.y - im.rows + 1, 0));
		if (border != cv::Rect(0, 0, 0, 0)){//如果有越界，在sample基础上进行处理，得到一个边界复制的子图
			cv::copyMakeBorder(sample, sample, border.y, border.height, border.x, border.width, cv::BORDER_REFLECT);
		}

		int interpolation;
		if (scale_model_sz.width > sample_sz.height) interpolation = CV_INTER_LINEAR;
		else interpolation = CV_INTER_AREA;
		cv::resize(sample, sample, scale_model_sz, 0, 0, interpolation);//把图片缩放回scale_model_sz

		cv::Mat tmp = GetScaleFeatures(sample, scale_hann_window.at<float>(i, 0), 4);//提取特征，返回一维向量
		if (i == 0) out = tmp; //first scale
		else {//other scales
			cv::hconcat(out, tmp, out);//往后拼接，成一个矩阵
		}
	}
	return out; //返回加窗后的特征子图
}

/**************************
Function: get a fhog scale sample
@ im_patch : the scale subwindow
@ factor_window : a hann window factor with respect to this scale
@ cell_size: a parameter of fhog
@ return: a vector of mat which is fhog of subwindow
***************************/
cv::Mat DSST::GetScaleFeatures(cv::Mat im_patch, float factor_window, int cell_size){
	vector<cv::Mat> x_vector;
	im_patch.convertTo(im_patch, CV_32FC1);
	x_vector = scale_fhog.extract(im_patch, 2, cell_size);//31 channels = 27 gradiens orientation + 4 textures
	cv::Mat res = cv::Mat::zeros(x_vector.size()*x_vector[0].rows*x_vector[0].cols, 1, CV_32FC1);//a col vector

	int w = x_vector[0].cols;
	int h = x_vector[0].rows;
	int len = x_vector.size();
	for (int k = 0; k < len; k++){//channel
		for (int x = 0; x < w; x++){//col
			for (int y = 0; y < h; y++){ //row
				res.at<float>(k*w*h + x*h + y, 0) = x_vector[k].at<float>(y, x)*factor_window;
			}
		}
	}
	return res;//返回一个列向量，代表某一尺度下图片HOG特征1维形式(加窗后)
}