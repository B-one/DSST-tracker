/*==================================
Author: B-one
Date  : 2017/5/30
Task  : Update appearance model, get a newest result_rect
====================================*/

#include "dsst.hpp"
#include <cmath>

cv::Mat ComplexMul(const cv::Mat &x1, const cv::Mat &x2);
cv::Mat ComplexDivReal(const cv::Mat &x1, const cv::Mat &x2);


/**************************
Function: update the appearance model and result_rect
@ im    : image
@ return: a result_rect
***************************/
cv::Rect DSST::Update(cv::Mat im){

	TransPredict(im);//predict the target position in current frame
	ScalePredict(im);//predict the current scale factor 

	TransLearn(im);//update the model
	ScaleLearn(im);

	current_target_sz = FloorSizeScale(init_target_sz, currentScaleFactor);//update the current target size
	cv::Rect result_rect(pos_.x - cvFloor(float(current_target_sz.width) / 2.0),
		pos_.y - cvFloor(float(current_target_sz.height) / 2.0),
		current_target_sz.width, current_target_sz.height);
	return result_rect;
}


/**************************
Function: predict the target position in current frame
@ im    : image
***************************/
void DSST::TransPredict(cv::Mat &im){
	vector<cv::Mat> trans_xt = GetTransSample(im);
	vector<cv::Mat> trans_xtf(trans_xt.size());
	cv::Mat trans_complex_response;

	for (int i = 0; i < trans_xt.size(); i++){//遍历所有通道,这个过程接近0.02s
		cv::dft(trans_xt[i], trans_xtf[i], CV_HAL_DFT_COMPLEX_OUTPUT);//对每通道的特征子图进行FFT操作
		cv::Mat tmp = ComplexMul(_hf_num[i], trans_xtf[i]);//每个频道复数点乘
		if (i == 0) trans_complex_response = tmp;
		else trans_complex_response += tmp;
	}


	trans_complex_response = ComplexDivReal(trans_complex_response, (_hf_den + lambda));

	cv::Mat response;//response of translation 
	cv::idft(trans_complex_response, response, CV_HAL_DFT_REAL_OUTPUT | CV_HAL_DFT_SCALE);//

	cv::Point maxLoc;
	cv::minMaxLoc(response, NULL, NULL, NULL, &maxLoc);//get max response location

	pos_.x = pos_.x + round((-float(window_sz.width) / 2.0 + maxLoc.x)*currentScaleFactor)+1;//得到新坐标，后一项代表偏移量
	pos_.y = pos_.y + round((-float(window_sz.height) / 2.0 + maxLoc.y)*currentScaleFactor)+1;
}

/**************************
Function: predict the current scale factor
@ im    : image
***************************/
void DSST::ScalePredict(cv::Mat &im){
	cv::Mat scale_xs = GetScaleSample(im);
	cv::Mat scale_xsf = cv::Mat(cv::Size(scale_xs.cols, scale_xs.rows), CV_32F, float(0));
	cv::Mat scale_complex_response;

	cv::dft(scale_xs, scale_xsf, CV_HAL_DFT_COMPLEX_OUTPUT | CV_HAL_DFT_ROWS);
	scale_complex_response = ComplexMul(_sf_num, scale_xsf);
	cv::reduce(scale_complex_response, scale_complex_response, 0, CV_REDUCE_SUM);
	scale_complex_response = ComplexDivReal(scale_complex_response, (_sf_den + lambda));

	cv::Mat scale_response;
	cv::idft(scale_complex_response, scale_response, CV_HAL_DFT_REAL_OUTPUT | CV_HAL_DFT_SCALE);

	cv::Point scale_maxLoc;
	cv::minMaxLoc(scale_response, NULL, NULL, NULL, &scale_maxLoc);
	int recovered_scale = scale_maxLoc.x;

	currentScaleFactor = currentScaleFactor * scaleFactors[recovered_scale];//update the scale factor
	if (currentScaleFactor < min_scale_factor) currentScaleFactor = min_scale_factor;
	else if (currentScaleFactor>max_scale_factor) currentScaleFactor = max_scale_factor;
}


/**************************
Function: complex matrix .* complex matrix
@ x1 : matrix_1
@ x2 : matrix_2
@ return: a complex matrix result
***************************/
cv::Mat ComplexMul(const cv::Mat &x1, const cv::Mat &x2){
	vector<cv::Mat> x1_vec;
	vector<cv::Mat> x2_vec;
	cv::split(x1, x1_vec);
	cv::split(x2, x2_vec);

	vector<cv::Mat> tmp_result(2);
	tmp_result[0] = x1_vec[0].mul(x2_vec[0]) - x1_vec[1].mul(x2_vec[1]);
	tmp_result[1] = x1_vec[0].mul(x2_vec[1]) + x1_vec[1].mul(x2_vec[0]);

	cv::Mat result;
	cv::merge(tmp_result, result);
	return result;
}

/**************************
Function: complex_matrix1 ./ real_matrix2
@ x1 : matrix_1
@ x2 : matrix_2
@ return: a complex matrix result
***************************/
cv::Mat ComplexDivReal(const cv::Mat &x1, const cv::Mat &x2){
	vector<cv::Mat> x1_vec;
	cv::split(x1, x1_vec);
	vector<cv::Mat> tmp_result(2);

	tmp_result[0] = x1_vec[0] / x2;
	tmp_result[1] = x1_vec[1] / x2;

	cv::Mat result;
	cv::merge(tmp_result, result);
	return result;
}
