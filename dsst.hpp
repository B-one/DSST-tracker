/*==================================
Author: B-one
Date  : 2017/5/30
Task  : Define a tracker class
====================================*/

#ifndef _DSST_HPP
#define _DSST_HPP

#include <opencv2\opencv.hpp>
#include <vector>
#include "fhog.hpp"

using namespace std;

class DSST{
public:
	DSST()=default;
	void Init(cv::Mat im, cv::Rect rect);//Init tracker in first frame
	cv::Rect Update(cv::Mat im);//update model and result

private:
//---------function member-----------------
	cv::Mat TranGaussianLabels(float sigma, cv::Size sz);//create 2D gaussian shape labels for translation labels
	cv::Mat CreatTransHann(cv::Size sz);//create 2D hann window
	vector<cv::Mat> GetTransSample(cv::Mat &im);
	vector<cv::Mat> GetTransFeatures(cv::Mat im_patch, int cell_size = 1);//get a fhog feature of im_patch
	void TransPredict(cv::Mat &im);
	void TransLearn(cv::Mat &im);


	cv::Mat ScaleGaussianLabels(float sigma, int n);//create 1D gaussian labels for scale filter
	cv::Mat CreatScaleHann(int n);//create 1D hann window
	cv::Mat GetScaleSample(cv::Mat &im);
	cv::Mat GetScaleFeatures(cv::Mat im_patch, float factor_window, int cell_size = 1 );
	void ScalePredict(cv::Mat &im);
	void ScaleLearn(cv::Mat &im);

	inline cv::Size FloorSizeScale(cv::Size sz, double sf) {//sf:scale factor
		if (sf > 0.9999 && sf < 1.0001)
			return sz;
		return cv::Size(cvFloor(sz.width * sf),
			cvFloor(sz.height * sf));
	}

//--------------Data member-----------------
	cv::Point pos_;//the current center position of target 
	cv::Size init_target_sz;//init target size
	cv::Size current_target_sz;//current target size
	float lambda = 0.01f;//t
	float learning_rate = 0.025f;


  //---------Translate Correlation filter------------
	float padding =0.5f;
	float output_sigma_factor = 1. / 16;
	cv::Size window_sz;//taking padding into account
	cv::Mat trans_yf_;
	cv::Mat trans_hann_window;
	vector<cv::Mat> _hf_num;
	cv::Mat _hf_den;

  //---------Scale Correlation filter------------
	float scale_sigma_factor = 1. / 4;
	int number_of_scales = 33;
	float scale_step = 1.02;
	int scale_model_max_area = 512; 
	float currentScaleFactor = 1.0f;
	float min_scale_factor = 0.0f;
	float max_scale_factor = 0.0f;
	vector<float> scaleFactors; 
	cv::Size scale_model_sz;
	cv::Mat scale_yf_;
	cv::Mat scale_hann_window;
	cv::Mat _sf_num;
	cv::Mat _sf_den;

//---------Feature Extractor ---
	FHoG trans_fhog;
	FHoG scale_fhog;
};

/**************************
Function: extract a real part from a complex matrix
@ complex : a complex matrix
@return : a real part
***************************/
inline cv::Mat GetRealMatrix(cv::Mat &complex){
	vector<cv::Mat> tmp;
	cv::split(complex, tmp);
	return tmp[0];
}

#endif