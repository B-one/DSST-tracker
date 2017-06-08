/*==================================
Author: B-one
Date  : 2017/5/30
Task  : Complete a tracker class implemention
====================================*/

#include "dsst.hpp"
#include <cmath>

extern int g_trackState;

//实现复数矩阵点乘
cv::Mat ComplexMul(const cv::Mat &x1, const cv::Mat &x2){
	vector<cv::Mat> x1_vec;
	vector<cv::Mat> x2_vec;
	cv::split(x1, x1_vec);//把原本的复数矩阵，拆成两个矩阵(一实一虚)。
	cv::split(x2, x2_vec);

	vector<cv::Mat> tmp_result(2);//建立两个元素
	tmp_result[0] = x1_vec[0].mul(x2_vec[0]) - x1_vec[1].mul(x2_vec[1]);
	tmp_result[1] = x1_vec[0].mul(x2_vec[1]) + x1_vec[1].mul(x2_vec[0]);

	cv::Mat result;
	cv::merge(tmp_result, result);//把计算完的结果重新合并成复数矩阵
	return result;
}

//实现复数矩阵点除; x2为实数矩阵
cv::Mat ComplexDivReal(const cv::Mat &x1, const cv::Mat &x2){
	vector<cv::Mat> x1_vec;
	cv::split(x1, x1_vec);//把原本的复数矩阵，拆成两个矩阵(一实一虚)。

	vector<cv::Mat> tmp_result(2);//建立两个元素

	tmp_result[0] = x1_vec[0]/x2;//矩阵的除法、加法都有直接重定义了
	tmp_result[1] = x1_vec[1]/x2;

	cv::Mat result;
	cv::merge(tmp_result, result);//把计算完的结果重新合并成复数矩阵
	return result;
}

void DSST::Init(cv::Mat image, cv::Rect rect){
	center_pos = cv::Point(rect.x + cvFloor((float(rect.width) / 2.)),
		rect.y + cvFloor((float(rect.height)) / 2.) ); //得到初始帧的中心坐标
	current_target_sz = rect.size();//得到初始帧的目标尺寸
	init_target_sz = current_target_sz;
	trans_subwindows_sz = FloorSizeScale(current_target_sz,1+padding);//得到标准的搜索区域，常
	
	
	//----------建立位置滤波器的高斯理想输出和2D hann窗--------------
	float output_sigma = sqrt(init_target_sz.area())*output_sigma_factor;

	cv::dft(TranGaussianLabels(output_sigma, trans_subwindows_sz),trans_yf_, 
			CV_HAL_DFT_COMPLEX_OUTPUT);//返回尺寸：原目标padding过  ——这里有转置很耗时
	//cout << trans_yf_.rows << " " << trans_yf_.cols << endl;
	//cout << trans_yf_.row(1) << endl;

	trans_hann_window = CreatTransHann(trans_yf_.size());//返回尺寸：原目标padding过
	//cout << trans_hann_window.rows << " " << trans_hann_window.cols << endl;
	//cout << trans_hann_window.row(0) << endl;

	//----------建立尺度滤波器的高斯理想输出和1D hann窗--------------
	float scale_sigma = float(number_of_scales) / sqrt(33) *scale_sigma_factor;
	cv::dft(ScaleGaussianLabels(scale_sigma, number_of_scales), scale_yf_,
			CV_HAL_DFT_COMPLEX_OUTPUT); //返回的是行向量

	//cout << scale_yf_ << endl;
	scale_hann_window = CreatScaleHann(scale_yf_.size().width);//返回列向量
	//cout << scale_hann_window << endl;
	for (int i = 0; i < number_of_scales; i++){//生成尺度因子序列
		float tmp = pow(scale_step, ceil(float(number_of_scales) / 2.) - i - 1);
		scaleFactors.push_back(tmp);
	}
	//for (auto i : scaleFactors) cout << i << " ";
	//cout << endl;

	//得到在预估目标尺度变化时的一致尺寸scale_model_sz，面积不能超过512
	float scale_model_factor = 1;
	if (init_target_sz.area()>scale_model_max_area){
		scale_model_factor = sqrt(float(scale_model_max_area) / (init_target_sz.area()));
	}

	//基于第一帧的目标尺寸，把尺度的尺寸缩放到面积不大于512
	scale_model_sz = FloorSizeScale(init_target_sz, scale_model_factor);

	currentScaleFactor = 1.0f;//当前的尺寸变化，相对于初始帧，好像是

	//------------最大尺度因子和最小尺度因子---
	float minscale_tmp = max(float(5) / trans_subwindows_sz.height, float(5) / trans_subwindows_sz.width);
	float maxscale_tmp = min(float(image.size().height) / init_target_sz.height,
							float(image.size().width) / init_target_sz.width);
	min_scale_factor = pow(scale_step, ceil(log(minscale_tmp) / log(scale_step)));
	max_scale_factor = pow(scale_step, floor(log(maxscale_tmp) / log(scale_step)));
	
	
	//--------得到位移相关滤波器(translation的学习)----------
	TransLearn(image);
	
	//--------得到尺度相关滤波器(translation的学习)----------
	ScaleLearn(image);

}

cv::Rect DSST::Update(cv::Mat image){
//---------------------位置相关滤波器的测量------------

	vector<cv::Mat> trans_xt = GetTransSample(image, center_pos, trans_subwindows_sz, currentScaleFactor);
	vector<cv::Mat> trans_xtf(trans_xt.size());
	cv::Mat trans_complex_response;//位置响应图(复数阶段)

	for (int i = 0; i < trans_xt.size(); i++){//遍历所有通道
		cv::dft(trans_xt[i], trans_xtf[i], CV_HAL_DFT_COMPLEX_OUTPUT);//对每通道的特征子图进行FFT操作
		cv::Mat tmp	=ComplexMul(_hf_num[i], trans_xtf[i]);//每个频道复数点乘
		if (i == 0) trans_complex_response = tmp;
		else trans_complex_response += tmp;
	}
	/*cout << trans_complex_response.colRange(4, 6) << endl;*/ //未进行除法
	trans_complex_response= ComplexDivReal(trans_complex_response, (_hf_den + lambda));
	//cout << trans_complex_response.colRange(4, 6) << endl;//进行了复数除法

	cv::Mat response;//位置响应图
	cv::idft(trans_complex_response, response, CV_HAL_DFT_REAL_OUTPUT|CV_HAL_DFT_SCALE);//输出实矩阵，如果src是具有共轭对称性，为什么要加DFT_SCALE
	//cout << response.colRange(4, 6) << endl; //处理后的结果图


	cv::Point maxLoc;//最大值位置
	cv::minMaxLoc(response, NULL, NULL, NULL, &maxLoc);//找到最大响应点

	center_pos.x = center_pos.x + round((-float(trans_subwindows_sz.width)/2.0 + maxLoc.x)*currentScaleFactor)+1;//得到新坐标，后一项代表偏移量
	center_pos.y = center_pos.y + round((-float(trans_subwindows_sz.height) / 2.0 + maxLoc.y)*currentScaleFactor)+1;

	
//-------------尺度相关滤波器的测量-----------------------
	cv::Mat scale_xs = GetScaleSample(image);
	cv::Mat tmp_scale_response;

	cv::dft(scale_xs, scale_xsf, CV_HAL_DFT_COMPLEX_OUTPUT | CV_HAL_DFT_ROWS);//进行一维FFT
	tmp_scale_response = ComplexMul(_sf_num,scale_xsf);
	cv::reduce(tmp_scale_response, tmp_scale_response, 0, CV_REDUCE_SUM);//叠加起来
	tmp_scale_response = ComplexDivReal(tmp_scale_response, (_sf_den + lambda));

	cv::Mat scale_response;
	cv::idft(tmp_scale_response, scale_response, CV_HAL_DFT_REAL_OUTPUT | CV_HAL_DFT_SCALE);//输出实矩阵，如果src是具有共轭对称性
	//cout << scale_response << endl;

	cv::Point scale_maxLoc;
	cv::minMaxLoc(scale_response, NULL, NULL, NULL, &scale_maxLoc);
	int recovered_scale = scale_maxLoc.x;
	//更新尺度因子
	currentScaleFactor = currentScaleFactor * scaleFactors[recovered_scale];
	if (currentScaleFactor < min_scale_factor) currentScaleFactor = min_scale_factor;
	else if (currentScaleFactor>max_scale_factor) currentScaleFactor = max_scale_factor;

	TransLearn(image);//进行模型的更新
	ScaleLearn(image);
	
	current_target_sz = FloorSizeScale(init_target_sz,currentScaleFactor);//更新当前目标尺寸
	cv::Rect result_rect(center_pos.x - cvFloor(float(current_target_sz.width) / 2.0), center_pos.y - cvFloor(float(current_target_sz.height) / 2.0),
						current_target_sz.width, current_target_sz.height);
	return result_rect;
}


void DSST::TransLearn(cv::Mat &image){
	//用于保存加窗后的特征子图和进行傅利叶变换后的子图
	vector<cv::Mat> trans_xl = GetTransSample(image, center_pos, trans_subwindows_sz, currentScaleFactor); //这个耗时为0.02，需要改进
	vector<cv::Mat> trans_xlf(trans_xl.size());

	//用于保存当前求得的滤波器组
	vector<cv::Mat> new_hf_num(trans_xl.size()); //位置滤波器当前得到的分子
	cv::Mat new_hf_den;//位置滤波器当前得到的分母

	for (int i = 0; i < trans_xl.size(); i++){ //这段for循环花了0.01秒
		cv::dft(trans_xl[i], trans_xlf[i], CV_HAL_DFT_COMPLEX_OUTPUT);//对每通道的特征子图进行FFT操作
		cv::mulSpectrums(trans_yf_, trans_xlf[i], new_hf_num[i], 0, true);//将每个通道进行共轭相乘
		cv::Mat tmp_den;
		cv::mulSpectrums(trans_xlf[i], trans_xlf[i], tmp_den, 0, true);//通道共轭相乘
		if (i == 0) new_hf_den = tmp_den;
		else new_hf_den += tmp_den;
	} 
	//cout << new_hf_num[3].col(0) << endl;

	//这里拆了对吗？？
	vector<cv::Mat> tmp_hf_den;
	cv::split(new_hf_den, tmp_hf_den);//把含有复数的矩阵拆开
	new_hf_den = tmp_hf_den[0];//得到实数通道
	/*cout << new_hf_den.row(0) << endl;*/

	if (g_trackState == -1){//初始帧
		_hf_num = new_hf_num;
		_hf_den = new_hf_den;//由于是第一帧，所以直接相等，不用再处理
	}
	else if (g_trackState == 1){//往后帧
		for (int i = 0; i < _hf_num.size(); i++){
			_hf_num[i] = (1 - learning_rate)*_hf_num[i] + learning_rate*new_hf_num[i];
		}
		_hf_den = (1 - learning_rate)*_hf_den + learning_rate*new_hf_den;
		//cv::addWeighted(_hf_num, (1 - learning_rate), new_hf_num, learning_rate, 0, _hf_num);
		//cv::addWeighted(_hf_den, (1 - learning_rate), new_hf_den, learning_rate, 0, _hf_den);
	}
	else{//例外
		cout << "something of frame_n is wrong!" << endl;
	}

}

void DSST::ScaleLearn(cv::Mat &image){
	//---------得到尺度相关滤波器的特征子块(加过窗)-----
	//int64 tio = cv::getTickCount();
	cv::Mat scale_xs = GetScaleSample(image);
	//int64 tie = cv::getTickCount() - tio;
	//double time = double(tie) / double(cv::getTickFrequency());

	cv::Mat new_sf_num; //用于保存新模型
	cv::Mat new_sf_den;

	if (g_trackState == -1){
		//建立一个744*33的矩阵，初始化为0.仅在第一帧初始化
		scale_xsf = cv::Mat(cv::Size(scale_xs.cols, scale_xs.rows), CV_32F, float(0));
		scale_yf_ = cv::repeat(scale_yf_, scale_xs.rows, 1);//将yf按垂直复制744次。
	}
	cv::dft(scale_xs, scale_xsf, CV_HAL_DFT_COMPLEX_OUTPUT | CV_HAL_DFT_ROWS);//进行一维FFT
	cv::mulSpectrums(scale_yf_, scale_xsf,new_sf_num, 0, true);//得到最新的模型
	cv::mulSpectrums(scale_xsf, scale_xsf, new_sf_den, 0, true);
	cv::reduce(new_sf_den, new_sf_den, 0, CV_REDUCE_SUM);//累加成一个行

	vector<cv::Mat> tmp_sf_den;
	cv::split(new_sf_den, tmp_sf_den);//把含有复数的矩阵拆开
	new_sf_den = tmp_sf_den[0];//得到实数通道

	if (g_trackState == -1){//初始帧
		_sf_num = new_sf_num;
		_sf_den = new_sf_den;//由于是第一帧，所以直接相等，不用再处理
	}
	else if (g_trackState == 1){//往后帧
		cv::addWeighted(_sf_num, (1 - learning_rate), new_sf_num, learning_rate, 0,_sf_num);
		cv::addWeighted(_sf_den, (1 - learning_rate), new_sf_den, learning_rate, 0, _sf_den);
	}
	else{//例外
		cout << "something of frame_n is wrong!" << endl;
	}
	//cout << _sf_num.colRange(4,6) << endl;
	//cout << _sf_den << endl;
}

//这个可以放到基类，CF
cv::Mat DSST::CreateGaussian1(int n, float sigma) {
	cv::Mat gaussian1(n, 1, CV_32F);//对应的是Mat_<float>,生成一个列向量
	
	float sigma2 = sigma*sigma;
	float x = 0;
	float tmp = 0;

	for (int i = 0; i < n; i++){
		x = i - floor(float(n) / 2.) + 1;//
		tmp = exp(-0.5*((x*x) / sigma2));
		gaussian1.at<float>(i,0) = tmp;
	}

	return gaussian1;
}

/**************************
Function: get a desried output of translation filter
@sigma : standard deviation for translation filter output
@ sz    : the dimention of translation filter
@return: a gaussian shape output
***************************/
cv::Mat DSST::TranGaussianLabels(float sigma, cv::Size sz){
	cv::Mat a = CreateGaussian1(sz.height, sigma);
	cv::Mat b = CreateGaussian1(sz.width, sigma);
	cv::Mat labels = a*b.t(); //不中心化
	return labels;
}

/**************************
Function: get a desried output of scale filter
@sigma : standard deviation for scale filter output
@ n    : the dimention of scale filter
@return: a gaussian shape output
***************************/
cv::Mat DSST::ScaleGaussianLabels(float sigma, int n) {
	cv::Mat labels(1, n, CV_32F);
	
	float sigma2 = sigma*sigma;
	int x = 0;
	float tmp = 0;
	for (int i = 0; i < n; i++){
		x = i - ceil(float(n) / 2.) + 1;
		tmp = exp(-0.5* (x*x) / sigma2);
		labels.at<float>(0,i) = tmp;
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
	cv::Mat tmp2(sz.width, 1, CV_32F);

	for (int i = 0; i < sz.height; i++){
		tmp1.at<float>(i, 0) = 0.5*(1 - cos(2 * CV_PI*i / (sz.height - 1)));
	}
	for (int i = 0; i < sz.width; i++){
		tmp2.at<float>(i, 0) = 0.5*(1 - cos(2 * CV_PI*i / (sz.width - 1)));
	}
	return tmp1*tmp2.t();
}

/**************************
Function: creat a hann windows for scale filter
@ sz : the dimention of scale input
@return: 1D Hann in the same size, (col vector)
***************************/
cv::Mat DSST::CreatScaleHann(int n){
	cv::Mat tmp1(n,1, CV_32F);
	for (int i = 0; i < n; i++){
		tmp1.at<float>(i, 0) = 0.5*(1 - cos(2 * CV_PI*i / (n - 1)));
	}
	return tmp1;
}

/**************************
Function: get a sample of the frame
@ im : image
@ center_pos : the center coordinate of target in image
@ model_sz : the size of sample in first frame
@ cScaleFactor : = sample_sz / model_sz
@return: a vector of mat which has been add window and take fhog
***************************/
vector<cv::Mat> DSST::GetTransSample(cv::Mat &im, cv::Point center_pos, cv::Size model_sz, float cScaleFactor){
	cv::Mat sample;//建立图片，存储patch

	cv::Size sample_sz = FloorSizeScale(model_sz, cScaleFactor);//得到ROI的尺寸
	//保证patch尺寸最小为2*2
	if (sample_sz.width <= 1) sample_sz.width = 2;
	if (sample_sz.height <= 1) sample_sz.height = 2;

	//得到左上角坐标和右下角坐标
	cv::Point lefttop(center_pos.x +1- cvFloor(float(sample_sz.width) / 2.0), center_pos.y +1- cvFloor(float(sample_sz.height) / 2.0));//从0开始计，
	cv::Point rightbottom(lefttop.x + sample_sz.width, lefttop.y + sample_sz.height);//右下角坐标,不减1了，没必要？
	cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
		max(rightbottom.x - im.cols + 1, 0), max(rightbottom.y - im.rows + 1, 0));//得到越出多少范围
	//把子框控制在图像区域内,即得到范围被限制过的左上角和右下角
	cv::Point lefttop_limit(max(lefttop.x, 0), max(lefttop.y, 0));//得到不越界的左上角
	cv::Point rightbottom_limit(min(rightbottom.x, im.cols - 1), min(rightbottom.y, im.rows - 1));//得到不越界的右下角

	cv::Rect roiRect(lefttop_limit, rightbottom_limit);//得到待裁的矩形
	im(roiRect).copyTo(sample);//im(roiRect)这是一个构造函数，创建了一个中间对象

	if (border != cv::Rect(0, 0, 0, 0)){//如果有越界，在sample基础上进行处理，得到一个边界复制的子图
		cv::copyMakeBorder(sample, sample, border.y, border.height, border.x, border.width, cv::BORDER_REFLECT);
	}
	int interpolation;
	if (model_sz.width > sample_sz.height){//????
		interpolation = CV_INTER_LINEAR;
	}else{
		interpolation = CV_INTER_AREA;
	}
	cv::resize(sample, sample, model_sz, 0, 0, interpolation);//把图片缩放回model_sz

	vector<cv::Mat> out = GetTransFeatures(sample); //0.018s
	//cout << out[4].row(68) << endl;
	return out;//返回子图像
}

/**************************
Function: get a fhog translation sample 
@ im_patch : the search window
@ cell_size: a parameter of fhog
@return: a vector of mat which is fhog of subwindow
***************************/
std::vector<cv::Mat> DSST::GetTransFeatures(cv::Mat im_patch, int cell_size){
	cv::Mat x;
	vector<cv::Mat> x_vector(28);//一个灰度特征，外加hog的27个通道特征
	vector<cv::Mat> tmp_vector;

	if (im_patch.channels() == 3){//如果是彩色图像
		cv::cvtColor(im_patch, im_patch, CV_BGR2GRAY);//换成灰度图像
	}

	im_patch.convertTo(im_patch, CV_32FC1);//转成float型
	/*cout << im_patch.row(33) << endl;*/
	x_vector[0] = im_patch / 255.0 - 0.5;//相当于提取一个灰度特征

	//cell_size的大小影响了计算速度
	tmp_vector = trans_fhog.extract(im_patch, 2, cell_size);//这个有31个通道，前27个梯度角直方图、4个纹理

	for (int i = 0; i < 27; i++){//把计算出来的hog特征提取,不要4个纹理特征通道和全0通道
		x_vector[i + 1] = tmp_vector[i];
	}
	for (int i = 0; i < 28; i++){//进行每个通道的加窗处理
		x_vector[i] = x_vector[i].mul(trans_hann_window);
	}
	return x_vector;
}

/**************************
Function: get a fhog scale sample
@ im : image
@return:  a vector of mat which has been add window and take fhog
***************************/
cv::Mat DSST::GetScaleSample(cv::Mat &im){
	vector<float> cScaleFactors;//当前尺度下的尺度序列
	for (int i = 0; i < number_of_scales; i++){//得到最新的尺度序列
		cScaleFactors.push_back(scaleFactors[i] * currentScaleFactor);
	}

	cv::Mat out;//保存结果
	//int64 tio = cv::getTickCount();
	for (int i = 0; i < number_of_scales; i++){//遍历所有尺度
		cv::Mat sample;
		cv::Size sample_sz = FloorSizeScale(init_target_sz, cScaleFactors[i]);
		cv::Point lefttop(center_pos.x+1 - cvFloor(float(sample_sz.width) / 2.0), center_pos.y+1 - cvFloor(float(sample_sz.height) / 2.0));
		cv::Point rightbottom(lefttop.x + sample_sz.width, lefttop.y + sample_sz.height);//右下角坐标,这里要减1,因为是从0开始计
		cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
			max(rightbottom.x - im.cols + 1, 0), max(rightbottom.y - im.rows + 1, 0));//得到越出多少范围
		//把子框控制在图像区域内,即得到范围被限制过的左上角和右下角
		cv::Point lefttop_limit(max(lefttop.x, 0), max(lefttop.y, 0));//得到不越界的左上角
		cv::Point rightbottom_limit(min(rightbottom.x, im.cols - 1), min(rightbottom.y, im.rows - 1));//得到不越界的右下角
		cv::Rect roiRect(lefttop_limit, rightbottom_limit);//得到待裁的矩形
		im(roiRect).copyTo(sample);//im(roiRect)这是一个构造函数，创建了一个中间对象
		if (border != cv::Rect(0, 0, 0, 0)){//如果有越界，在sample基础上进行处理，得到一个边界复制的子图
			cv::copyMakeBorder(sample, sample, border.y, border.height, border.x, border.width, cv::BORDER_REFLECT);
		}
		int interpolation;
		if (scale_model_sz.width > sample_sz.height){//????
			interpolation = CV_INTER_LINEAR;
		}
		else{
			interpolation = CV_INTER_AREA;
		}
		cv::resize(sample, sample, scale_model_sz, 0, 0, interpolation);//把图片缩放回scale_model_sz
		cv::Mat tmp = GetScaleFeatures(sample,scale_hann_window.at<float>(i,0),4);//提取特征，返回一维向量
		if (i == 0) out = tmp; //第一个尺度进行拷贝赋值
		else {//往后尺度
			cv::hconcat(out, tmp, out);//往后拼接，成一个矩阵
		}
	}
	//int64 tie = cv::getTickCount() - tio;
	//double time = double(tie) / double(cv::getTickFrequency());
	//cout << out.colRange(4,7) << endl; //这个输出仅有接近200行
	return out; //返回加窗后的特征子图
}

/**************************
Function: get a fhog scale sample
@ im_patch : the scale subwindow
@ cell_size: a parameter of fhog
@return: a vector of mat which is fhog of subwindow
***************************/
cv::Mat DSST::GetScaleFeatures(cv::Mat im_patch, float factor_window, int cell_size){
	vector<cv::Mat> x_vector;
	im_patch.convertTo(im_patch, CV_32FC1);//转成float型
	x_vector = scale_fhog.extract(im_patch, 2, cell_size);//这个有31个通道，前27个梯度角直方图、4个纹理
	cv::Mat res = cv::Mat::zeros(x_vector.size()*x_vector[0].rows*x_vector[0].cols,1,CV_32FC1);//返回一个列向量(把整个fhog特征图拉直)

	int w = x_vector[0].cols;//得到每个通道图片的宽
	int h = x_vector[0].rows;//得到每个通道图片的长
	int len = x_vector.size();//得到通道数
	for (int k = 0; k < len; k++){//表示第几个通道
		for (int x = 0; x < w; x++){//表示第几列
			for (int y = 0; y < h; y++){ //表示第几行
				res.at<float>(k*w*h + x*h + y, 0) = x_vector[k].at<float>(y, x)*factor_window;
			}	
		}
	}
	return res;//返回一个列向量，代表某一尺度下图片HOG特征1维形式(加窗后)
}
