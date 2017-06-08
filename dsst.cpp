/*==================================
Author: B-one
Date  : 2017/5/30
Task  : Complete a tracker class implemention
====================================*/

#include "dsst.hpp"
#include <cmath>

extern int g_trackState;

//ʵ�ָ���������
cv::Mat ComplexMul(const cv::Mat &x1, const cv::Mat &x2){
	vector<cv::Mat> x1_vec;
	vector<cv::Mat> x2_vec;
	cv::split(x1, x1_vec);//��ԭ���ĸ������󣬲����������(һʵһ��)��
	cv::split(x2, x2_vec);

	vector<cv::Mat> tmp_result(2);//��������Ԫ��
	tmp_result[0] = x1_vec[0].mul(x2_vec[0]) - x1_vec[1].mul(x2_vec[1]);
	tmp_result[1] = x1_vec[0].mul(x2_vec[1]) + x1_vec[1].mul(x2_vec[0]);

	cv::Mat result;
	cv::merge(tmp_result, result);//�Ѽ�����Ľ�����ºϲ��ɸ�������
	return result;
}

//ʵ�ָ���������; x2Ϊʵ������
cv::Mat ComplexDivReal(const cv::Mat &x1, const cv::Mat &x2){
	vector<cv::Mat> x1_vec;
	cv::split(x1, x1_vec);//��ԭ���ĸ������󣬲����������(һʵһ��)��

	vector<cv::Mat> tmp_result(2);//��������Ԫ��

	tmp_result[0] = x1_vec[0]/x2;//����ĳ������ӷ�����ֱ���ض�����
	tmp_result[1] = x1_vec[1]/x2;

	cv::Mat result;
	cv::merge(tmp_result, result);//�Ѽ�����Ľ�����ºϲ��ɸ�������
	return result;
}

void DSST::Init(cv::Mat image, cv::Rect rect){
	center_pos = cv::Point(rect.x + cvFloor((float(rect.width) / 2.)),
		rect.y + cvFloor((float(rect.height)) / 2.) ); //�õ���ʼ֡����������
	current_target_sz = rect.size();//�õ���ʼ֡��Ŀ��ߴ�
	init_target_sz = current_target_sz;
	trans_subwindows_sz = FloorSizeScale(current_target_sz,1+padding);//�õ���׼���������򣬳�
	
	
	//----------����λ���˲����ĸ�˹���������2D hann��--------------
	float output_sigma = sqrt(init_target_sz.area())*output_sigma_factor;

	cv::dft(TranGaussianLabels(output_sigma, trans_subwindows_sz),trans_yf_, 
			CV_HAL_DFT_COMPLEX_OUTPUT);//���سߴ磺ԭĿ��padding��  ����������ת�úܺ�ʱ
	//cout << trans_yf_.rows << " " << trans_yf_.cols << endl;
	//cout << trans_yf_.row(1) << endl;

	trans_hann_window = CreatTransHann(trans_yf_.size());//���سߴ磺ԭĿ��padding��
	//cout << trans_hann_window.rows << " " << trans_hann_window.cols << endl;
	//cout << trans_hann_window.row(0) << endl;

	//----------�����߶��˲����ĸ�˹���������1D hann��--------------
	float scale_sigma = float(number_of_scales) / sqrt(33) *scale_sigma_factor;
	cv::dft(ScaleGaussianLabels(scale_sigma, number_of_scales), scale_yf_,
			CV_HAL_DFT_COMPLEX_OUTPUT); //���ص���������

	//cout << scale_yf_ << endl;
	scale_hann_window = CreatScaleHann(scale_yf_.size().width);//����������
	//cout << scale_hann_window << endl;
	for (int i = 0; i < number_of_scales; i++){//���ɳ߶���������
		float tmp = pow(scale_step, ceil(float(number_of_scales) / 2.) - i - 1);
		scaleFactors.push_back(tmp);
	}
	//for (auto i : scaleFactors) cout << i << " ";
	//cout << endl;

	//�õ���Ԥ��Ŀ��߶ȱ仯ʱ��һ�³ߴ�scale_model_sz��������ܳ���512
	float scale_model_factor = 1;
	if (init_target_sz.area()>scale_model_max_area){
		scale_model_factor = sqrt(float(scale_model_max_area) / (init_target_sz.area()));
	}

	//���ڵ�һ֡��Ŀ��ߴ磬�ѳ߶ȵĳߴ����ŵ����������512
	scale_model_sz = FloorSizeScale(init_target_sz, scale_model_factor);

	currentScaleFactor = 1.0f;//��ǰ�ĳߴ�仯������ڳ�ʼ֡��������

	//------------���߶����Ӻ���С�߶�����---
	float minscale_tmp = max(float(5) / trans_subwindows_sz.height, float(5) / trans_subwindows_sz.width);
	float maxscale_tmp = min(float(image.size().height) / init_target_sz.height,
							float(image.size().width) / init_target_sz.width);
	min_scale_factor = pow(scale_step, ceil(log(minscale_tmp) / log(scale_step)));
	max_scale_factor = pow(scale_step, floor(log(maxscale_tmp) / log(scale_step)));
	
	
	//--------�õ�λ������˲���(translation��ѧϰ)----------
	TransLearn(image);
	
	//--------�õ��߶�����˲���(translation��ѧϰ)----------
	ScaleLearn(image);

}

cv::Rect DSST::Update(cv::Mat image){
//---------------------λ������˲����Ĳ���------------

	vector<cv::Mat> trans_xt = GetTransSample(image, center_pos, trans_subwindows_sz, currentScaleFactor);
	vector<cv::Mat> trans_xtf(trans_xt.size());
	cv::Mat trans_complex_response;//λ����Ӧͼ(�����׶�)

	for (int i = 0; i < trans_xt.size(); i++){//��������ͨ��
		cv::dft(trans_xt[i], trans_xtf[i], CV_HAL_DFT_COMPLEX_OUTPUT);//��ÿͨ����������ͼ����FFT����
		cv::Mat tmp	=ComplexMul(_hf_num[i], trans_xtf[i]);//ÿ��Ƶ���������
		if (i == 0) trans_complex_response = tmp;
		else trans_complex_response += tmp;
	}
	/*cout << trans_complex_response.colRange(4, 6) << endl;*/ //δ���г���
	trans_complex_response= ComplexDivReal(trans_complex_response, (_hf_den + lambda));
	//cout << trans_complex_response.colRange(4, 6) << endl;//�����˸�������

	cv::Mat response;//λ����Ӧͼ
	cv::idft(trans_complex_response, response, CV_HAL_DFT_REAL_OUTPUT|CV_HAL_DFT_SCALE);//���ʵ�������src�Ǿ��й���Գ��ԣ�ΪʲôҪ��DFT_SCALE
	//cout << response.colRange(4, 6) << endl; //�����Ľ��ͼ


	cv::Point maxLoc;//���ֵλ��
	cv::minMaxLoc(response, NULL, NULL, NULL, &maxLoc);//�ҵ������Ӧ��

	center_pos.x = center_pos.x + round((-float(trans_subwindows_sz.width)/2.0 + maxLoc.x)*currentScaleFactor)+1;//�õ������꣬��һ�����ƫ����
	center_pos.y = center_pos.y + round((-float(trans_subwindows_sz.height) / 2.0 + maxLoc.y)*currentScaleFactor)+1;

	
//-------------�߶�����˲����Ĳ���-----------------------
	cv::Mat scale_xs = GetScaleSample(image);
	cv::Mat tmp_scale_response;

	cv::dft(scale_xs, scale_xsf, CV_HAL_DFT_COMPLEX_OUTPUT | CV_HAL_DFT_ROWS);//����һάFFT
	tmp_scale_response = ComplexMul(_sf_num,scale_xsf);
	cv::reduce(tmp_scale_response, tmp_scale_response, 0, CV_REDUCE_SUM);//��������
	tmp_scale_response = ComplexDivReal(tmp_scale_response, (_sf_den + lambda));

	cv::Mat scale_response;
	cv::idft(tmp_scale_response, scale_response, CV_HAL_DFT_REAL_OUTPUT | CV_HAL_DFT_SCALE);//���ʵ�������src�Ǿ��й���Գ���
	//cout << scale_response << endl;

	cv::Point scale_maxLoc;
	cv::minMaxLoc(scale_response, NULL, NULL, NULL, &scale_maxLoc);
	int recovered_scale = scale_maxLoc.x;
	//���³߶�����
	currentScaleFactor = currentScaleFactor * scaleFactors[recovered_scale];
	if (currentScaleFactor < min_scale_factor) currentScaleFactor = min_scale_factor;
	else if (currentScaleFactor>max_scale_factor) currentScaleFactor = max_scale_factor;

	TransLearn(image);//����ģ�͵ĸ���
	ScaleLearn(image);
	
	current_target_sz = FloorSizeScale(init_target_sz,currentScaleFactor);//���µ�ǰĿ��ߴ�
	cv::Rect result_rect(center_pos.x - cvFloor(float(current_target_sz.width) / 2.0), center_pos.y - cvFloor(float(current_target_sz.height) / 2.0),
						current_target_sz.width, current_target_sz.height);
	return result_rect;
}


void DSST::TransLearn(cv::Mat &image){
	//���ڱ���Ӵ����������ͼ�ͽ��и���Ҷ�任�����ͼ
	vector<cv::Mat> trans_xl = GetTransSample(image, center_pos, trans_subwindows_sz, currentScaleFactor); //�����ʱΪ0.02����Ҫ�Ľ�
	vector<cv::Mat> trans_xlf(trans_xl.size());

	//���ڱ��浱ǰ��õ��˲�����
	vector<cv::Mat> new_hf_num(trans_xl.size()); //λ���˲�����ǰ�õ��ķ���
	cv::Mat new_hf_den;//λ���˲�����ǰ�õ��ķ�ĸ

	for (int i = 0; i < trans_xl.size(); i++){ //���forѭ������0.01��
		cv::dft(trans_xl[i], trans_xlf[i], CV_HAL_DFT_COMPLEX_OUTPUT);//��ÿͨ����������ͼ����FFT����
		cv::mulSpectrums(trans_yf_, trans_xlf[i], new_hf_num[i], 0, true);//��ÿ��ͨ�����й������
		cv::Mat tmp_den;
		cv::mulSpectrums(trans_xlf[i], trans_xlf[i], tmp_den, 0, true);//ͨ���������
		if (i == 0) new_hf_den = tmp_den;
		else new_hf_den += tmp_den;
	} 
	//cout << new_hf_num[3].col(0) << endl;

	//������˶��𣿣�
	vector<cv::Mat> tmp_hf_den;
	cv::split(new_hf_den, tmp_hf_den);//�Ѻ��и����ľ����
	new_hf_den = tmp_hf_den[0];//�õ�ʵ��ͨ��
	/*cout << new_hf_den.row(0) << endl;*/

	if (g_trackState == -1){//��ʼ֡
		_hf_num = new_hf_num;
		_hf_den = new_hf_den;//�����ǵ�һ֡������ֱ����ȣ������ٴ���
	}
	else if (g_trackState == 1){//����֡
		for (int i = 0; i < _hf_num.size(); i++){
			_hf_num[i] = (1 - learning_rate)*_hf_num[i] + learning_rate*new_hf_num[i];
		}
		_hf_den = (1 - learning_rate)*_hf_den + learning_rate*new_hf_den;
		//cv::addWeighted(_hf_num, (1 - learning_rate), new_hf_num, learning_rate, 0, _hf_num);
		//cv::addWeighted(_hf_den, (1 - learning_rate), new_hf_den, learning_rate, 0, _hf_den);
	}
	else{//����
		cout << "something of frame_n is wrong!" << endl;
	}

}

void DSST::ScaleLearn(cv::Mat &image){
	//---------�õ��߶�����˲����������ӿ�(�ӹ���)-----
	//int64 tio = cv::getTickCount();
	cv::Mat scale_xs = GetScaleSample(image);
	//int64 tie = cv::getTickCount() - tio;
	//double time = double(tie) / double(cv::getTickFrequency());

	cv::Mat new_sf_num; //���ڱ�����ģ��
	cv::Mat new_sf_den;

	if (g_trackState == -1){
		//����һ��744*33�ľ��󣬳�ʼ��Ϊ0.���ڵ�һ֡��ʼ��
		scale_xsf = cv::Mat(cv::Size(scale_xs.cols, scale_xs.rows), CV_32F, float(0));
		scale_yf_ = cv::repeat(scale_yf_, scale_xs.rows, 1);//��yf����ֱ����744�Ρ�
	}
	cv::dft(scale_xs, scale_xsf, CV_HAL_DFT_COMPLEX_OUTPUT | CV_HAL_DFT_ROWS);//����һάFFT
	cv::mulSpectrums(scale_yf_, scale_xsf,new_sf_num, 0, true);//�õ����µ�ģ��
	cv::mulSpectrums(scale_xsf, scale_xsf, new_sf_den, 0, true);
	cv::reduce(new_sf_den, new_sf_den, 0, CV_REDUCE_SUM);//�ۼӳ�һ����

	vector<cv::Mat> tmp_sf_den;
	cv::split(new_sf_den, tmp_sf_den);//�Ѻ��и����ľ����
	new_sf_den = tmp_sf_den[0];//�õ�ʵ��ͨ��

	if (g_trackState == -1){//��ʼ֡
		_sf_num = new_sf_num;
		_sf_den = new_sf_den;//�����ǵ�һ֡������ֱ����ȣ������ٴ���
	}
	else if (g_trackState == 1){//����֡
		cv::addWeighted(_sf_num, (1 - learning_rate), new_sf_num, learning_rate, 0,_sf_num);
		cv::addWeighted(_sf_den, (1 - learning_rate), new_sf_den, learning_rate, 0, _sf_den);
	}
	else{//����
		cout << "something of frame_n is wrong!" << endl;
	}
	//cout << _sf_num.colRange(4,6) << endl;
	//cout << _sf_den << endl;
}

//������Էŵ����࣬CF
cv::Mat DSST::CreateGaussian1(int n, float sigma) {
	cv::Mat gaussian1(n, 1, CV_32F);//��Ӧ����Mat_<float>,����һ��������
	
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
	cv::Mat labels = a*b.t(); //�����Ļ�
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
	cv::Mat sample;//����ͼƬ���洢patch

	cv::Size sample_sz = FloorSizeScale(model_sz, cScaleFactor);//�õ�ROI�ĳߴ�
	//��֤patch�ߴ���СΪ2*2
	if (sample_sz.width <= 1) sample_sz.width = 2;
	if (sample_sz.height <= 1) sample_sz.height = 2;

	//�õ����Ͻ���������½�����
	cv::Point lefttop(center_pos.x +1- cvFloor(float(sample_sz.width) / 2.0), center_pos.y +1- cvFloor(float(sample_sz.height) / 2.0));//��0��ʼ�ƣ�
	cv::Point rightbottom(lefttop.x + sample_sz.width, lefttop.y + sample_sz.height);//���½�����,����1�ˣ�û��Ҫ��
	cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
		max(rightbottom.x - im.cols + 1, 0), max(rightbottom.y - im.rows + 1, 0));//�õ�Խ�����ٷ�Χ
	//���ӿ������ͼ��������,���õ���Χ�����ƹ������ϽǺ����½�
	cv::Point lefttop_limit(max(lefttop.x, 0), max(lefttop.y, 0));//�õ���Խ������Ͻ�
	cv::Point rightbottom_limit(min(rightbottom.x, im.cols - 1), min(rightbottom.y, im.rows - 1));//�õ���Խ������½�

	cv::Rect roiRect(lefttop_limit, rightbottom_limit);//�õ����õľ���
	im(roiRect).copyTo(sample);//im(roiRect)����һ�����캯����������һ���м����

	if (border != cv::Rect(0, 0, 0, 0)){//�����Խ�磬��sample�����Ͻ��д����õ�һ���߽縴�Ƶ���ͼ
		cv::copyMakeBorder(sample, sample, border.y, border.height, border.x, border.width, cv::BORDER_REFLECT);
	}
	int interpolation;
	if (model_sz.width > sample_sz.height){//????
		interpolation = CV_INTER_LINEAR;
	}else{
		interpolation = CV_INTER_AREA;
	}
	cv::resize(sample, sample, model_sz, 0, 0, interpolation);//��ͼƬ���Ż�model_sz

	vector<cv::Mat> out = GetTransFeatures(sample); //0.018s
	//cout << out[4].row(68) << endl;
	return out;//������ͼ��
}

/**************************
Function: get a fhog translation sample 
@ im_patch : the search window
@ cell_size: a parameter of fhog
@return: a vector of mat which is fhog of subwindow
***************************/
std::vector<cv::Mat> DSST::GetTransFeatures(cv::Mat im_patch, int cell_size){
	cv::Mat x;
	vector<cv::Mat> x_vector(28);//һ���Ҷ����������hog��27��ͨ������
	vector<cv::Mat> tmp_vector;

	if (im_patch.channels() == 3){//����ǲ�ɫͼ��
		cv::cvtColor(im_patch, im_patch, CV_BGR2GRAY);//���ɻҶ�ͼ��
	}

	im_patch.convertTo(im_patch, CV_32FC1);//ת��float��
	/*cout << im_patch.row(33) << endl;*/
	x_vector[0] = im_patch / 255.0 - 0.5;//�൱����ȡһ���Ҷ�����

	//cell_size�Ĵ�СӰ���˼����ٶ�
	tmp_vector = trans_fhog.extract(im_patch, 2, cell_size);//�����31��ͨ����ǰ27���ݶȽ�ֱ��ͼ��4������

	for (int i = 0; i < 27; i++){//�Ѽ��������hog������ȡ,��Ҫ4����������ͨ����ȫ0ͨ��
		x_vector[i + 1] = tmp_vector[i];
	}
	for (int i = 0; i < 28; i++){//����ÿ��ͨ���ļӴ�����
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
	vector<float> cScaleFactors;//��ǰ�߶��µĳ߶�����
	for (int i = 0; i < number_of_scales; i++){//�õ����µĳ߶�����
		cScaleFactors.push_back(scaleFactors[i] * currentScaleFactor);
	}

	cv::Mat out;//������
	//int64 tio = cv::getTickCount();
	for (int i = 0; i < number_of_scales; i++){//�������г߶�
		cv::Mat sample;
		cv::Size sample_sz = FloorSizeScale(init_target_sz, cScaleFactors[i]);
		cv::Point lefttop(center_pos.x+1 - cvFloor(float(sample_sz.width) / 2.0), center_pos.y+1 - cvFloor(float(sample_sz.height) / 2.0));
		cv::Point rightbottom(lefttop.x + sample_sz.width, lefttop.y + sample_sz.height);//���½�����,����Ҫ��1,��Ϊ�Ǵ�0��ʼ��
		cv::Rect border(-min(lefttop.x, 0), -min(lefttop.y, 0),
			max(rightbottom.x - im.cols + 1, 0), max(rightbottom.y - im.rows + 1, 0));//�õ�Խ�����ٷ�Χ
		//���ӿ������ͼ��������,���õ���Χ�����ƹ������ϽǺ����½�
		cv::Point lefttop_limit(max(lefttop.x, 0), max(lefttop.y, 0));//�õ���Խ������Ͻ�
		cv::Point rightbottom_limit(min(rightbottom.x, im.cols - 1), min(rightbottom.y, im.rows - 1));//�õ���Խ������½�
		cv::Rect roiRect(lefttop_limit, rightbottom_limit);//�õ����õľ���
		im(roiRect).copyTo(sample);//im(roiRect)����һ�����캯����������һ���м����
		if (border != cv::Rect(0, 0, 0, 0)){//�����Խ�磬��sample�����Ͻ��д����õ�һ���߽縴�Ƶ���ͼ
			cv::copyMakeBorder(sample, sample, border.y, border.height, border.x, border.width, cv::BORDER_REFLECT);
		}
		int interpolation;
		if (scale_model_sz.width > sample_sz.height){//????
			interpolation = CV_INTER_LINEAR;
		}
		else{
			interpolation = CV_INTER_AREA;
		}
		cv::resize(sample, sample, scale_model_sz, 0, 0, interpolation);//��ͼƬ���Ż�scale_model_sz
		cv::Mat tmp = GetScaleFeatures(sample,scale_hann_window.at<float>(i,0),4);//��ȡ����������һά����
		if (i == 0) out = tmp; //��һ���߶Ƚ��п�����ֵ
		else {//����߶�
			cv::hconcat(out, tmp, out);//����ƴ�ӣ���һ������
		}
	}
	//int64 tie = cv::getTickCount() - tio;
	//double time = double(tie) / double(cv::getTickFrequency());
	//cout << out.colRange(4,7) << endl; //���������нӽ�200��
	return out; //���ؼӴ����������ͼ
}

/**************************
Function: get a fhog scale sample
@ im_patch : the scale subwindow
@ cell_size: a parameter of fhog
@return: a vector of mat which is fhog of subwindow
***************************/
cv::Mat DSST::GetScaleFeatures(cv::Mat im_patch, float factor_window, int cell_size){
	vector<cv::Mat> x_vector;
	im_patch.convertTo(im_patch, CV_32FC1);//ת��float��
	x_vector = scale_fhog.extract(im_patch, 2, cell_size);//�����31��ͨ����ǰ27���ݶȽ�ֱ��ͼ��4������
	cv::Mat res = cv::Mat::zeros(x_vector.size()*x_vector[0].rows*x_vector[0].cols,1,CV_32FC1);//����һ��������(������fhog����ͼ��ֱ)

	int w = x_vector[0].cols;//�õ�ÿ��ͨ��ͼƬ�Ŀ�
	int h = x_vector[0].rows;//�õ�ÿ��ͨ��ͼƬ�ĳ�
	int len = x_vector.size();//�õ�ͨ����
	for (int k = 0; k < len; k++){//��ʾ�ڼ���ͨ��
		for (int x = 0; x < w; x++){//��ʾ�ڼ���
			for (int y = 0; y < h; y++){ //��ʾ�ڼ���
				res.at<float>(k*w*h + x*h + y, 0) = x_vector[k].at<float>(y, x)*factor_window;
			}	
		}
	}
	return res;//����һ��������������ĳһ�߶���ͼƬHOG����1ά��ʽ(�Ӵ���)
}
