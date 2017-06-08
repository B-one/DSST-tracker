#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "dsst.hpp"

using namespace std;
using namespace cv;
using std::cout;

//-------------FUNCTION PROTYPE----------------
cv::Rect GetInitRect(string gt_path);

//------------
int g_trackState = 0; //-1:Tracking Initing; 1:Tacking; 0:nothing

int main(int argc, char **argv){

	//----reading images in the sequence------------- 	
	string video_path = "F:\\Tracking\\data_seq\\Dog1\\img\\*.jpg";
	vector<cv::String> image_files;//a container with images files name of sequence
	cv::glob(video_path, image_files);
	if (image_files.size() == 0){
		cout << "The sequence not exists!" << endl;
	}

	//-----------Reading the init rect--------------
	string groundtruth_path = "F:\\Tracking\\data_seq\\Dog1\\groundtruth_rect.txt";
	cv::Rect init_rect = GetInitRect(groundtruth_path);

	//----------
	cv::Mat image;
	cv::Mat image_show;
	int64 tic = 0, toc = 0;
	double time = 0;
	bool is_show_visualization = true;
	cv::namedWindow("Tracking");
	cv::Rect result_rect(init_rect);

	//the default parameter: feature = fhog; 
	DSST dsst_tracker;
	//-------------The tracking loop--------------
	for (unsigned frame = 0; frame < 100; frame++){

		image = imread(image_files[frame], -1);//the type is CV_8UC3 or CV_8UC1
		if (image.channels() == 1){
			image_show = image.clone();
			cvtColor(image_show, image_show, CV_GRAY2BGR);
		}else{
			image_show = image;
		}

		tic = getTickCount();
		if (frame == 0){

			g_trackState = -1;//表示初始帧
			dsst_tracker.Init(image, init_rect); //----这里有个转置相乘操作0.3s
			g_trackState = 1;//表示非初始帧

		}
		else{
			result_rect = dsst_tracker.Update(image);
		}
		toc = getTickCount() - tic;
		time += toc;

		if (is_show_visualization){//control the result can be seen in GUI
			cv::putText(image_show, to_string(frame + 1), cv::Point(10, 40), CV_FONT_HERSHEY_SIMPLEX, 1,
				cv::Scalar(0, 255, 255), 2);
			cv::rectangle(image_show, result_rect, cv::Scalar(0, 0, 255), 2);//draw a result_rect in image
			cv::imshow("Tracking", image_show);
			char key = cv::waitKey(1);
			if (key == 27 || tolower(key) == 'q'){
				break;
			}
		}
	}

	//-----------Calculate FPS---------------
	time = time / double(getTickFrequency()); //the time of tracking costs
	double fps = double(image_files.size()) / time;
	cout << "FPS: " << fps << endl;
	cv::destroyAllWindows();

	return 0;
}




/**************************
Function: Get the init rect in groundtruth
@gt_path: the path of groundtruth
@return: init rect
***************************/
cv::Rect GetInitRect(string gt_path){
	ifstream gt(gt_path);
	if (!gt.is_open()){
		cout << "The file" << gt_path << "can not read!" << endl;
	}
	string line;
	getline(gt, line);
	replace(line.begin(), line.end(), ',', ' ');
	stringstream ss(line);
	int tmp1, tmp2, tmp3, tmp4;
	ss >> tmp1 >> tmp2 >> tmp3 >> tmp4; //这里的字符串和整形数可以默认转？
	gt.close();

	return cv::Rect(--tmp1, --tmp2, tmp3, tmp4);//note: the top left is (0,0) not (1,1)
}