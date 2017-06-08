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
	// the callbackfunction of mouse in tracking
static void onMouse(int event, int x, int y, int, void*);

//-------------GLOBAL VARIABLE-------------------
Point g_origin;//start point
Rect  g_targetInitBox;
bool g_selectObject = false; //a flag 
int g_trackState = 0; //-1:Tracking Initing; 1:Tacking; 0:nothing
Mat g_image;


//*****************************************************
//*****************************************************
int main(int argc, char **argv){
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())  return -1; //verify the camera is open?
	namedWindow("Tracking");
	setMouseCallback("Tracking", onMouse, 0);//set a callbackfunction for mouse

	cv::Mat frame; 
	cv::Rect result_rect;
	int64 tic = 0, toc = 0;
	double time = 0;
	DSST dsst_tracker;//creat a tracker
	uint64 frame_n = 0;
	
	while (true){//tracking loop
		cap >> frame; //reading a frame from camera

		if (g_trackState==0||g_selectObject){ //show the selecting-target in real-time
			frame.copyTo(g_image);
			if (g_targetInitBox.width > 0 && g_targetInitBox.height > 0){
				Mat roi(frame, g_targetInitBox);
				bitwise_not(roi, roi);
			}
		}
		
		if (g_trackState == -1){//Init tracker
			dsst_tracker.Init(frame, g_targetInitBox);
			result_rect=g_targetInitBox; 
			g_trackState = 1; //into tracking
		}else if (g_trackState == 1){//tracking
			frame_n++;
			tic = getTickCount();
			result_rect = dsst_tracker.Update(frame);
			toc = getTickCount() - tic;
			time += toc;
			time = time / double(getTickFrequency());
		}
		
	//show frame in GUI
		if (g_trackState){//tracking
			cv::rectangle(frame, result_rect, cv::Scalar(0, 0, 255), 2);//draw a result_rect in image
			cv::imshow("Tracking", frame);
		}else{
			cv::imshow("Tracking", frame);
		}

	//wait a instruement from user
		char key = cv::waitKey(1);
		if (key == 27 || tolower(key) == 'q'){
			break;
		}
	}

//-----------Calculate FPS---------------
	time = time / double(getTickFrequency()); //the time of tracking costs
	double fps = double(frame_n) / time;
	cout.precision(2);
	cout << "FPS: " << fps << cout.precision() << endl;
	cv::destroyAllWindows();

	return 0;
}

/**************************
Function: 
***************************/
static void onMouse(int event, int x, int y, int, void*) //加上static，该函数只能在本文件中被使用；没有名字只有类型，表示接收但不使用
{
	if (g_selectObject){
		g_targetInitBox.x = MIN(x, g_origin.x);//从此时读取鼠标的x,y与第一次点的x,y进行对比，始终保持矩形框的左上角
		g_targetInitBox.y = MIN(y, g_origin.y);
		g_targetInitBox.width = std::abs(x - g_origin.x);//得到两者的绝对值，这个abs是来自于std命名空间
		g_targetInitBox.height = std::abs(y - g_origin.y);//

		g_targetInitBox &= Rect(0, 0, g_image.cols, g_image.rows);//矩形是按标准的x,y,宽,高；得到画的矩形与图形的交集，即两者相与
	}
	switch (event){
	case EVENT_LBUTTONDOWN://左键按下
		g_origin = Point(x, y);//存储点击点
		g_targetInitBox = Rect(x, y, 0, 0);//左上角
		g_selectObject = true;//开始挑选目标
		break;
	case EVENT_LBUTTONUP://左键松开
		g_selectObject = false;//结果目标挑选
		if (g_targetInitBox.width > 0 && g_targetInitBox.height > 0){//高度和宽度都大于0，即该框合理
			g_trackState = -1; //初始化目标标志
		}
		break;
	}
}
