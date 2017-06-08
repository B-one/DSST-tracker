/*
    - c++ wrapper for the piotr toolbox
    Created by Tomas Vojir, 2014
*/


#ifndef FHOG_HEADER_7813784354687
#define FHOG_HEADER_7813784354687

#include <vector>
#include <opencv2/opencv.hpp>

#include "gradientMex.h"//fhog的实现层


class FHoG
{
public:
    //description: extract hist. of gradients(use_hog == 0), hog(use_hog == 1) or fhog(use_hog == 2)
    //input: float one channel frame as input, hog type
    //return: computed descriptor
    std::vector<cv::Mat> extract(const cv::Mat & img, int use_hog = 2, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
    {
        // d frame dimension -> gray frame d = 1
        // h, w -> height, width of frame
        // full -> 0：代表从0-180度；1：代表是0-360度
        // I -> input frame, M, O -> mag, orientation OUTPUT
        int h = img.rows, w = img.cols, d = 1;
        bool full = true;
        if (h < 2 || w < 2) {
            std::cerr << "I must be at least 2x2." << std::endl;
            return std::vector<cv::Mat>();
        }

//        //frame rows-by-rows
//        float * I = new float[h*w];
//        for (int y = 0; y < h; ++y) {
//            const float * row_ptr = img.ptr<float>(y);
//            for (int x = 0; x < w; ++x) {
//                I[y*w + x] = row_ptr[x];
//            }
//        }

        //frame cols-by-cols

        //cv::Mat i = img.clone();
        //i = i.t();
        //float *I = i.ptr<float>(0);

        float * I = new float[h*w];//申请一个能装有patch所有像素的空间，I是float*
        for (int x = 0; x < w; ++x) {//从
            for (int y = 0; y < h; ++y) {//按列进行stack
                I[x*h + y] = img.at<float>(y, x)/255.f;//这里进行了归一化
            }
        }

        float *M = new float[h*w], *O = new float[h*w];//同样建立对的M、O的空间
        gradMag(I, M, O, h, w, d, full);//M：该像素处的梯度幅值；O：该像素值的梯度角

		//选择最后结果的通道数
        int n_chns = (use_hog == 0) ? n_orients : (use_hog==1 ? n_orients*4 : n_orients*3+5);
        int hb = h/bin_size, wb = w/bin_size;

        float *H = new float[hb*wb*n_chns];//保存最后得出来的结果，是以cell_size为单位的
        memset(H, 0, hb*wb*n_chns*sizeof(float));//对这块空间进行初始化为0

        if (use_hog == 0) {
            full = false;   //by default
            gradHist( M, O, H, h, w, bin_size, n_orients, soft_bin, full );//光计算梯度直方图就可以了
        } else if (use_hog == 1) {
            full = false;   //by default
            hog( M, O, H, h, w, bin_size, n_orients, soft_bin, full, clip );
        } else {//计算fhog,不需要full参数
            fhog( M, O, H, h, w, bin_size, n_orients, soft_bin, clip );
        }

        //convert, assuming row-by-row-by-channel storage
        std::vector<cv::Mat> res;
		//remove the last channel which is all zeros
        int n_res_channels = (use_hog == 2) ? n_chns-1 : n_chns;    //last channel all zeros for fhog
        res.reserve(n_res_channels);//至少要开辟多少空间，这是vector的函数
        for (int i = 0; i < n_res_channels; ++i) {//遍历所有通道
            //output rows-by-rows
//            cv::Mat desc(hb, wb, CV_32F, (H+hb*wb*i));

            //output cols-by-cols
            cv::Mat desc(hb, wb, CV_32F);
            for (int x = 0; x < wb; ++x) {
                for (int y = 0; y < hb; ++y) {//将H中的元素按要求转回矩阵形式
                    desc.at<float>(y,x) = H[i*hb*wb + x*hb + y];
                }
            }
            res.push_back(desc.clone());//转好的放到结果处

        }

        //clean 把用到的动态空间释放一下
        delete [] I;
        delete [] M;
        delete [] O;
        delete [] H;

        return res;
    }
};

#endif //FHOG_HEADER_7813784354687