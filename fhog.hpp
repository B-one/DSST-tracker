/*
    - c++ wrapper for the piotr toolbox
    Created by Tomas Vojir, 2014
*/


#ifndef FHOG_HEADER_7813784354687
#define FHOG_HEADER_7813784354687

#include <vector>
#include <opencv2/opencv.hpp>

#include "gradientMex.h"//fhog��ʵ�ֲ�


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
        // full -> 0�������0-180�ȣ�1��������0-360��
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

        float * I = new float[h*w];//����һ����װ��patch�������صĿռ䣬I��float*
        for (int x = 0; x < w; ++x) {//��
            for (int y = 0; y < h; ++y) {//���н���stack
                I[x*h + y] = img.at<float>(y, x)/255.f;//��������˹�һ��
            }
        }

        float *M = new float[h*w], *O = new float[h*w];//ͬ�������Ե�M��O�Ŀռ�
        gradMag(I, M, O, h, w, d, full);//M�������ش����ݶȷ�ֵ��O��������ֵ���ݶȽ�

		//ѡ���������ͨ����
        int n_chns = (use_hog == 0) ? n_orients : (use_hog==1 ? n_orients*4 : n_orients*3+5);
        int hb = h/bin_size, wb = w/bin_size;

        float *H = new float[hb*wb*n_chns];//�������ó����Ľ��������cell_sizeΪ��λ��
        memset(H, 0, hb*wb*n_chns*sizeof(float));//�����ռ���г�ʼ��Ϊ0

        if (use_hog == 0) {
            full = false;   //by default
            gradHist( M, O, H, h, w, bin_size, n_orients, soft_bin, full );//������ݶ�ֱ��ͼ�Ϳ�����
        } else if (use_hog == 1) {
            full = false;   //by default
            hog( M, O, H, h, w, bin_size, n_orients, soft_bin, full, clip );
        } else {//����fhog,����Ҫfull����
            fhog( M, O, H, h, w, bin_size, n_orients, soft_bin, clip );
        }

        //convert, assuming row-by-row-by-channel storage
        std::vector<cv::Mat> res;
		//remove the last channel which is all zeros
        int n_res_channels = (use_hog == 2) ? n_chns-1 : n_chns;    //last channel all zeros for fhog
        res.reserve(n_res_channels);//����Ҫ���ٶ��ٿռ䣬����vector�ĺ���
        for (int i = 0; i < n_res_channels; ++i) {//��������ͨ��
            //output rows-by-rows
//            cv::Mat desc(hb, wb, CV_32F, (H+hb*wb*i));

            //output cols-by-cols
            cv::Mat desc(hb, wb, CV_32F);
            for (int x = 0; x < wb; ++x) {
                for (int y = 0; y < hb; ++y) {//��H�е�Ԫ�ذ�Ҫ��ת�ؾ�����ʽ
                    desc.at<float>(y,x) = H[i*hb*wb + x*hb + y];
                }
            }
            res.push_back(desc.clone());//ת�õķŵ������

        }

        //clean ���õ��Ķ�̬�ռ��ͷ�һ��
        delete [] I;
        delete [] M;
        delete [] O;
        delete [] H;

        return res;
    }
};

#endif //FHOG_HEADER_7813784354687