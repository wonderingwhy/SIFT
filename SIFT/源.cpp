#include <iostream>  
#include <fstream>  
#include <vector>
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/stitching.hpp"  

using namespace std;
using namespace cv;

#define DEBUG
#define SQR(x) ((x)*(x))
#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 
const string IMG_PATH = "jobs.jpg";
const string D_DIR = "./DoGPyramid/";
const string G_DIR = "./GussianPyramid/";
const double INIT_SIGMA = 0.5;
const double SIGMA = 1.6;
const int INTERVALS = 3;
const double RATIO = 10.;
const int MAX_INTERPOLATION_STEPS = 10;
const double DXTHRESHOLD = 0.03;
const int ORI_HIST_BINS = 36;
const double ORI_SIGMA_TIMES = 1.5;
const double ORI_WINDOW_RADIUS = 3 * ORI_SIGMA_TIMES;
const double ORI_PEAK_RATIO = 0.8;
const int FEATURE_ELEMENT_LENGTH = 128;
const int  DESCR_HIST_BINS = 8;
const int IMG_BORDER = 20;
const int DESCR_WINDOW_WIDTH = 4;
const int DESCR_SCALE_ADJUST = 3;
const double DESCR_MAG_THR = 0.2;
const double INT_DESCR_FCTR = 512.;
const double K_RATIO = pow(2., 1. / INTERVALS);

Mat img_show;
struct KPoint {
	int octave;
	double interval;
	double offset_interval;
	double x, y;
	double scale;
	int dx, dy;
	double offset_x, offset_y;
	double octave_scale;
	double ori;
	int descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];
	//double val;
	KPoint(){}
	KPoint(int octave, int interval, int x, int y) :octave(octave), interval(interval), x(x), y(y) {}
};
void GetImageData(Mat_<double> &src) {
	Mat img = imread(IMG_PATH, 0);
#ifdef DEBUG
	img_show = imread(IMG_PATH, 1);
#endif
	img.copyTo(src);
}
void CalculateScale(vector<KPoint> &goodExtremas)
{
#ifdef DEBUG
	puts("enter \"CalculateScale\"");
#endif
	for (int i = 0; i < goodExtremas.size(); i++)
	{
		goodExtremas[i].scale = SIGMA * pow(2.0, goodExtremas[i].octave + goodExtremas[i].interval / INTERVALS);
		goodExtremas[i].octave_scale = SIGMA * pow(2.0, goodExtremas[i].interval / INTERVALS);
		goodExtremas[i].dx = cvRound(goodExtremas[i].x * pow(2, goodExtremas[i].octave - 1));
		goodExtremas[i].dy = cvRound(goodExtremas[i].y * pow(2, goodExtremas[i].octave - 1));
	}

}
void BuildGaussianPyramid(const Mat_<double> src, vector<Mat_<double>> &GPyramid, const int octaves) {
#ifdef DEBUG
	puts("enter \"BuildGaussianPyramid\"");
#endif
	double *sigmas = new double[INTERVALS + 3];
	sigmas[0] = SIGMA;
	Mat_<double> src2;

	pyrUp(src, src2, Size(src.cols << 1, src.rows << 1));

#ifdef DEBUG
	//imwrite("src2.jpg", src2);
#endif
	int interval_total = INTERVALS + 3;
	for (int i = 1; i < interval_total; i++) {
		double sig_prev = pow(K_RATIO, i - 1) * SIGMA;
		double sig_total = sig_prev * K_RATIO;
		sigmas[i] = sqrt(SQR(sig_total) - SQR(sig_prev));
	}
	for (int o = 0; o < octaves; o++) {
		for (int i = 0; i < interval_total; i++) {
			Mat_<double> m;
			if (o == 0 && i == 0) {
				double sig_init = sqrt(SQR(SIGMA) - SQR(INIT_SIGMA * 2));
				GaussianBlur(src2, m, Size(), sig_init, sig_init);
			}
			else if (i == 0) {
				Mat_<double> oFirst = GPyramid[(o - 1) * (interval_total)+INTERVALS];
				pyrDown(oFirst, m, Size(oFirst.cols >> 1, oFirst.rows >> 1));
			}
			else {
				Mat_<double> prev = GPyramid[o * (interval_total)+i - 1];
				GaussianBlur(prev, m, Size(), sigmas[i], sigmas[i]);
			}
			GPyramid.push_back(m);
		}
	}
	delete[] sigmas;

#ifdef DEBUG
	for (int i = 0; i < GPyramid.size(); ++i)  {
		char a[5];
		itoa(i, a, 10);
		if (i < 10) {
			imwrite(G_DIR + "GPyramid0" + (string)a + ".jpg", GPyramid[i]);
		}
		else {
			imwrite(G_DIR + "GPyramid" + (string)a + ".jpg", GPyramid[i]);
		}
	}
#endif
}
void BuildDoGPyramid(const vector<Mat_<double>> &GPyramid, vector<Mat_<double>> &DPyramid, const int octaves) {
#ifdef DEBUG
	puts("enter \"BuildDoGPyramid\"");
#endif
	int interval_total = INTERVALS + 2;
	DPyramid.resize(GPyramid.size() - octaves);
	for (int o = 0; o < octaves; ++o) {
		for (int i = 0; i < interval_total; ++i) {
			DPyramid[o*interval_total + i] = GPyramid[o * (interval_total + 1) + i + 1] - GPyramid[o * (interval_total + 1) + i];
		}
	}
#ifdef DEBUG
	for (int i = 0; i < DPyramid.size(); ++i)  {
		char a[5];
		itoa(i, a, 10);
		if (i < 10) {
			imwrite(D_DIR + "DPyramid0" + (string)a + ".jpg", DPyramid[i]);
		}
		else {
			imwrite(D_DIR + "DPyramid" + (string)a + ".jpg", DPyramid[i]);
		}
	}
#endif
}
bool isExtrema(const vector<Mat_<double>> &DPyramid, const int onum, const int inum, const int rnum, const int cnum) {
	int intervals_total = INTERVALS + 2;
	double v = DPyramid[onum * intervals_total + inum].at<double>(rnum, cnum);
	KPoint cp = KPoint(onum, inum, rnum, cnum);
	double maxi = v, mini = v;
	for (int di = -1; di <= 1; ++di) {
		for (int dr = -1; dr <= 1; ++dr) {
			for (int dc = -1; dc <= 1; ++dc) {
				if (di == 0 && dr == 0 && dc == 0) {
					continue;
				}
				int i = inum + di, r = rnum + dr, c = cnum + dc;
				double temp = DPyramid[onum * intervals_total + i].at<double>(r, c);
				maxi = max(maxi, temp);
				mini = min(mini, temp);
				if ((maxi != v && mini != v) || temp == v) {
					return false;
				}
			}
		}
	}
	return true;
}
void GetExtremas(const vector<Mat_<double>> &DPyramid, vector<KPoint> &extremas, const int octaves) {
#ifdef DEBUG
	puts("enter \"GetExtremas\"");
#endif
	int intervals_total = INTERVALS + 2;
	for (int o = 0; o < octaves; ++o) {
		for (int i = 1; i < intervals_total - 1; ++i) {
			Mat_<double> m = DPyramid[o * intervals_total + i];
			for (int r = IMG_BORDER; r < m.rows - IMG_BORDER; ++r) {
				for (int c = IMG_BORDER; c < m.cols - IMG_BORDER; ++c) {
					if (isExtrema(DPyramid, o, i, r, c)) {
						extremas.push_back(KPoint(o, i, r, c));
					}
				}
			}
		}
	}
#ifdef DEBUG
	int cnt = 0;
	for (int i = 0; i < extremas.size(); ++i) {
		printf("%d: (%d, %lf, %lf, %lf)\n", ++cnt,
			extremas[i].octave, extremas[i].interval, extremas[i].x, extremas[i].y);
	}
	//imshow("show", img_show);
	//cvWaitKey();
#endif
}
bool isInRange(int onum, int inum, int rnum, int cnum) {
	if (inum < 1 || inum >= INTERVALS + 1) {
		return false;
	}
	if (rnum < 1 || rnum * pow(2, onum - 1) >= img_show.rows - 1) {
		return false;
	}
	if (cnum < 1 || cnum * pow(2, onum - 1) >= img_show.cols - 1) {
		return false;
	}
	return true;
}
int num = 0;
void SelectGoodExtremas(const vector<Mat_<double>> &DPyramid, vector<KPoint> &extremas, vector<KPoint> &goodExtremas) {
#ifdef DEBUG
	puts("enter \"SelectGoodExtremas\"");
#endif
	int intervals_total = INTERVALS + 2;
	for (int i = 0; i < extremas.size(); ++i) {
		Mat dX, DX = Mat::zeros(Size(1, 3), CV_64FC1);
		Mat Hessian3X3 = Mat::zeros(Size(3, 3), CV_64FC1);
		Mat Hessian2X2 = Mat::zeros(Size(2, 2), CV_64FC1);
		bool flag = false;
		int onum, inum, rnum, cnum;
		double v;
		if (i == 496) {
			++num;
		}
		for (int j = 0; j < MAX_INTERPOLATION_STEPS; ++j) {
			onum = extremas[i].octave;
			inum = cvRound(extremas[i].interval);
			rnum = cvRound(extremas[i].x);
			cnum = cvRound(extremas[i].y);
			
			//cout << "o = " << onum << endl;
			//cout << "i = " << inum << endl;
			//cout << "r = " << rnum << endl;
			//cout << "c = " << cnum << endl;
			//printf("%d %lf %lf %lf\n", extremas[i].octave, extremas[i].interval, extremas[i].x, extremas[i].y);
			if (rnum == 28 && cnum == 27) {
				++num;
			}
			if (isInRange(onum, inum, rnum, cnum) == false) {
				break;
			}

			
			Mat_<double> mat = DPyramid[onum * intervals_total + inum];
			Mat_<double> mat1 = DPyramid[onum * intervals_total + inum + 1];
			Mat_<double> mat_1 = DPyramid[onum * intervals_total + inum - 1];
			v = mat.at<double>(rnum, cnum);
			//cout << "v = " << v << endl;

			double dxx = mat.at<double>(rnum + 1, cnum) + mat.at<double>(rnum - 1, cnum) - 2 * v;
			double dyy = mat.at<double>(rnum, cnum + 1) + mat.at<double>(rnum, cnum - 1) - 2 * v;
			double dzz = mat1.at<double>(rnum, cnum) + mat_1.at<double>(rnum, cnum) - 2 * v;
			double dxy = (mat.at<double>(rnum + 1, cnum + 1) + mat.at<double>(rnum - 1, cnum - 1) -
				mat.at<double>(rnum + 1, cnum - 1) - mat.at<double>(rnum - 1, cnum + 1)) / 4;
			double dxz = (mat1.at<double>(rnum, cnum + 1) + mat_1.at<double>(rnum, cnum - 1) -
				mat1.at<double>(rnum, cnum - 1) - mat_1.at<double>(rnum, cnum + 1)) / 4;
			double dyz = (mat1.at<double>(rnum + 1, cnum) + mat_1.at<double>(rnum - 1, cnum) -
				mat1.at<double>(rnum - 1, cnum) - mat_1.at<double>(rnum + 1, cnum)) / 4;
			
			Hessian3X3.at<double>(0, 0) = dxx;
			Hessian3X3.at<double>(0, 1) = dxy;
			Hessian3X3.at<double>(0, 2) = dxz;
			Hessian3X3.at<double>(1, 0) = dxy;
			Hessian3X3.at<double>(1, 1) = dyy;
			Hessian3X3.at<double>(1, 2) = dyz;
			Hessian3X3.at<double>(2, 0) = dxz;
			Hessian3X3.at<double>(2, 1) = dyz;
			Hessian3X3.at<double>(2, 2) = dzz;
			
			Hessian2X2.at<double>(0, 0) = dxx;
			Hessian2X2.at<double>(0, 1) = dxy;
			Hessian2X2.at<double>(1, 0) = dxy;
			Hessian2X2.at<double>(1, 1) = dyy;

			double dx = mat.at<double>(rnum + 1, cnum) - mat.at<double>(rnum, cnum);
			double dy = mat.at<double>(rnum, cnum + 1) - mat.at<double>(rnum, cnum);
			double dz = mat1.at<double>(rnum, cnum) - mat.at<double>(rnum, cnum);

			DX.at<double>(0, 0) = dx;
			DX.at<double>(1, 0) = dy;
			DX.at<double>(2, 0) = dz;

			dX = -Hessian3X3.inv() * DX;
			//cout << dX << endl;
			extremas[i].x = rnum + dX.at<double>(0, 0);
			extremas[i].y = cnum + dX.at<double>(1, 0);
			extremas[i].interval = inum + dX.at<double>(2, 0);
			if (fabs(dX.at<double>(0, 0)) < 0.5 && fabs(dX.at<double>(1, 0)) < 0.5 && fabs(dX.at<double>(2, 0)) < 0.5) {
				flag = true;
				break;
			}
			//cout << "v = " << v << endl;
		}
		if (flag) {
			Mat temp = DX.t() * dX * 0.5;
			double DValue = temp.at<double>(0, 0) + v;
			cout << DValue << endl;
			//cout << Hessian2X2 << endl;
			//cout << trace(Hessian2X2)[0] << endl;
			double ratio = SQR(trace(Hessian2X2)[0]) / determinant(Hessian2X2);

			if (fabs(DValue) > 255 * DXTHRESHOLD && ratio < SQR(RATIO + 1) / RATIO) {
				KPoint kp = extremas[i];
				goodExtremas.push_back(kp);
			}
		}
	}
#ifdef DEBUG
	int cnt = 0;
	for (int i = 0; i < goodExtremas.size(); ++i) {
		printf("%d: (%d, %lf, %lf, %lf)\n", ++cnt,
			goodExtremas[i].octave, goodExtremas[i].interval, goodExtremas[i].x, goodExtremas[i].y);

		int tx = (int)(goodExtremas[i].x * pow(2, goodExtremas[i].octave - 1) + 0.5);
		int ty = (int)(goodExtremas[i].y * pow(2, goodExtremas[i].octave - 1) + 0.5);
		for (int j = -1; j <= 1; ++j) {
			for (int k = -1; k <= 1; ++k) {
				img_show.at<Vec3b>(tx + j, ty + k) = Vec3b(0, 0, 255);
			}
		}

	}
	imshow("show", img_show);
	cvWaitKey();
#endif
}
void GetOrientations(const vector<KPoint> &goodExtremas, vector<KPoint> &features, vector<Mat_<double>> &GPyramid) {
#ifdef DEBUG
	puts("enter \"GetOrientations\"");
#endif
	int intervals_total = INTERVALS + 3;
	double *hist = new double[ORI_HIST_BINS];
	int c0 = 0;
	for (int i = 0; i < goodExtremas.size(); ++i) {
		memset(hist, 0, sizeof(double)* ORI_HIST_BINS);
		int onum = goodExtremas[i].octave, inum = cvRound(goodExtremas[i].interval);
		int rnum = cvRound(goodExtremas[i].x), cnum = cvRound(goodExtremas[i].y);
		Mat_<double> mat = GPyramid[onum * intervals_total + inum];

		double radius = ORI_WINDOW_RADIUS * goodExtremas[i].octave_scale;
		double sigma = ORI_SIGMA_TIMES * goodExtremas[i].octave_scale;
		double econs = -1. / (2. * sigma * sigma);

		for (int j = -(int)radius; j <= (int)radius; ++j) {
			for (int k = -(int)radius; k <= (int)radius; ++k) {

				double x1y0 = mat.at<double>(rnum + j + 1, cnum + k);
				double x_1y0 = mat.at<double>(rnum + j - 1, cnum + k);
				double x0y1 = mat.at<double>(rnum + j, cnum + k + 1);
				double x0y_1 = mat.at<double>(rnum + j, cnum + k - 1);

				double weight = exp((SQR(j) + SQR(k)) * econs);
				double m = sqrt(SQR(x1y0 - x_1y0) + SQR(x0y1 - x0y_1));
				double theta = atan2(x0y1 - x0y_1, x1y0 - x_1y0);

				int bin = cvRound(ORI_HIST_BINS * (CV_PI - theta) / (2 * CV_PI));
				bin = bin < ORI_HIST_BINS ? bin : 0;

				hist[bin] += m * weight;
			}
		}
		double *temp = new double[ORI_HIST_BINS];
		memset(temp, 0, sizeof(double)* ORI_HIST_BINS);
		for (int j = 0; j < ORI_HIST_BINS; ++j) {
			int l = j == 0 ? ORI_HIST_BINS - 1 : j - 1;
			int r = j == ORI_HIST_BINS - 1 ? 0 : j + 1;
			temp[j] = 0.25 * hist[l] + 0.5 * hist[j] + 0.25 * hist[r];
		}
		for (int j = 0; j < ORI_HIST_BINS; ++j) {
			int l = j == 0 ? ORI_HIST_BINS - 1 : j - 1;
			int r = j == ORI_HIST_BINS - 1 ? 0 : j + 1;
			hist[j] = 0.25 * temp[l] + 0.5 * temp[j] + 0.25 * temp[r];
		}
		double maxi = 0;
		for (int j = 0; j < ORI_HIST_BINS; ++j) {
			maxi = max(maxi, hist[j]);
		}
		int c1 = 0;
		for (int j = 0; j < ORI_HIST_BINS; ++j) {
			int l = j == 0 ? ORI_HIST_BINS - 1 : j - 1;
			int r = j == ORI_HIST_BINS - 1 ? 0 : j + 1;
			if (hist[j] > hist[l] && hist[j] > hist[r] && hist[j] > maxi * ORI_PEAK_RATIO) {
				++c1;
				double bin = j + Parabola_Interpolate(hist[l], hist[j], hist[r]);
				bin = bin < 0 ? (bin + ORI_HIST_BINS) : bin;
				bin = bin >= ORI_HIST_BINS ? (bin - ORI_HIST_BINS) : bin;
				KPoint kp = goodExtremas[i];
				kp.ori = ((CV_PI * 2 * bin) / ORI_HIST_BINS) - CV_PI;
				features.push_back(kp);
			}
		}

		if (c1 != 1) {
			++c0;
		}
#ifdef DEBUG
		
		printf("***i = %d***\n", i);
		/*
		printf("temp:\n", i);
		for (int j = 0; j < ORI_HIST_BINS; ++j) {
			printf("%lf\n", temp[j]);
		}
		*/
		printf("hist:\n", i);
		for (int j = 0; j < ORI_HIST_BINS; ++j) {
			printf("%lf\n", hist[j]);
		}
		
#endif
		delete[] temp;
	}
#ifdef DEBUG
	printf("ratio = %lf\n", c0 * 1. / goodExtremas.size());
#endif
	delete[] hist;
#ifdef DEBUG
	for (int i = 0; i < features.size(); ++i) {
		printf("features %d: (%d, %lf, %lf, %lf, %lf)\n", i,
			features[i].octave, features[i].interval, features[i].x, features[i].y, features[i].ori);
	}
#endif
}
void DrawSiftFeatures(const vector<KPoint> features) {
	for (int i = 0; i < features.size(); ++i) {
		double len = 15;
		int start_x = features[i].dx;
		int start_y = features[i].dy;
		int end_x = (int)(len * cos(features[i].ori) + 0.5) + start_x;
		int end_y = (int)(len * -sin(features[i].ori) + 0.5) + start_y;
		CvPoint start = cvPoint(start_y, start_x);
		CvPoint end = cvPoint(end_y, end_x);
		line(img_show, start, end, CV_RGB(0, 255, 0), 1, 8, 0);
	}
}
void GetSiftFeatures(Mat_<uchar> src, vector<KPoint> &features) {

#ifdef DEBUG
	puts("enter \"GetSiftFeatures\"");
#endif

	int octaves = (int)log(min(src.rows, src.cols) * 1.) / log(2.) - 2;

#ifdef DEBUG
	printf("rows = %d, cols = %d, octaves = %d\n", src.rows, src.cols, octaves);
	//imshow("show", src);
	//cvWaitKey();
#endif

	vector<Mat_<double>> GPyramid, DPyramid;
	BuildGaussianPyramid(src, GPyramid, octaves);
	BuildDoGPyramid(GPyramid, DPyramid, octaves);

	vector<KPoint> extremas, goodExtremas;
	GetExtremas(DPyramid, extremas, octaves);
	SelectGoodExtremas(DPyramid, extremas, goodExtremas);
	CalculateScale(goodExtremas);
	GetOrientations(goodExtremas, features, GPyramid);
#ifdef DEBUG
	DrawSiftFeatures(features);
	imshow("show", img_show);
	cvWaitKey();
#endif
}
int main(int argc, char** argv)
{
	Mat_<double> src;
	GetImageData(src);

	vector<KPoint> features;
	GetSiftFeatures(src, features);
	//write_features(features, "descriptor.txt");
	//imshow("src", src);
	system("pause");
	return 0;
}
