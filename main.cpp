#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "helper.hpp"
#include "eigenfaces.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

void loadQMUL(vector<vector<vector<Mat>>> &QMULImages, String QMULAddress, vector<String> QMULNames, vector<String> QMULSubject,
	vector<String> QMULTilt, vector<String> QMULPan);

void loadHeadPose(vector<vector<vector<Mat>>> &headPoseImages, String headPoseAddress, vector<String>headPoseId,
	vector<String>headPoseTilt, vector<String>headPosePan, vector<vector<vector<Point>>> &headPoseAnnotation);

Point readFilebyLine(String path);

int main() {

	String QMULAddress = "C:/Users/Administrator/Desktop/ecse415/project/QMUL";
	String headPosePath = "C:/Users/Administrator/Desktop/ecse415/project/HeadPoseImageDatabase";


	vector<String> QMULNames;

	QMULNames = { "AdamBGrey", "AndreeaVGrey", "CarlaBGrey", "ColinPGrey", "DanJGrey",
		"DennisPGrey", "DennisPNoGlassesGrey", "DerekCGrey", "GrahamWGrey", "HeatherLGrey", "JackGrey", "JamieSGrey",
		"JeffNGrey", "JohnGrey", "JonGrey", "KateSGrey", "KatherineWGrey", "KeithCGrey", "KrystynaNGrey", "PaulVGrey",
		"RichardBGrey", "RichardHGrey", "SarahLGrey", "SeanGGrey", "SeanGNoGlassesGrey", "SimonBGrey", "SueWGrey",
		"TasosHGrey", "TomKGrey", "YogeshRGrey", "YongminYGrey" };

	vector<String> QMULSubject;

	QMULSubject = { "AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ",
		"DennisP", "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack", "JamieS",
		"JeffNG", "John", "OngEJ", "KateS", "KatherineW", "KeithC", "KrystynaN", "PaulV",
		"RichardB", "RichardH", "SarahL", "SeanG", "SeanGNoGlasses", "SimonB", "SueW",
		"TasosH", "TomK", "YogeshR", "YongminY" };

	vector<String> QMULTilt;

	QMULTilt = { "060", "070", "080", "090", "100", "110", "120" };

	vector<String> QMULPan;

	QMULPan = { "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100",
		"110", "120", "130", "140", "150", "160", "170", "180" };
	vector<vector<vector<Mat>>> QMULImages;
	loadQMUL(QMULImages, QMULAddress, QMULNames, QMULSubject, QMULTilt, QMULPan);
	///////////////////////////////////
	
	vector<String>headPoseId = { "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15" };
	vector<String>headPosePan = { "-90", "-75", "-60", "-45", "-30", "-15", "+0", "+15", "+30", "+45", "+60", "+75", "+90" };
	vector<String>headPoseTilt = { "-30", "-15", "+0", "+15", "+30" };
	
	vector<vector<vector<Mat>>> headPoseImages;
	vector<vector<vector<Point>>> headPoseAnnotation;

	loadHeadPose(headPoseImages, headPosePath, headPoseId, headPoseTilt, headPosePan, headPoseAnnotation);
	///////////////////////////////////

	// Build training images set
	// labels: all the image belong to same subject have the same label, label is from 0 to 30
	vector<Mat> trainImages;
	vector<int> trainLabels;
	for (int i = 0; i < QMULNames.size()-12; i++){
		for (int k = 0; k < QMULTilt.size()-1; k++){
			for (int l = 0; l < QMULPan.size(); l++){
				Mat temp = QMULImages[i][k][l];
				trainImages.push_back(temp);
				trainLabels.push_back(i);
			}
		}
	}
	// Build testing images set and labels
	vector<Mat> testImages;
	vector<int> testLabels;
	for (int i = 0; i < QMULNames.size()-25; i++){
		int k = QMULTilt.size() - 1;
		for (int l = 0; l < QMULPan.size(); l++){
			Mat temp = QMULImages[i][k][l];
			testImages.push_back(temp);
			testLabels.push_back(i);
		}
	}
	cout << "training set images " << trainImages.size() << endl;
	cout << "training label numbers " << trainLabels.size() << endl;
	cout << "testing image set " << testImages.size() << endl;
	cout << "testing label numbers " << testLabels.size() << endl;
	cout << "trainImagetype: " << trainImages[0].type() << endl;
	cout << "trainImage size: " << trainImages[0].size() << endl;
	///

	////
	//waitKey();
	//debug
	//imshow("image",trainImages[0]);
	//cout << trainImages[0] << endl;
	//imshow("aftererrorimage", trainImages[1368]);
	waitKey(0);
	
	// get height
	int height = trainImages[0].rows;

	// get test instances
//	Mat testSample = trainImages[trainImages.size() - 1];
//	int testLabel = trainLabels[trainLabels.size() - 1];
	// ... and delete last element
//	trainImages.pop_back();
//	trainLabels.pop_back();

	// num_components eigenfaces
	int num_components = 1000;
	// compute the eigenfaces
	Eigenfaces eigenfaces(trainImages, trainLabels, num_components);
	
	int numberOfCorrect = 0;
	// get a prediction
	for (int i = 0; i < testImages.size(); i++){
		int predicted = eigenfaces.predict(testImages[i]);
		
		if (predicted == testLabels[i]){
			numberOfCorrect++;
		}
	}
	cout << "total test number: " << testLabels.size() << endl;
	cout << "number of correct: " << numberOfCorrect << endl;
	//cout << "actual=" << testLabel << " / predicted=" << predicted << endl;
	
	
	/*
	// reconstruction for test set
	Mat pt = eigenfaces.project(testSample.reshape(1, 1));
	Mat rt = eigenfaces.reconstruct(pt);
	Mat result = toGrayscale(rt.reshape(1, height));
	Mat d;
	absdiff(result, testSample, d);
	cout << "reconstruction error for test set: " << sum(d) << endl;
	//

	// see the reconstruction with num_components
	Mat p = eigenfaces.project(trainImages[0].reshape(1, 1));
	Mat r = eigenfaces.reconstruct(p);
//	imshow("original", trainImages[0]);
	Mat rec_result = toGrayscale(r.reshape(1, height));
//	imshow("reconstruction", rec_result);
//	waitKey(0);
	
	// calculate for reconstruction error
	Mat dst;
	absdiff(rec_result, trainImages[0], dst);
	cout << "reconstruction error for training sets: " << sum(dst) << endl;
	//imshow("dst", dst);
	*/

	//

	/*
	// get the eigenvectors
	Mat W = eigenfaces.eigenvectors();
	// show first 10 eigenfaces
	for (int i = 0; i < min(10, W.cols); i++) {
		Mat ev = W.col(i).clone();
		cout << ev << endl;
		imshow(format("%d", i), norm_0_255(ev.reshape(1, height));
		//toGrayscale(ev.reshape(1, height))	
		waitKey();
	}
	*/
	waitKey(0);
	system("pause");
}

void loadQMUL(vector<vector<vector<Mat>>> &QMULImages, String QMULAddress, vector<String> QMULNames, vector<String> QMULSubject,
	vector<String> QMULTilt, vector<String> QMULPan){
	for (int i = 0; i < QMULNames.size(); i++){
		vector<vector<Mat>> ImageTemp2;

		for (int k = 0; k < QMULTilt.size(); k++){
			vector<Mat> ImageTemp1;
			for (int l = 0; l < QMULPan.size(); l++){
				String pathTemp = QMULSubject[i] + "_" + QMULTilt[k] + "_" + QMULPan[l] + ".ras";
				Mat temp = imread(QMULAddress + "/" + QMULNames[i] + "/" + pathTemp, CV_LOAD_IMAGE_COLOR);
				Mat gray;
				cvtColor(temp, gray, CV_BGR2GRAY);
				ImageTemp1.push_back(gray);
			}
			ImageTemp2.push_back(ImageTemp1);
		}

		QMULImages.push_back(ImageTemp2);
	}

}

void loadHeadPose(vector<vector<vector<Mat>>> &headPoseImages, String headPosePath, vector<String>headPoseId,
	vector<String>headPoseTilt, vector<String>headPosePan, vector<vector<vector<Point>>> &headPoseAnnotation){


	vector<String> series;

	for (int i = 114; i <= 178; i++){
		String temp = to_string(i);
		series.push_back(temp);
	}

	int j = 0;
	for (int i = 0; i < headPoseId.size(); i++){

		vector<vector<Mat>> ImageTemp2;
		vector<vector<Point>> AnnotationTemp2;
		for (int k = 0; k < headPoseTilt.size(); k++){
			vector<Mat> ImageTemp1;
			vector<Point> AnnotationTemp1;

			for (int l = 0; l < headPosePan.size(); l++){
				String path1 = headPosePath + "/" + "Person" + headPoseId[i] + "/" + "person" + headPoseId[i] +
					series[j] + headPoseTilt[k] + headPosePan[l] + ".jpg";
				String path2 = headPosePath + "/" + "Person" + headPoseId[i] + "/" + "person" + headPoseId[i] +
					series[j] + headPoseTilt[k] + headPosePan[l] + ".txt";
				Mat temp = imread(path1, IMREAD_GRAYSCALE);
				Point tempPoint = readFilebyLine(path2);
				ImageTemp1.push_back(temp);
				AnnotationTemp1.push_back(tempPoint);
				j++;
			}
			ImageTemp2.push_back(ImageTemp1);
			AnnotationTemp2.push_back(AnnotationTemp1);
		}

		headPoseImages.push_back(ImageTemp2);
		headPoseAnnotation.push_back(AnnotationTemp2);
		j = 0;
	}
}

Point readFilebyLine(String path){
	ifstream infile(path);
	ifstream infile1(path);
	String x, y;
	for (int i = 0; i < 4; i++){
		getline(infile, x);
	}
	for (int i = 0; i < 5; i++){
		getline(infile1, y);
	}
	int xCoord = atoi(x.c_str());
	int yCoord = atoi(y.c_str());
	Point toReturn;
	toReturn.x = xCoord;
	toReturn.y = yCoord;
	return toReturn;

}