#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

void loadQMUL(vector<vector<vector<Mat>>> &QMULImages, String QMULAddress, vector<String> QMULNames, vector<String> QMULSubject,
	vector<String> QMULTilt, vector<String> QMULPan);
void loadHeadPose(vector<vector<vector<Mat>>> &headPoseImages, String headPoseAddress, vector<String>headPoseId,
	vector<String>headPoseTilt, vector<String>headPosePan, vector<vector<vector<Point>>> &headPoseAnnotation);
Point readFilebyLine(String path);

void Train(const vector<vector<vector<Mat>>> &QMULImages, Mat &codeBook, vector<vector<vector<Mat>>> &imageDescriptors, const int numCodewords);
void Test(const vector<vector<vector<Mat>>> &QMULImages, const Mat codeBook, const vector<vector<vector<Mat>>> imageDescriptors);
void drawAnnotationRectangleWithKeypoints(const vector<vector<vector<Mat>>> &QMULImages, vector<KeyPoint> detectedKeypoints, int catIndex, int tiltIndex, int panIndex);

void main() {

	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	String QMULAddress = "C:/Users/Chianyu/OneDrive/MARVIN/ECSE 415/Project/QMUL/";
	String headPosePath = "C:/Users/Chianyu/OneDrive/MARVIN/ECSE 415/Project/HeadPoseImageDatabase/";

	vector<String> QMULNames;

	QMULNames = { "AdamBGrey", "AndreeaVGrey", "CarlaBGrey", "ColinPGrey", "DanJGrey",
		"DennisPGrey", "DennisPNoGlassesGrey", "DerekCGrey", "GrahamWGrey", "HeatherLGrey", "JackGrey", "JamieSGrey",
		"JeffNGrey", "JohnGrey", "JonGrey", "KateSGrey", "KatherineWGrey", "KeithCGrey", "KrystynaNGrey", "PaulVGrey",
		"RichardBGrey", "RichardHGrey", "SarahLGrey", "SeanGGrey", "SeanGNoGlassesGrey", "SimonBGrey", "SueWGrey",
		"TasosHGrey", "TomKGrey", "YogeshRGrey", "YongminYGrey" };

	vector<String> QMULSubject;

	QMULSubject = { "AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ",
		"DennisP", "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack", "JamieS",
		"JeffN", "John", "Jon", "KateS", "KatherineW", "KeithC", "KrystynaN", "PaulV",
		"RichardB", "RichardH", "SarahL", "SeanG", "SeanGNoGlasses", "SimonB", "SueW",
		"TasosH", "TomK", "YogeshR", "YongminY" };

	vector<String> QMULTilt;

	QMULTilt = { "060", "070", "080", "090", "100", "110", "120" };

	vector<String> QMULPan;

	QMULPan = { "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100",
		"110", "120", "130", "140", "150", "160", "170", "180" };

	vector<vector<vector<Mat>>> QMULImages;

	loadQMUL(QMULImages, QMULAddress, QMULNames, QMULSubject, QMULTilt, QMULPan);

	vector<vector<vector<Mat>>> headPoseImages;
	vector<vector<vector<Point>>> headPoseAnnotation;

	vector<String>headPoseId = { "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15" };
	vector<String>headPosePan = { "-90", "-75", "-60", "-45", "-30", "-15", "+0", "+15", "+30", "+45", "+60", "+75", "+90" };
	vector<String>headPoseTilt = { "-30", "-15", "+0", "+15", "+30" };

	//loadHeadPose(headPoseImages, headPosePath, headPoseId, headPoseTilt, headPosePan, headPoseAnnotation);


	/***********Bags of Words***********/

	/* Set the number of codewords*/
	const int numCodewords = 20;

	/* Variable definition */
	Mat codeBook;
	vector<vector<vector<Mat>>> imageDescriptors;



	/* Training */
	Train(QMULImages, codeBook, imageDescriptors, numCodewords);
	
	/* Testing */
	Test(QMULImages, codeBook, imageDescriptors);

	cin.ignore();
}

void loadQMUL(vector<vector<vector<Mat>>> &QMULImages, String QMULAddress, vector<String> QMULNames, vector<String> QMULSubject,
	vector<String> QMULTilt, vector<String> QMULPan) {

	for (int i = 0; i < QMULNames.size(); i++) {
		vector<vector<Mat>> ImageTemp2;

		for (int k = 0; k < QMULTilt.size(); k++){
			vector<Mat> ImageTemp1;
			for (int l = 0; l < QMULPan.size(); l++){
				String pathTemp = QMULSubject[i] + "_" + QMULTilt[k] + "_" + QMULPan[l] + ".ras";
				Mat temp = imread(QMULAddress + "/" + QMULNames[i] + "/" + pathTemp);
				ImageTemp1.push_back(temp);
			}
			ImageTemp2.push_back(ImageTemp1);
		}
		QMULImages.push_back(ImageTemp2);
	}
}

void loadHeadPose(vector<vector<vector<Mat>>> &headPoseImages, String headPosePath, vector<String>headPoseId,
	vector<String>headPoseTilt, vector<String>headPosePan, vector<vector<vector<Point>>> &headPoseAnnotation) {

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
				Mat temp = imread(path1);
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


void Train(const vector<vector<vector<Mat>>> &QMULImages, Mat &codeBook, vector<vector<vector<Mat>>> &imageDescriptors, const int numCodewords) {
	cout << "Training" << endl;

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	vector<KeyPoint> keypoints;
	Mat D;

	// detect SIFT key points for each training image of each object category
	for (int i = 0; i < QMULImages.size(); i++) {
		for (int j = 0; j < QMULImages[i].size(); j++) {
			for (int k = 0; k < QMULImages[i][j].size(); k++) {

				Mat img = QMULImages[i][j][k];
				featureDetector->detect(img, keypoints);
				
				// drawAnnotationRectangleWithKeypoints(QMULImages, keypoints, i, j, k);

				Mat SIFTdescriptors;
				descriptorExtractor->compute(img, keypoints, SIFTdescriptors);
				D.push_back(SIFTdescriptors);
			}
		}
		// Person completion notice
		cout << "Person " << (i+1) << " of " << QMULImages.size() << " completed" << endl;
	}

	BOWKMeansTrainer bowTrainer(numCodewords);
	bowTrainer.add(D);
	codeBook = bowTrainer.cluster();
	
	Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);
	bowDescriptorExtractor->setVocabulary(codeBook);

	// Extract image descriptor from each image
	for (int i = 0; i < QMULImages.size(); i++) {
		vector<vector<Mat>> histogram_person;
		for (int j = 0; j < QMULImages[i].size(); j++) {
			vector<Mat> histogram_tilt;
			for (int k = 0; k < QMULImages[i][j].size(); k++) {
				Mat img = QMULImages[i][j][k];

				featureDetector->detect(img, keypoints);

				Mat histogram_pan;
				bowDescriptorExtractor->compute2(img, keypoints, histogram_pan);
				
				// Normalize BoW histogram
				normalize(histogram_pan, histogram_pan, 0, 1, NORM_MINMAX, -1, Mat());
				
				histogram_tilt.push_back(histogram_pan);		
			}
			histogram_person.push_back(histogram_tilt);
		}
		imageDescriptors.push_back(histogram_person);

		// Person completion notice
		cout << "BoW " << (i+1) << " completed" << endl;
	}

	// End of function
	cout << "Training complete" << endl;
}

void Test(const vector<vector<vector<Mat>>> &QMULImages, const Mat codeBook, const vector<vector<vector<Mat>>> imageDescriptors) {

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);

	bowDescriptorExtractor->setVocabulary(codeBook);

	vector<KeyPoint> keypoints;

	int matches = 0;

	cout << "Testing" << endl;

	for (int i = 0; i < QMULImages.size(); i++) {
		for (int j = 0; j < QMULImages[i].size(); j++) {
			for (int k = 0; k < QMULImages[i][j].size(); k++) {
				
				Mat img = QMULImages[i][j][j];
				featureDetector->detect(img, keypoints);
				
				Mat histogram;
				bowDescriptorExtractor->compute2(img, keypoints, histogram);

				// Normalize BoW histogram
				normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

				double minDist = INFINITY;
				int bestMatch = -1;
				for (int l = 0; l < imageDescriptors.size(); l++) {
					for (int m = 0; m < imageDescriptors[l].size(); m++) {
						for (int n = 0; n < imageDescriptors[l][m].size(); n++) {
							
							double dist = compareHist(imageDescriptors[l][m][n], histogram, CV_COMP_CHISQR);
							// double dist = norm(imageDescriptors[l][m][n], histogram);

							if (dist < minDist) {
								minDist = dist;
								bestMatch = l;
							}							
						}
					}
				}

				if (bestMatch == i) {
					matches++;
					// cout << matches << " match(es) found" << endl;
				}
			}
		}

		// Person completion notice
		cout << "Person " << (i + 1) << " of " << QMULImages.size() << " tested" << endl;
	}

	// ratio = matches/numTrainingData
	double ratio = (double)matches / (133*31);

	cout << "Recognition success ratio: " << ratio << endl;

	// End of function
	cout << "Testing complete" << endl;
}

void drawAnnotationRectangleWithKeypoints(const vector<vector<vector<Mat>>> &QMULImages, vector<KeyPoint> detectedKeypoints, int catIndex, int tiltIndex, int panIndex) {
	Mat testImage;
	Rect annotation = Rect(0, 0, QMULImages[catIndex][tiltIndex][panIndex].cols, QMULImages[catIndex][tiltIndex][panIndex].rows);
	drawKeypoints(QMULImages[catIndex][tiltIndex][panIndex], detectedKeypoints, testImage);
	rectangle(testImage, annotation, Scalar(255, 0, 255), 2);
	imshow("Test Image", testImage);
	waitKey(0);
}