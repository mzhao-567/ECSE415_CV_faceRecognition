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

void kFold(const vector<vector<vector<Mat>>> &QMULImages);
void Train(const vector<vector<vector<Mat>>> &QMULImages, Mat &codeBook, vector<vector<vector<Mat>>> &imageDescriptors, const int numCodewords);
double Test(const vector<vector<vector<Mat>>> &QMULImages, const Mat codeBook, const vector<vector<vector<Mat>>> imageDescriptors);
void drawAnnotationRectangleWithKeypoints(const vector<vector<vector<Mat>>> &QMULImages, vector<KeyPoint> detectedKeypoints, int catIndex, int tiltIndex, int panIndex);

// Initialize constants
const int numCodewords = 20;
const String QMULAddress = "C:/Users/Chianyu/OneDrive/MARVIN/ECSE 415/Project/QMUL/";
const String headPosePath = "C:/Users/Chianyu/OneDrive/MARVIN/ECSE 415/Project/HeadPoseImageDatabase/";

void main() {

	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

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
	QMULPan = { "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", 
				"100", "110", "120", "130", "140", "150", "160", "170", "180" };

	vector<vector<vector<Mat>>> QMULImages;
	loadQMUL(QMULImages, QMULAddress, QMULNames, QMULSubject, QMULTilt, QMULPan);

	vector<vector<vector<Mat>>> headPoseImages;
	vector<vector<vector<Point>>> headPoseAnnotation;

	vector<String>headPoseId = { "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15" };
	vector<String>headPosePan = { "-90", "-75", "-60", "-45", "-30", "-15", "+0", "+15", "+30", "+45", "+60", "+75", "+90" };
	vector<String>headPoseTilt = { "-30", "-15", "+0", "+15", "+30" };

	//loadHeadPose(headPoseImages, headPosePath, headPoseId, headPoseTilt, headPosePan, headPoseAnnotation);

	/***********Bags of Words***********/

	kFold(QMULImages);

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


void kFold(const vector<vector<vector<Mat>>> &QMULImages) {
	cout << "K-Fold Cross Validation with " << numCodewords << " code words" << endl;

	// Vectors of size 7 containing training data and testing data for each k
	vector<vector<vector<vector<Mat>>>> trainingData;
	vector<vector<vector<vector<Mat>>>> testingData;

	for (int f = 0; f < 7; f++) {
		vector<vector<vector<Mat>>> tr_person, te_person;
		for (int i = 0; i < QMULImages.size(); i++) {
			vector<vector<Mat>> tr_tilt, te_tilt;
			for (int j = 0; j < QMULImages[i].size(); j++) {
				vector<Mat> tr_pan, te_pan;
				for (int k = 0; k < QMULImages[i][j].size(); k++) {

					Mat img = QMULImages[i][j][k];

					// int index = 19*j + k;

					if (j == f) {
						te_pan.push_back(img); 
					} else {
						tr_pan.push_back(img);
					}
				}
				if (tr_pan.size() != 0) { tr_tilt.push_back(tr_pan); }
				if (te_pan.size() != 0) { te_tilt.push_back(te_pan); }
			}
			te_person.push_back(te_tilt);
			tr_person.push_back(tr_tilt);
		}
		trainingData.push_back(tr_person);
		testingData.push_back(te_person);
	}

	int tr1 = trainingData[0][0].size();
	int tr2 = trainingData[0][0][0].size();
	int te1 = testingData[0][0].size();
	int te2 = testingData[0][0][0].size();
	cout << trainingData[0].size() << " people in dataset, with " << tr1 << " x "  << tr2 << " images in training per person" << endl;
	cout << testingData[0].size()  << " people in dataset, with " << te1 << " x "  << te2 << " images in testing per person"  << endl;

	double res = 0;

	for (int k = 0; k < 7; k++) {
		Mat codeBook;
		vector<vector<vector<Mat>>> imageDescriptors;

		Train(trainingData[k], codeBook, imageDescriptors, numCodewords);
		res = res + Test(testingData[k], codeBook, imageDescriptors);

		cout << "fold " << (k+1) << " completed;" << endl;
	}
	
	res = res / 7.0;

	cout << "Averaged result = " << res << endl;

	cout << "K-Fold Cross Validation Complete" << endl;
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
		vector<vector<Mat>> histogram_face;
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
			histogram_face.push_back(histogram_tilt);
		}
		imageDescriptors.push_back(histogram_face);

		// Person completion notice
		cout << "BoW " << (i+1) << " completed" << endl;
	}

	// End of function
	cout << "Training complete" << endl;
}

double Test(const vector<vector<vector<Mat>>> &QMULImages, const Mat codeBook, const vector<vector<vector<Mat>>> imageDescriptors) {

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
				
				Mat img = QMULImages[i][j][k];
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
							
							// cout << "image descripter[l][m][n] = \n" << imageDescriptors[l][m][n] << endl;
							// cout << "computed histogram = \n" << histogram << endl;

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
	double ratio = (double)matches / (589.0);

	cout << "Recognition success ratio: " << ratio << endl;
	return ratio;

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