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
void probabilistic(vector<vector<Mat>> &imageDescriptors, const int numCodewords);
void Train(const vector<vector<Mat>> &trainingData, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords);
double Test(const vector<vector<Mat>> &testingData, const Mat codeBook, const vector<vector<Mat>> imageDescriptors);
void drawAnnotationRectangleWithKeypoints(const vector<vector<vector<Mat>>> &QMULImages, vector<KeyPoint> detectedKeypoints, int catIndex, int tiltIndex, int panIndex);

// Initialize constants
const int numCodewords = 50;
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
	vector<vector<vector<Mat>>> trainingData;
	vector<vector<vector<Mat>>> testingData;

	// for each fold
	for (int f = 0; f < 7; f++) {
		vector<vector<Mat>> tr_person, te_person;

		// for each person
		for (int i = 0; i < 31; i++) {
			vector<Mat> tr_pics, te_pics;
			
			// for each tilt angle
			for (int j = 0; j < QMULImages[i].size(); j++) {		

				// for each pan angle
				for (int k = 0; k < QMULImages[i][j].size(); k++) {

					Mat img = QMULImages[i][j][k];

					// pseudorandom pick for folds
					int index = (19*j + k) % 7;
					if (index == f) { 
						te_pics.push_back(img); 
					}
					else { 
						tr_pics.push_back(img); 
					}
				}
			}

			te_person.push_back(te_pics);
			tr_person.push_back(tr_pics);
		}
		trainingData.push_back(tr_person);
		testingData.push_back(te_person);
	}

	int tr = trainingData[0][0].size();
	int te = testingData[0][0].size();
	cout << trainingData[0].size() << " people in dataset, with " << tr << " images in training per person" << endl;
	cout << testingData[0].size() <<  " people in dataset, with " << te << " images in testing per person" << endl;

	double res = 0;

	for (int k = 0; k < 7; k++) {
		Mat codeBook;
		vector<vector<Mat>> imageDescriptors;

		Train(trainingData[k], codeBook, imageDescriptors, numCodewords);
		// probabilistic(imageDescriptors, numCodewords);
		res = res + Test(testingData[k], codeBook, imageDescriptors);

		cout << "fold " << (k + 1) << " completed;" << endl;
	}

	res = res / 7.0;

	cout << "Averaged result = " << res << endl;

	cout << "K-Fold Cross Validation Complete" << endl;
}

void probabilistic(vector<vector<Mat>> &imageDescriptors, const int numCodewords) {
	cout << "Probabilistic initiated" << endl;
	cout << imageDescriptors.size() << " histogram sets with " << imageDescriptors[0].size() << " histograms per set" << endl;

	vector<vector<vector<Mat>>> gaussian;

	// for each set
	for (int i = 0; i < imageDescriptors.size(); i++) {
		Mat histogram_mat;

		// for each histogram
		for (int j = 0; j < imageDescriptors[i].size(); j++) {
			Mat histogram = imageDescriptors[i][j];		
			histogram_mat.push_back(histogram.row(j));
		}

		Mat means(1, numCodewords, CV_64FC1);
		Mat covar(numCodewords, numCodewords, CV_64FC1);

		cout << "test" << endl;
		calcCovarMatrix(histogram_mat, covar, means, CV_COVAR_NORMAL | CV_COVAR_ROWS, CV_64F);
		cout << "test" << endl;
		//Mat invCo = covar.inv();
		// cout << "det = " << determinant(covar) << endl;
		cout << "means of size " << means.size() << " = \n" << means << endl;
		cout << "covar of size " << covar.size() << " = \n" << covar << endl;

	}

	cout << "Probabilistic complete" << endl;
}

void Train(const vector<vector<Mat>> &trainingData, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords) {
	cout << "Training" << endl;

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	Mat D;

	// detect SIFT key points for each training image of each object category

	// for each person
	for (int i = 0; i < trainingData.size(); i++) {

		// for each picture
		for (int j = 0; j < trainingData[i].size(); j++) {

			Mat img = trainingData[i][j];
			vector<KeyPoint> keypoints;
			featureDetector->detect(img, keypoints);

			// drawAnnotationRectangleWithKeypoints(QMULImages, keypoints, i, j, k);

			Mat SIFTdescriptors;
			descriptorExtractor->compute(img, keypoints, SIFTdescriptors);
			D.push_back(SIFTdescriptors);
		}

		// Person completion notice
		cout << "Person " << (i+1) << " of " << trainingData.size() << " completed" << endl;
	}

	BOWKMeansTrainer bowTrainer = BOWKMeansTrainer(numCodewords);
	bowTrainer.add(D);
	codeBook = bowTrainer.cluster();

	Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);
	bowDescriptorExtractor->setVocabulary(codeBook);

	// Extract image descriptor from each image
	// for each person
	for (int i = 0; i < trainingData.size(); i++) {
		vector<Mat> histogram_face;

		// for each image
		for (int j = 0; j < trainingData[i].size(); j++) {

			Mat img = trainingData[i][j];
			vector<KeyPoint> keypoints;
			featureDetector->detect(img, keypoints);

			Mat histogram_temp;
			bowDescriptorExtractor->compute2(img, keypoints, histogram_temp);

			// Normalize histogram 0 to 1
			normalize(histogram_temp, histogram_temp, 0, 1, NORM_MINMAX, -1, Mat());

			histogram_face.push_back(histogram_temp);
		}

		imageDescriptors.push_back(histogram_face);

		// Person completion notice
		cout << "BoW " << (i + 1) << " completed" << endl;
	}

	// End of function
	cout << "Training complete" << endl;
}

double Test(const vector<vector<Mat>> &testingData, const Mat codeBook, const vector<vector<Mat>> imageDescriptors) {

	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher>  descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);

	bowDescriptorExtractor->setVocabulary(codeBook);

	int matches = 0;

	cout << "Testing" << endl;

	// for each person
	for (int i = 0; i < testingData.size(); i++) {

		// for each tilt image
		for (int j = 0; j < testingData[i].size(); j++) {

			Mat img = testingData[i][j];
			vector<KeyPoint> keypoints;
			featureDetector->detect(img, keypoints);

			Mat histogram;
			bowDescriptorExtractor->compute2(img, keypoints, histogram);

			// Normalize BoW histogram
			normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat());

			double minDist = INFINITY;
			int bestMatch = -1;

			// for each person descriptor
			for (int l = 0; l < imageDescriptors.size(); l++) {

				// for each tilt angle descriptor
				for (int m = 0; m < imageDescriptors[l].size(); m++) {

					Mat h1 = imageDescriptors[l][m];
					Mat h2 = histogram;

					Mat h3 = (h1 - h2).mul(h1 - h2);
					Mat h4 = (h1 + h2);
					Mat chi_mat = h3.mul(1 / h4);
					double chi = cv::sum(chi_mat)[0];

					if (chi < minDist) {
						minDist = chi;
						bestMatch = l;
					}
				}
			}

			if (bestMatch == i) {
				matches++;
				// cout << matches << " match(es) found" << endl;
			}
		}

		// Person completion notice
		cout << "Person " << (i + 1) << " of " << testingData.size() << " tested" << endl;
	}

	// ratio = matches/numTestingData
	double ratio = (double)matches / double(testingData.size() * testingData[0].size());
	cout << "Recognition success ratio: " << ratio << endl;
	
	// End of function
	cout << "Testing complete" << endl;
	
	return ratio;	
}

void drawAnnotationRectangleWithKeypoints(const vector<vector<vector<Mat>>> &QMULImages, vector<KeyPoint> detectedKeypoints, int catIndex, int tiltIndex, int panIndex) {
	Mat testImage;
	Rect annotation = Rect(0, 0, QMULImages[catIndex][tiltIndex][panIndex].cols, QMULImages[catIndex][tiltIndex][panIndex].rows);
	drawKeypoints(QMULImages[catIndex][tiltIndex][panIndex], detectedKeypoints, testImage);
	rectangle(testImage, annotation, Scalar(255, 0, 255), 2);
	imshow("Test Image", testImage);
	waitKey(0);
}