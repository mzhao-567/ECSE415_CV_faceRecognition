//
//  main.cpp
//  ECSE415-A2
//
//  Created by Zhao on 16/2/18.
//  Copyright © 2016年 Zhao. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>


using namespace cv;
using namespace std;



/* Helper class declaration and definition */
class Caltech101
{
public:
    Caltech101(string datasetPath, const int numTrainingImages, const int numTestImages)
    {
        cout << "Loading Caltech 101 dataset" << endl;
        numImagesPerCategory = numTrainingImages + numTestImages;
        
        // load "Categories.txt"
        ifstream infile(datasetPath + "/" + "Categories.txt");
        cout << "\tChecking Categories.txt" << endl;
        if (!infile.is_open())
        {
            cout << "\t\tError: Cannot find Categories.txt in " << datasetPath << endl;
            return;
        }
        cout << "\t\tOK!" << endl;
        
        // Parse category names
        cout << "\tParsing category names" << endl;
        string catname;
        while (getline(infile, catname))
        {
            categoryNames.push_back(catname);
        }
        cout << "\t\tdone!" << endl;
        
        // set num categories
        int numCategories = (int)categoryNames.size();
        
        // initialize outputs size
        trainingImages = vector<vector<Mat>>(numCategories);
        trainingAnnotations = vector<vector<Rect>>(numCategories);
        testImages = vector<vector<Mat>>(numCategories);
        testAnnotations = vector<vector<Rect>>(numCategories);
        
        // generate training and testing indices
        randomShuffle();
        
        // Load data
        cout << "\tLoading images and annotation files" << endl;
        string imgDir = datasetPath + "/" + "Images";
        string annotationDir = datasetPath + "/" + "Annotations";
        for (int catIdx = 0; catIdx < categoryNames.size(); catIdx++)
            //for (int catIdx = 0; catIdx < 1; catIdx++)
        {
            string imgCatDir = imgDir + "/" + categoryNames[catIdx];
            string annotationCatDir = annotationDir + "/" + categoryNames[catIdx];
            for (int fileIdx = 0; fileIdx < numImagesPerCategory; fileIdx++)
            {
                // use shuffled training and testing indices
                int shuffledFileIdx = indices[fileIdx];
                // generate file names
                stringstream imgFilename, annotationFilename;
                imgFilename << "image_" << setfill('0') << setw(4) << shuffledFileIdx << ".jpg";
                annotationFilename << "annotation_" << setfill('0') << setw(4) << shuffledFileIdx << ".txt";
                
                // Load image
                string imgAddress = imgCatDir + '/' + imgFilename.str();
                //cout<<"imgAddress"<<imgAddress<<endl;
                Mat img = imread(imgAddress, CV_LOAD_IMAGE_COLOR);
                //imshow("hehe", img);
                // check image data
                if (!img.data)
                {
                    cout << "\t\tError loading image in " << imgAddress << endl;
                    return;
                }
                
                // Load annotation
                string annotationAddress = annotationCatDir + '/' + annotationFilename.str();
                ifstream annotationIFstream(annotationAddress);
                // Checking annotation file
                if (!annotationIFstream.is_open())
                {
                    cout << "\t\tError: Error loading annotation in " << annotationAddress << endl;
                    return;
                }
                int tl_col, tl_row, width, height;
                Rect annotRect;
                while (annotationIFstream >> tl_col >> tl_row >> width >> height)
                {
                    annotRect = Rect(tl_col - 1, tl_row - 1, width, height);
                }
                
                // Split training and testing data
                if (fileIdx < numTrainingImages)
                {
                    // Training data
                    trainingImages[catIdx].push_back(img);
                    trainingAnnotations[catIdx].push_back(annotRect);
                }
                else
                {
                    // Testing data
                    testImages[catIdx].push_back(img);
                    testAnnotations[catIdx].push_back(annotRect);
                }
            }
        }
        cout << "\t\tdone!" << endl;
        successfullyLoaded = true;
        cout << "Dataset successfully loaded: " << numCategories << " categories, " << numImagesPerCategory  << " images per category" << endl << endl;
    }
    
    bool isSuccessfullyLoaded()	{  return successfullyLoaded; }
    
    void dispTrainingImage(int categoryIdx, int imageIdx)
    {
        Mat image = trainingImages[categoryIdx][imageIdx];
        Rect annotation = trainingAnnotations[categoryIdx][imageIdx];
        rectangle(image, annotation, Scalar(255, 0, 255), 2);
        imshow("Annotated training image", image);
        waitKey(0);
        destroyWindow("Annotated training image");
    }
    
    void dispTestImage(int categoryIdx, int imageIdx)
    {
        Mat image = testImages[categoryIdx][imageIdx];
        Rect annotation = testAnnotations[categoryIdx][imageIdx];
        rectangle(image, annotation, Scalar(255, 0, 255), 2);
        imshow("Annotated test image", image);
        waitKey(0);
        destroyWindow("Annotated test image");
    }
    
    vector<string> categoryNames;
    vector<vector<Mat>> trainingImages;
    vector<vector<Rect>> trainingAnnotations;
    vector<vector<Mat>> testImages;
    vector<vector<Rect>> testAnnotations;
    
private:
    bool successfullyLoaded = false;
    int numImagesPerCategory;
    vector<int> indices;
    void randomShuffle()
    {
        // set init values
        for (int i = 1; i <= numImagesPerCategory; i++) indices.push_back(i);
        
        // permute using built-in random generator
        random_shuffle(indices.begin(), indices.end());
    }
};

/* Function prototypes */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords);
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors);

int main(void)
{
    /* Initialize OpenCV nonfree module */
    initModule_nonfree();
    
    /* Put the full path of the Caltech 101 folder here */
    const string datasetPath = "/Users/Zhao/Desktop/Caltech101";
    
    /* Set the number of training and testing images per category */
    const int numTrainingData = 40;
    const int numTestingData = 2;
    
    /* Set the number of codewords*/
    const int numCodewords = 10;
    
    /* Load the dataset by instantiating the helper class */
    Caltech101 Dataset(datasetPath, numTrainingData, numTestingData);
    

    
    /* Terminate if dataset is not successfull loaded */
    if (!Dataset.isSuccessfullyLoaded())
    {
        cout << "An error occurred, press Enter to exit" << endl;
        getchar();
        return 0;
    }	
    
    /* Variable definition */
    Mat codeBook;	
    vector<vector<Mat>> imageDescriptors;
    
    /* Training */	
    Train(Dataset, codeBook, imageDescriptors, numCodewords);
    
    
    /* Testing */	
    Test(Dataset, codeBook, imageDescriptors);
}

/* Train BoW */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
{
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );
    Mat D;
    //use build-in classes
    //vector<KeyPoint> pointsSift;
    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 40; j++){
            vector<KeyPoint> kp_sift;
            //Mat imgToShow;
            Mat Descriptor;
            detector->detect(Dataset.trainingImages[i][j], kp_sift);
            //drawKeypoints(Dataset.trainingImages[i][j],kp_sift,imgToShow,Scalar::all(-1));
            //cout << kp_sift.size() << endl;
            for(int k = 0; k < kp_sift.size(); k++){ // discard key points
                if(kp_sift[k].pt.x > Dataset.trainingAnnotations[i][j].x + Dataset.trainingAnnotations[i][j].height || kp_sift[k].pt.y > Dataset.trainingAnnotations[i][j].y + Dataset.trainingAnnotations[i][j].width){
                    kp_sift.erase(kp_sift.begin() + k);
                }
            }
            //rectangle(imgToShow,cvPoint(Dataset.trainingAnnotations[i][j].x, Dataset.trainingAnnotations[i][j].y),cvPoint(Dataset.trainingAnnotations[i][j].x + Dataset.trainingAnnotations[i][j].height,Dataset.trainingAnnotations[i][j].y + Dataset.trainingAnnotations[i][j].width),Scalar(255,0,0),2);
            descriptor_extractor->compute(Dataset.trainingImages[i][j], kp_sift, Descriptor);
            //resize(Descriptor, Descriptor2, Size(D.cols,Descriptor.rows));
            D.push_back(Descriptor); //generate descriptors, later use it to generate the codebook.
            //imshow("hehehe:",imgToShow);
            //waitKey();
        }
    }
    //cout << "hehe1" << endl;

    
    BOWKMeansTrainer bowTraining(numCodewords);//10 clusters count
    bowTraining.add(D);
    codeBook = bowTraining.cluster();//generate the codebook.
    
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );
    BOWImgDescriptorExtractor bowDE(descriptor_extractor, descriptor_matcher);
    bowDE.setVocabulary(codeBook);// try to find a bag of words histogram for each training images using the codebook.
    vector<Mat> imgDescriptorsTemp;
    for(int i = 0; i < 20; i++){
        vector<Mat> imgDescriptorsTemp;
        //Mat meanDescriptor;
        for(int j = 0; j < 40; j++){
            vector<KeyPoint> kp_sift;
            Mat bowHistogram;
            
            detector->detect(Dataset.trainingImages[i][j], kp_sift);
            for(int k = 0; k < kp_sift.size(); k++){
                if(kp_sift[k].pt.x > Dataset.trainingAnnotations[i][j].x + Dataset.trainingAnnotations[i][j].height || kp_sift[k].pt.y > Dataset.trainingAnnotations[i][j].y + Dataset.trainingAnnotations[i][j].width){
                    kp_sift.erase(kp_sift.begin() + k);
                }
            }
            bowDE.compute2(Dataset.trainingImages[i][j], kp_sift, bowHistogram);
            //meanDescriptor.push_back(bowHistogram);
            imgDescriptorsTemp.push_back(bowHistogram);
        }
        //cout << mean(imgDescriptorsTemp) << endl;
        //Scalar catMean = mean(imgDescriptorsTemp);
        //cout << mean(meanDescriptor) << endl;
        imageDescriptors.push_back(imgDescriptorsTemp);
    }
    //cout << "hehe4" << endl;
}

/* Test BoW */
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors)
{
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );
    BOWImgDescriptorExtractor bowDE(descriptor_extractor, descriptor_matcher);
    bowDE.setVocabulary(codeBook);
    String testLabelTemp;
    int correctDectCounter = 0;
    double numImg = 0.0;
    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 2; j++){
            vector<KeyPoint> kp_sift;
            
            Mat bowHistogram;
            detector->detect(Dataset.testImages[i][j], kp_sift);
            for(int k = 0; k < kp_sift.size(); k++){
                if(kp_sift[k].pt.x > Dataset.testAnnotations[i][j].x + Dataset.testAnnotations[i][j].height || kp_sift[k].pt.y > Dataset.testAnnotations[i][j].y + Dataset.testAnnotations[i][j].width){
                    kp_sift.erase(kp_sift.begin() + k);
                }
            }
            //generate bag of word histogram for test images. With 2 images per category with 20 category. BogHistogram is generated
            //the same way as in the training function.
            bowDE.compute2(Dataset.testImages[i][j], kp_sift, bowHistogram);
            double minDistance = 100;
            double currentDistance = 101;
            int catIndex = 100;
            // Compare the histogram for each test image with each training image. Pick the category label for the closest match.
            for(int l = 0; l < 20; l++){
                for (int m = 0; m < 40; m++){
                    currentDistance = norm(bowHistogram, imageDescriptors[l][m]);
                    if(currentDistance < minDistance){
                        minDistance = currentDistance;
                        catIndex = l;
                    }
                }
            }
            testLabelTemp = Dataset.categoryNames[catIndex];
            if (testLabelTemp == Dataset.categoryNames[i]) {
                correctDectCounter++;
            }
            numImg++;
            //double distance = norm(bowHistogram,imageDescriptors[i]);
            
        }
    }
    double recongRatio = correctDectCounter / numImg;
    cout << recongRatio << endl;
}




