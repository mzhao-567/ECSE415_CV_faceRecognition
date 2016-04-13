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

void loadQMUL(vector<vector<vector<Mat>>> &QMULImages, String QMULAddress, vector<String> QMULNames,vector<String> QMULSubject,
              vector<String> QMULTilt, vector<String> QMULPan);

void loadHeadPose(vector<vector<vector<Mat>>> &headPoseImages, String headPoseAddress,vector<String>headPoseId,
                  vector<String>headPoseTilt, vector<String>headPosePan, vector<vector<vector<Point>>> &headPoseAnnotation);

Point readFilebyLine(String path);



//Fuctions include in LBP

vector<Mat> LBPHist(vector<Mat> trainData, vector<int> labels);
vector<Mat> DivideFour(vector<Mat> Images, vector<int> Labels, vector<int> &returnLabels);
double LBPTest(vector<int> trainLabels, vector<Mat> trainImages, vector<int> testLabels, vector<Mat> testImages, int pyramidLevel);
double LBPTestProbability(vector<int> trainLabels, vector<Mat> trainImages, vector<int> testLabels, vector<Mat> testImages, int pyramidLevel);
vector<int> LBPConfusionMatrix(vector<int> trainLabels, vector<Mat> trainImages, vector<int> testLabels, vector<Mat> testImages, int pyramidLevel);





int main(void) {
    
    
    String QMULAddress = "/Users/Zhao/Desktop/415_Project/QMUL";
    String headPosePath = "/Users/Zhao/Desktop/415_Project/HeadPoseImageDatabase";
    
    
    vector<String> QMULNames;
    
    QMULNames = {"AdamBGrey", "AndreeaVGrey","CarlaBGrey","ColinPGrey","DanJGrey",
    "DennisPGrey","DennisPNoGlassesGrey","DerekCGrey","GrahamWGrey","HeatherLGrey","JackGrey","JamieSGrey",
    "JeffNGrey","JohnGrey","JonGrey","KateSGrey","KatherineWGrey","KeithCGrey","KrystynaNGrey","PaulVGrey",
    "RichardBGrey","RichardHGrey","SarahLGrey","SeanGGrey","SeanGNoGlassesGrey","SimonBGrey","SueWGrey",
    "TasosHGrey","TomKGrey","YogeshRGrey","YongminYGrey"};
    
    vector<String> QMULSubject;
    
    QMULSubject = {"AdamB", "AndreeaV","CarlaB","ColinP","DanJ",
        "DennisP","DennisPNoGlasses","DerekC","GrahamW","HeatherL","Jack","JamieS",
        "JeffNG","John","OngEJ","KateS","KatherineW","KeithC","KrystynaN","PaulV",
        "RichardB","RichardH","SarahL","SeanG","SeanGNoGlasses","SimonB","SueW",
        "TasosH","TomK","YogeshR","YongminY"};
    
    vector<String> QMULTilt;
    
    QMULTilt = {"060","070","080","090","100","110","120"};
    
    vector<String> QMULPan;
    
    QMULPan = {"000","010","020","030","040","050","060","070","080","090","100",
        "110","120","130","140","150","160","170","180"};
    
    vector<vector<vector<Mat>>> QMULImages;
    
    loadQMUL(QMULImages, QMULAddress, QMULNames, QMULSubject, QMULTilt, QMULPan);
    
    vector<vector<vector<Mat>>> headPoseImages;
    vector<vector<vector<Point>>> headPoseAnnotation;
    
    vector<String>headPoseId = {"01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"};
    vector<String>headPosePan = {"-90","-75","-60","-45","-30","-15","+0","+15","+30","+45","+60","+75","+90"};
    vector<String>headPoseTilt ={"-30","-15","+0","+15","+30"};
    
    //loadHeadPose(headPoseImages, headPosePath, headPoseId, headPoseTilt, headPosePan,headPoseAnnotation);
    
    
    vector<Mat> LBPTrain;
    vector<int> LBPTrainLabel;
    vector<Mat> LBPHistogram;
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 19; j++){
            cvtColor(QMULImages[0][i][j], QMULImages[0][i][j], CV_BGR2GRAY);
            LBPTrain.push_back(QMULImages[0][i][j]);
            int label = 0;
            LBPTrainLabel.push_back(label);
        }
    }
    
    //LBPTest is the first one, original LBP, section 2.2.1
    //double rate = LBPTest(LBPTrainLabel, LBPTrain, LBPTrainLabel, LBPTrain, 1);
    //cout << rate << endl;
    //vector<int> hehe = LBPConfusionMatrix(LBPTrainLabel, LBPTrain, LBPTrainLabel, LBPTrain, 1);
    double hehe = LBPTestProbability(LBPTrainLabel, LBPTrain, LBPTrainLabel, LBPTrain, 0);
    cout << hehe << endl;
    
}



void loadQMUL(vector<vector<vector<Mat>>> &QMULImages, String QMULAddress, vector<String> QMULNames,vector<String> QMULSubject,
              vector<String> QMULTilt, vector<String> QMULPan){
    
    
    
    for(int i = 0; i < QMULNames.size(); i++){
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

void loadHeadPose(vector<vector<vector<Mat>>> &headPoseImages, String headPosePath,vector<String>headPoseId,
             vector<String>headPoseTilt, vector<String>headPosePan, vector<vector<vector<Point>>> &headPoseAnnotation){
    
    
    vector<String> series;
    
    for(int i = 114; i <=178; i++){
        String temp = to_string(i);
        series.push_back(temp);
    }
    
    int j = 0;
    for(int i = 0; i < headPoseId.size(); i++){
        
        vector<vector<Mat>> ImageTemp2;
        vector<vector<Point>> AnnotationTemp2;
            for(int k = 0; k < headPoseTilt.size(); k++){
                vector<Mat> ImageTemp1;
                vector<Point> AnnotationTemp1;
                
                for(int l = 0; l < headPosePan.size(); l++){
                    String path1 = headPosePath + "/" + "Person" + headPoseId[i] + "/" + "person" + headPoseId[i] +
                    series[j] + headPoseTilt[k] + headPosePan[l]+ ".jpg";
                    String path2 = headPosePath + "/" + "Person" + headPoseId[i] + "/" + "person" + headPoseId[i] +
                    series[j] + headPoseTilt[k] + headPosePan[l]+ ".txt";
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
    String x,y;
    for(int i = 0; i < 4; i++){
        getline(infile,x);
    }
    for(int i = 0; i < 5; i++){
        getline(infile1,y);
    }
    int xCoord = atoi(x.c_str());
    int yCoord = atoi(y.c_str());
    Point toReturn;
    toReturn.x = xCoord;
    toReturn.y = yCoord;
    return toReturn;

}

vector<Mat> LBPHist(vector<Mat> trainData, vector<int> labels){
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    model->train(trainData, labels);
    vector<Mat> LBPHistogram = model->getMatVector("histograms");
    return LBPHistogram;
    //model->set("threshold", 0.0);
    //Mat testSample = trainData[1];
    //int predictedLabel = model->predict(testSample);
    //cout << predictedLabel << endl;
}


double LBPTest(vector<int> trainLabels, vector<Mat> trainImages, vector<int> testLabels, vector<Mat> testImages, int pyramidLevel){
    
    vector<Mat> trainHist;
    vector<Mat> testHist;
    
    vector<int> dividedTrainLabels;
    vector<int> dividedTestLabels;
    
    vector<Mat> dividedTrainImages;
    vector<Mat> dividedTestImages;
    
    if(pyramidLevel == 0){
        trainHist = LBPHist(trainImages,trainLabels);
        testHist = LBPHist(testImages, testLabels);
        //in this case we do not need to use the divided labels, we can use the passed in test and train labels.
    }
    
    if(pyramidLevel == 1){
        dividedTrainImages = DivideFour(trainImages, trainLabels, dividedTrainLabels);
        dividedTestImages = DivideFour(testImages, testLabels, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    if(pyramidLevel == 2){
        
        vector<int> labelTrain1;
        vector<int> labelTest1;
        
        vector<Mat> trainHist1 = DivideFour(trainImages, trainLabels, labelTrain1);
        vector<Mat> testHist1 = DivideFour(testImages, testLabels, labelTest1);
        dividedTrainImages = DivideFour(trainHist1, labelTrain1, dividedTrainLabels);
        dividedTestImages = DivideFour(testHist1, labelTest1, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    if(pyramidLevel == 3){
        
        vector<int> labelTrain1;
        vector<int> labelTrain2;
        vector<int> labelTest1;
        vector<int> labelTest2;
        
        vector<Mat> trainHist1 = DivideFour(trainImages, trainLabels, labelTrain1);
        vector<Mat> testHist1 = DivideFour(testImages, testLabels, labelTest1);
        vector<Mat> trainHist2 = DivideFour(trainHist1, labelTrain1, labelTrain2);
        vector<Mat> testHist2 = DivideFour(testHist1, labelTest1, labelTest2);
        dividedTrainImages = DivideFour(trainHist2, labelTrain2, dividedTrainLabels);
        dividedTestImages = DivideFour(testHist2, labelTest2, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    double numCorrectCase = 0;
    double totalNumImages = 0;
    double distance = 100;
    int MatchingIndex = 0;
    
    
    //Compare for the result.
    
    for(int i = 0; i < trainHist.size(); i++){
        for(int j = 0; j < testHist.size(); j++){
            double currentDistance = compareHist(trainHist[i], testHist[j], CV_COMP_CHISQR);
            if(currentDistance < distance){
                distance = currentDistance;
                MatchingIndex = i;
            }
            if(pyramidLevel == 0){
                if(trainLabels[MatchingIndex] == testLabels[j]){
                    numCorrectCase++;
                }
                totalNumImages++;
            }
            
            if(pyramidLevel > 0){
                if(dividedTrainLabels[MatchingIndex] == dividedTestLabels[j]){
                    numCorrectCase++;
                }
                totalNumImages++;
            }
        
        }
    }
    
    
    double correctRate = numCorrectCase / totalNumImages;
    return correctRate;
    
}



 vector<Mat> DivideFour(vector<Mat> Images, vector<int> Labels, vector<int> &returnLabels){
     //returnLabels should be passes as an empty vector<int> which stores the label for images after dividing into 4.
     vector<Mat> dividedSet;
     
     Rect rectLeftUp, rectRightUp, rectLeftDown, rectRightDown;
     
     rectLeftUp = Rect(0,0,Images[0].cols /2, Images[0].rows / 2);
     rectRightUp = Rect(Images[0].cols / 2, 0, Images[0].cols / 2, Images[0].rows / 2);
     rectLeftDown = Rect(0, Images[0].rows/2, Images[0].cols / 2, Images[0].rows / 2);
     rectRightDown = Rect(Images[0].cols/2, Images[0].rows / 2, Images[0].cols/2, Images[0].rows/2);
     
     
     for(int i = 0; i < Images.size(); i++){
         //Mat leftUp, rightUp, leftDown, rightDown;
         
         Mat leftUp = Mat(Images[i], rectLeftUp).clone();
         Mat rightUp = Mat(Images[i], rectRightUp).clone();
         Mat leftDown = Mat(Images[i], rectLeftDown).clone();
         Mat rightDown = Mat(Images[i], rectRightDown).clone();
         
         dividedSet.push_back(leftUp);
         dividedSet.push_back(rightUp);
         dividedSet.push_back(leftDown);
         dividedSet.push_back(rightDown);
         
         returnLabels.push_back(Labels[i]);
         returnLabels.push_back(Labels[i]);
         returnLabels.push_back(Labels[i]);
         returnLabels.push_back(Labels[i]);
         
         
     }
     return dividedSet;
}



double LBPTestProbability(vector<int> trainLabels, vector<Mat> trainImages, vector<int> testLabels, vector<Mat> testImages, int pyramidLevel){
    
    vector<Mat> trainHist;
    vector<Mat> testHist;
    
    vector<int> dividedTrainLabels;
    vector<int> dividedTestLabels;
    
    vector<Mat> dividedTrainImages;
    vector<Mat> dividedTestImages;
    
    if(pyramidLevel == 0){
        trainHist = LBPHist(trainImages,trainLabels);
        testHist = LBPHist(testImages, testLabels);
        //in this case we do not need to use the divided labels, we can use the passed in test and train labels.
    }
    
    if(pyramidLevel == 1){
        dividedTrainImages = DivideFour(trainImages, trainLabels, dividedTrainLabels);
        dividedTestImages = DivideFour(testImages, testLabels, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    if(pyramidLevel == 2){
        
        vector<int> labelTrain1;
        vector<int> labelTest1;
        
        vector<Mat> trainHist1 = DivideFour(trainImages, trainLabels, labelTrain1);
        vector<Mat> testHist1 = DivideFour(testImages, testLabels, labelTest1);
        dividedTrainImages = DivideFour(trainHist1, labelTrain1, dividedTrainLabels);
        dividedTestImages = DivideFour(testHist1, labelTest1, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    if(pyramidLevel == 3){
        
        vector<int> labelTrain1;
        vector<int> labelTrain2;
        vector<int> labelTest1;
        vector<int> labelTest2;
        
        vector<Mat> trainHist1 = DivideFour(trainImages, trainLabels, labelTrain1);
        vector<Mat> testHist1 = DivideFour(testImages, testLabels, labelTest1);
        vector<Mat> trainHist2 = DivideFour(trainHist1, labelTrain1, labelTrain2);
        vector<Mat> testHist2 = DivideFour(testHist1, labelTest1, labelTest2);
        dividedTrainImages = DivideFour(trainHist2, labelTrain2, dividedTrainLabels);
        dividedTestImages = DivideFour(testHist2, labelTest2, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    
    /* Use calcCovarMatrix to calcuate the mean and the covariance Matrix, and use it to construct the single Gaussian. */
    
    double numCorrectCase = 0;
    double totalNumImages = 0;
    //double distance = 100;
    int MatchingIndex = 0;
    
    //int numClusters = 1;
    
    Mat allHistogramHconcat = trainHist[0];
    //Compare for the result.
    
    //EM em_model(numClusters, EM::COV_MAT_DIAGONAL);
    //for(int i = 1; i < trainHist.size(); i++){
      //  vconcat(allHistogramHconcat, trainHist[i], allHistogramHconcat);
    //}
    
   // Mat meanAllHist;
    //Mat covAllHist;
    
   // calcCovarMatrix(allHistogramHconcat, covAllHist, meanAllHist, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    
    //Mat weightMatrix(1, numClusters, CV_32F);
    
    //Mat logLikelihoodsMatrix, labelMatrix, probabilityMatrix;
    
    
    //em_model.trainE(allHistogramHconcat, meanAllHist, vector<Mat>(1, covAllHist), weightMatrix,logLikelihoodsMatrix, labelMatrix, probabilityMatrix);

   
    
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    model->train(trainImages, trainLabels);

    //for(int i = 0; i < trainHist.size(); i++){
        
        for(int j = 0; j < testImages.size(); j++){
            
            int predictedLabel = model->predict(testImages[j]);
            MatchingIndex = predictedLabel;
            
            //if(pyramidLevel == 0){
                if(trainLabels[MatchingIndex] == testLabels[j]){
                    numCorrectCase++;
                }
                totalNumImages++;
            //}
            
            /*if(pyramidLevel > 0){
                if(dividedTrainLabels[MatchingIndex] == dividedTestLabels[j]){
                    numCorrectCase++;
                }
                totalNumImages++;
            }*/
            
        }
    //}
    
    
    double correctRate = numCorrectCase / totalNumImages;
    return correctRate;
    
}



vector<int> LBPConfusionMatrix(vector<int> trainLabels, vector<Mat> trainImages, vector<int> testLabels, vector<Mat> testImages, int pyramidLevel){
    
    vector<Mat> trainHist;
    vector<Mat> testHist;
    
    vector<int> dividedTrainLabels;
    vector<int> dividedTestLabels;
    
    vector<Mat> dividedTrainImages;
    vector<Mat> dividedTestImages;
    
    if(pyramidLevel == 0){
        trainHist = LBPHist(trainImages,trainLabels);
        testHist = LBPHist(testImages, testLabels);
        //in this case we do not need to use the divided labels, we can use the passed in test and train labels.
    }
    
    if(pyramidLevel == 1){
        dividedTrainImages = DivideFour(trainImages, trainLabels, dividedTrainLabels);
        dividedTestImages = DivideFour(testImages, testLabels, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    if(pyramidLevel == 2){
        
        vector<int> labelTrain1;
        vector<int> labelTest1;
        
        vector<Mat> trainHist1 = DivideFour(trainImages, trainLabels, labelTrain1);
        vector<Mat> testHist1 = DivideFour(testImages, testLabels, labelTest1);
        dividedTrainImages = DivideFour(trainHist1, labelTrain1, dividedTrainLabels);
        dividedTestImages = DivideFour(testHist1, labelTest1, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    if(pyramidLevel == 3){
        
        vector<int> labelTrain1;
        vector<int> labelTrain2;
        vector<int> labelTest1;
        vector<int> labelTest2;
        
        vector<Mat> trainHist1 = DivideFour(trainImages, trainLabels, labelTrain1);
        vector<Mat> testHist1 = DivideFour(testImages, testLabels, labelTest1);
        vector<Mat> trainHist2 = DivideFour(trainHist1, labelTrain1, labelTrain2);
        vector<Mat> testHist2 = DivideFour(testHist1, labelTest1, labelTest2);
        dividedTrainImages = DivideFour(trainHist2, labelTrain2, dividedTrainLabels);
        dividedTestImages = DivideFour(testHist2, labelTest2, dividedTestLabels);
        trainHist = LBPHist(dividedTrainImages, dividedTrainLabels);
        testHist = LBPHist(dividedTestImages, dividedTestLabels);
    }
    
    double distance = 100;
    int MatchingIndex = 0;
    vector<int> matchingClass;
    

    
    for(int i = 0; i < trainHist.size(); i++){
        for(int j = 0; j < testHist.size(); j++){
            double currentDistance = compareHist(trainHist[i], testHist[j], CV_COMP_CHISQR);
            if(currentDistance < distance){
                distance = currentDistance;
                MatchingIndex = i;
            }
            if(pyramidLevel == 0){
                if(trainLabels[MatchingIndex] == testLabels[j]){
                    matchingClass.push_back(MatchingIndex);
                }
            }
            
            if(pyramidLevel > 0){
                if(dividedTrainLabels[MatchingIndex] == dividedTestLabels[j]){
                    matchingClass.push_back(MatchingIndex);
                }
            }
            
        }
    }
    
    return matchingClass;

}






