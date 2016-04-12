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
    
    loadHeadPose(headPoseImages, headPosePath, headPoseId, headPoseTilt, headPosePan,headPoseAnnotation);
    
    
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


