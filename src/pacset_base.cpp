#include <iostream>
#include <string>
#include <cstring>

#include "pacset_base.h"


void BlocksetBase::initRandomForestClassifier(){
        obj = new PacsetRandomForestClassifier<float, float>();
	Config::setConfigItem(std::string("algorithm"), std::string("randomforest"));	
	Config::setConfigItem(std::string("task"), std::string("classification"));	
}
void BlocksetBase::initRandomForestRegressor(){
        obj = new PacsetRandomForestRegressor<float, float>();
	Config::setConfigItem(std::string("algorithm"), std::string("randomforest"));	
	Config::setConfigItem(std::string("task"), std::string("regression"));	
}
void BlocksetBase::initGradientBoostedClassifier(){
        obj = new PacsetGradientBoostedClassifier<float, float>();
	Config::setConfigItem(std::string("algorithm"), std::string("gradientboost"));	
	Config::setConfigItem(std::string("task"), std::string("classification"));	
}
void BlocksetBase::initGradientBoostedRegressor(){
        obj = new PacsetGradientBoostedRegressor<float, float>();
	Config::setConfigItem(std::string("algorithm"), std::string("gradientboost"));	
	Config::setConfigItem(std::string("task"), std::string("regression"));	
}

void BlocksetBase::loadJSONModel(std::string filename){
	Config::setConfigItem(std::string("modelfilename"), filename);
	obj->loadModel();
}

void BlocksetBase::loadBlocksetModel(std::string filename){
	Config::setConfigItem(std::string("modelfilename"), filename);
	obj->deserialize();
}

void BlocksetBase::pack(std::string filename){
	Config::setConfigItem(std::string("modelfilename"), filename);
	Config::setConfigItem(std::string("mode"), std::string("pack"));
	Config::setConfigItem(std::string("modelfilename"), filename);
	if(Config::getValue(std::string("numbins")) == std::string("notfound"))
		Config::setConfigItem(std::string("numbins"), std::string("1"));
	if(Config::getValue(std::string("blocksize")) == std::string("notfound"))
		Config::setConfigItem(std::string("blocksize"), std::string("128"));
	if(Config::getValue(std::string("layout")) == std::string("notfound"))
		Config::setConfigItem(std::string("layout"), std::string("binstatblock"));
	if(Config::getValue(std::string("interleave")) == std::string("notfound"))
		Config::setConfigItem(std::string("interleave"), std::string("2"));
	Config::setConfigItem(std::string("numthreads"), std::string("1")) ;
	
	obj->loadModel();
        obj->pack();
}

void BlocksetBase::pack(){
	Config::setConfigItem(std::string("mode"), std::string("pack"));
	if(Config::getValue(std::string("numbins")) == std::string("notfound"))
		Config::setConfigItem(std::string("numbins"), std::string("1"));
	if(Config::getValue(std::string("blocksize")) == std::string("notfound"))
		Config::setConfigItem(std::string("blocksize"), std::string("128"));
	if(Config::getValue(std::string("layout")) == std::string("notfound"))
		Config::setConfigItem(std::string("layout"), std::string("binstatblock"));
	if(Config::getValue(std::string("interleave")) == std::string("notfound"))
		Config::setConfigItem(std::string("interleave"), std::string("2"));
	Config::setConfigItem(std::string("numthreads"), std::string("1")) ;
        obj->pack();
}

void BlocksetBase::serialize(std::string filename){
	Config::setConfigItem(std::string("packfilename"), filename);
	Config::setConfigItem(std::string("metadatafilename"), filename + "metadata.txt");
	
	if(Config::getValue(std::string("format")) == std::string("notfound"))
		Config::setConfigItem(std::string("format"), std::string("binary"));

	obj->serialize();
}

std::vector<double> BlocksetBase::predict(std::vector<std::vector<float>> X){
	std::vector<double> preds;
        std::vector<double> predi;
        obj->predict(X, preds, predi, true);
	return predi;
}

double BlocksetBase::predict(std::vector<float> X){
	std::vector<double> preds;
        std::vector<double> predi;
	std::vector<std::vector<float>> obs;
	obs.push_back(X);
        obj->predict(obs, preds, predi, true);
	return predi[0];
}

/*void BlocksetBase::predict(std::string test_filename){
        std::vector<std::vector<float>> test_vec;    
       if ((Config::getValue("algorithm") == std::string("randomforest")) && (Config::getValue("task") == std::string("classification"))){
            std::vector<int> preds;
            std::vector<int> predi;
            std::vector<int> lab;
            //Perform prediction
            std::cout<<"loading test data\b";
            loadTestData(test_vec, lab);
                std::cout<<"test data loaded\n";
            obj->predict(test_vec, preds, predi, true);
            std::cout<<"predicted\n";
            //Compute accuracy
            std::cout<<"Size of predicted: "<<predi.size();
            for (auto i: predi)
                std::cout<<i<<", ";
            std::cout<<"\n";

            double acc = getAccuracy(predi, lab);
            std::cout<<"Accuracy: "<<acc<<"\n";
        }
        //Load test data from file
        else{
            if(Config::getValue("task") == std::string("regression")){
                std::cout<<"Not classification!\n";
                std::vector<double> preds;
                std::vector<double> predi;
                std::vector<double> lab;
                //Perform prediction
                loadTestData(test_vec, lab);
                obj->predict(test_vec, preds, predi, true);
                std::cout<<"predicted\n";
                //Compute accuracy
                double acc = getAccuracy(predi, lab);
                std::cout<<"Accuracy: "<<acc<<"\n";
            }
            else{
                std::vector<double> preds;
                std::vector<double> pred_d;
                std::vector<int> pred_i;
                std::vector<int> lab;
                //Perform prediction
                loadTestData(test_vec, lab);
                obj->predict(test_vec, preds, pred_d, true);
                for(auto i: pred_d)
                        pred_i.push_back((int)i);
                std::cout<<"predicted\n";
                double acc = getAccuracy(pred_i, lab);
                std::cout<<"Accuracy: "<<acc<<"\n";

            }
        }
}
*/
