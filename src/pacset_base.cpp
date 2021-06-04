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
	if(Config::getValue(std::string("numbins")) == std::string("notfound"))
		Config::setConfigItem(std::string("numbins"), std::string("1"));
	Config::setConfigItem(std::string("modelfilename"), filename);
	obj->loadModel();
}

void BlocksetBase::loadBlocksetModel(std::string filename){
	Config::setConfigItem(std::string("modelfilename"), filename);
	Config::setConfigItem(std::string("metadatafilename"), filename + "metadata.txt");
	Config::setConfigItem(std::string("numthreads"), std::string("1")) ;
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
		Config::setConfigItem(std::string("layout"), std::string("bindfs"));
	if(Config::getValue(std::string("interleave")) == std::string("notfound"))
		Config::setConfigItem(std::string("interleave"), std::string("1"));
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
		Config::setConfigItem(std::string("layout"), std::string("binblockstat"));
	if(Config::getValue(std::string("interleave")) == std::string("notfound"))
		Config::setConfigItem(std::string("interleave"), std::string("1"));
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

double BlocksetBase::predictLabel(std::vector<float> X){
	std::vector<int> preds;
        std::vector<int> predi;
	std::vector<std::vector<float>> obs;
	obs.push_back(X);
        
	obj->predict(obs, preds, predi, true);
	return predi[0];
}

