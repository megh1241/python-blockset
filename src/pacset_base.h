#ifndef BASE
#define BASE

#include <vector>
#include <iostream>
#include "pacset_base_model.h"
#include "pacset_factory.h"
#include "utils.h"
#include "config.h"
#include "pacset_rf_classifier.h"
#include "pacset_rf_regressor.h"
#include "pacset_gb_classifier.h"
#include "pacset_gb_regressor.h"

class BlocksetBase{
    //TODO: (Note) Upcaset to Node when serializing to file
        PacsetBaseModel<float, float> *obj;
    public:
	BlocksetBase(){}
	void initRandomForestClassifier();
	void initRandomForestRegressor();
	void initGradientBoostedClassifier();
	void initGradientBoostedRegressor();
	void loadJSONModel(std::string filename);
	void loadBlocksetModel(std::string filename);
	void pack(std::string filename);
	void pack();
	void serialize(std::string filename);
	double predict(std::vector<float> X);
	std::vector<double> predict(std::vector<std::vector<float>> X);
};

#endif
