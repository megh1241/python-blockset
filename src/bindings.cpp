//#include "config.h"
#include "node.h"
#include "stat_node.h"
#include "MemoryMapped.h"
#include "pacset_base.h"
#include "utils.h"
#include "packer.h"
#include "json_reader.h"
#include "pacset_rf_classifier.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <string>

namespace py = pybind11;


/*class PyPacsetBaseModel : public PacsetBaseModel<float, int> {
	using PacsetBaseModel<float, int>::PacsetBaseModel;
	void setMembers(const std::vector<int> &bin_sizes,
                const std::vector<int> &bin_node_sizes,
                const std::vector<std::vector<int>> &bin_start)override { PYBIND11_OVERRIDE_PURE(void, PacsetBaseModel, setMembers, bin_sizes, bin_node_sizes, bin_start); }
}*/	


PYBIND11_MODULE(pyblockset, m) {
m.def("logit2", &logit2);
 m.def("getAccuracy", py::overload_cast<const std::vector<int>&, const std::vector<int>&>(&getAccuracy));
 m.def("loadTestData", py::overload_cast<std::vector<std::vector<float>>&, std::vector<int>&>(&loadTestData) );
 m.def("loadTestData", py::overload_cast<std::vector<std::vector<float>>&, std::vector<double>&>(&loadTestData));
 m.def("getAccuracy", py::overload_cast<const std::vector<int>&, const std::vector<int>&>(&getAccuracy));
 m.def("getAccuracy", py::overload_cast<const std::vector<double>&, const std::vector<double>&>(&getAccuracy));
py::class_<Config>(m, "Config")
    	.def("getInstance", &Config::getInstance)
    	.def("getValue", &Config::getValue)
    	.def("setConfigItem", &Config::setConfigItem);


py::class_<BlocksetBase>(m, "BlocksetBase")
	.def(py::init<>())
	.def("initRandomForestClassifier", &BlocksetBase::initRandomForestClassifier)
	.def("initRandomForestRegressor", &BlocksetBase::initRandomForestRegressor)
	.def("initGradientBoostedClassifier", &BlocksetBase::initGradientBoostedClassifier)
	.def("initGradientBoostedRegressor", &BlocksetBase::initGradientBoostedRegressor)
	.def("loadJSONModel", &BlocksetBase::loadJSONModel)
	.def("loadBlocksetModel", &BlocksetBase::loadBlocksetModel)
	.def("pack", py::overload_cast<std::string>(&BlocksetBase::pack))
	.def("pack", py::overload_cast<>(&BlocksetBase::pack))
	.def("serialize", &BlocksetBase::serialize)
	//.def("predictLabel", &BlocksetBase::predictLabel);
	.def("predictLabelClassification", py::overload_cast<std::vector<float>> (&BlocksetBase::predictLabelClassification))
	.def("predictLabelClassification", py::overload_cast<std::vector<std::vector<float>>> (&BlocksetBase::predictLabelClassification))
	.def("predictLabelRegression", py::overload_cast<std::vector<float>> (&BlocksetBase::predictLabelRegression))
	.def("predictLabelRegression", py::overload_cast<std::vector<std::vector<float>>> (&BlocksetBase::predictLabelRegression));
}
