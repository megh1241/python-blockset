#ifndef PACSET_RF_CLASS
#define PACSET_RF_CLASS

#include <vector>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include <random>
#include <stdint.h>
#include <cstdint>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include "pacset_base_model.h"
#include "packer.h"
#include "config.h"
#include "json_reader.h"
#include "utils.h"
//#include "node.h"
#include "MemoryMapped.h"

#define LAT_LOGGING 2
#define BLOCK_LOGGING 1
#define BLOCK_SIZE 2048

using std::uint32_t;

const int blob_size = 10000;

template <typename T, typename F>
class PacsetRandomForestClassifier: public PacsetBaseModel<T, F> {
	public:

		inline void setMembers(const std::vector<int> &bin_sizes,
				const std::vector<int> &bin_node_sizes,
				const std::vector<std::vector<int>> &bin_start){

			PacsetBaseModel<T, F>::bin_sizes.clear();
			std::copy(bin_sizes.begin(), bin_sizes.end(), back_inserter(PacsetBaseModel<T, F>::bin_sizes)); 
			std::copy(bin_node_sizes.begin(), bin_node_sizes.end(), back_inserter(PacsetBaseModel<T, F>::bin_node_sizes)); 
			for (auto i: bin_start)
				PacsetBaseModel<T, F>::bin_start.push_back(i);  
		
		}

		inline void setBinNodeSizes(int pos, int siz){
			PacsetBaseModel<T, F>::bin_node_sizes[pos] = siz;
		}


		inline void loadModel() {
			JSONReader<T, F> J;
			J.convertSklToBinsRapidJson(PacsetBaseModel<T, F>::bins, 
					PacsetBaseModel<T, F>::bin_sizes, 
					PacsetBaseModel<T, F>::bin_start,
					PacsetBaseModel<T, F>::bin_node_sizes);
			
		}

		inline void pack(){
			std::string layout = Config::getValue("layout");

			auto bin = PacsetBaseModel<T, F>::bins[0];
			int num_bins = std::stoi(Config::getValue("numthreads"));
			for(int i=0; i<num_bins; ++i){
				Packer<T, F> packer_obj(layout);
				if(Config::getValue("intertwine") != std::string("notfound"))
					packer_obj.setDepthIntertwined(std::atoi(Config::getValue("intertwine").c_str()));
				//should pack in place
				packer_obj.pack(PacsetBaseModel<T, F>::bins[i], 
						PacsetBaseModel<T, F>::bin_sizes[i],
						PacsetBaseModel<T, F>::bin_start[i] 
					       );
				setBinNodeSizes(i, PacsetBaseModel<T, F>::bins[i].size());
			}
		}



		inline int mmapAndPredict(const std::vector<T>& observation, std::vector<int>& preds, int obsnum, bool mmap) {
			int num_classes = std::stoi(Config::getValue("numclasses"));
			int num_threads = std::stoi(Config::getValue("numthreads"));
			int num_bins = PacsetBaseModel<T, F>::bin_sizes.size();
			std::string modelfname = Config::getValue("modelfilename");
			Node<T, F> *data;
			MemoryMapped mmapped_obj(modelfname.c_str(), 0);
			//data = (Node<T, F>*)mmapped_obj.getData();
			if (mmap){
				data = (Node<T, F>*)mmapped_obj.getData();
			}else{
				FILE* fp;
				fp = fopen(modelfname.c_str(),"rb");
				std::vector<Node<T, F>> bin_elements;
				while(!feof(fp)){
				       Node<T, F> node;
				       fread((char*)&node,sizeof(node),1,fp);
				       bin_elements.push_back(node);
				}
				data = bin_elements.data();
			}

			std::unordered_set<int> blocks_accessed;
			int block_offset = 0;
			int offset = 0;
			std::vector<int> offsets;
			int curr_offset = 0;
			for (auto val: PacsetBaseModel<T, F>::bin_node_sizes){
				offsets.push_back(curr_offset);
				curr_offset += val;
			}
//#pragma omp parallel for num_threads(num_threads)
			for(int bin_counter=0; bin_counter<num_bins; ++bin_counter){
				int block_number = 0;
				Node<T, F> *bin  = data + offsets[bin_counter];

				std::vector<int> curr_node(PacsetBaseModel<T, F>::bin_node_sizes[bin_counter]);
				int i, feature_num=0, number_not_in_leaf=0;
				T feature_val;
				int siz = PacsetBaseModel<T, F>::bin_sizes[bin_counter];
				for(i=0; i<siz; ++i){
					curr_node[i] = PacsetBaseModel<T, F>::bin_start[bin_counter][i];
					__builtin_prefetch(&bin[curr_node[i]], 0, 3);
				}
				do{
					number_not_in_leaf = 0;
					for( i=0; i<siz; ++i){
						if(bin[curr_node[i]].isInternalNodeFront()){
							feature_num = bin[curr_node[i]].getFeature();
							feature_val = observation[feature_num];
							curr_node[i] = bin[curr_node[i]].nextNode(feature_val);
							__builtin_prefetch(&bin[curr_node[i]], 0, 3);
							++number_not_in_leaf;
						}
					}
				}while(number_not_in_leaf);

				for(i=0; i<siz; ++i){
//#pragma omp atomic update
					++preds[bin[curr_node[i]].getClass()];
				}

//#pragma omp critical
				block_offset += PacsetBaseModel<T, F>::bin_node_sizes[bin_counter];
			}
			mmapped_obj.close();
			return 0;
		}
		std::pair<int, int> transformIndex(int node_number, int bin_start_list, int bin_number){
			return std::make_pair(bin_start_list + node_number/blob_size, node_number % blob_size);
		}	
		
		inline void predict(const std::vector<std::vector<T>> &observations,
                	std::vector<double> &preds, std::vector<double> &result, bool mmap){}
		
		inline void predict(const std::vector<std::vector<T>>& observation, 
				std::vector<int>& preds, std::vector<int>&results, bool mmap) {

			//Predicts the class for a vector of observations
			//By calling predict for a single observation and 
			//tallying the observations
			//
			double cumi_time = 0;
			int num_classes = std::stoi(Config::getValue("numclasses"));
			int num_bins; 
			std::vector<double> elapsed_arr;
			int batchsize = 1;

			for(int i=0; i<num_classes; ++i){
				preds.push_back(0);
			}

			int max = -1;
			int maxid = -1;
			int blocks;
			int ct=0;
			std::vector<int> num_blocks;
			//writeGarbage();
			for(auto single_obs : observation){
				auto start = std::chrono::steady_clock::now();
				blocks = mmapAndPredict(single_obs, preds, ct+1, mmap);

				num_blocks.push_back(blocks);
				//TODO: change
				for(int i=0; i<num_classes; ++i){
					if(preds[i]>max){
						maxid = i;
						max = preds[i];
					}
				}
				int count = std::count(std::begin(preds), std::end(preds), max);
				auto end = std::chrono::steady_clock::now();
				ct++;
				results.push_back(maxid); 
				std::fill(preds.begin(), preds.end(), 0);
				max = -1;
				maxid = -1;
			}

		}


		inline void serializeMetadata() {
			//Write the metadata needed to reconstruct bins and for prediction
			auto bins = PacsetBaseModel<T, F>::bins;
			int num_classes = std::stoi(Config::getValue("numclasses"));
			int num_bins = bins.size();
			std::vector<int> bin_sizes = PacsetBaseModel<T, F>::bin_sizes;
			std::vector<int> bin_node_sizes = PacsetBaseModel<T, F>::bin_node_sizes;
			std::vector<std::vector<int>> bin_start  = PacsetBaseModel<T, F>::bin_start;
			std::string filename;

			std::string modelfname = Config::getValue("metadatafilename");

			if(modelfname != std::string("notfound"))
				filename = modelfname;
			else
				filename = "metadata.txt";
			std::fstream fout;
			fout.open(filename, std::ios::out );

			//Number of classes
			fout<<num_classes<<"\n";

			//Number of bins
			fout<<num_bins<<"\n";

			//Number of trees in each bin
			for(auto i: bin_sizes){
				fout<<i<<"\n";
			}

			//Number of nodes in each bin
			for(auto i: bin_node_sizes){
				fout<<i<<"\n";
			}

			//start position of each bin
			for(auto i: bin_start){
				for(auto tree_start: i){
					fout<<tree_start<<"\n";
				}
			}
			fout.close();

		}

		inline void serializeModelBinary() {
			auto bins = PacsetBaseModel<T, F>::bins;
			std::string modelfname = Config::getValue("packfilename");
			std::string filename;
			if(modelfname != std::string("notfound"))
			    filename = modelfname;
			else
			    filename = "packedmodel.bin";

			//Write the nodes
			std::fstream fout;
			fout.open(filename, std::ios::binary | std::ios::out );
			Node<T, F> node_to_write;
			for(auto bin: bins){
				for(auto node: bin){
					node_to_write = node;
					fout.write((char*)&node_to_write, sizeof(node_to_write));
				}
			}
			fout.close();
		}
	
		inline void serializeModelText(){
			auto bins = PacsetBaseModel<T, F>::bins;
			std::string modelfname = Config::getValue("packfilename");
			std::string filename;

			if(modelfname != std::string("notfound"))
			    filename = modelfname;
			else
			    filename = "packedmodel.txt";

			//Write the nodes
			std::fstream fout;
			fout.open(filename,  std::ios::out );
			for(auto bin: bins){
			    for(auto node: bin){
				fout<<node.getLeft()<<", "<<node.getRight()
				<<", "<<node.getFeature()<<", "<<node.getThreshold()<<"\n";
			    }
			}
			fout.close();
		}


		inline void serialize() {
			std::string format = Config::getValue("format");
			serializeMetadata();
			if(format == std::string("binary")){
			    serializeModelBinary();
			}
			else {
			    serializeModelText();
			}
		}

		inline void deserialize(){
			//Write the metadata needed to reconstruct bins and for prediction
			//TODO: change filename
			int num_classes, num_bins;
			std::string filename = Config::getValue("metadatafilename");
			//std::string filename = "metadata.txt";
			std::fstream f;
			f.open(filename, std::ios::in );

			//Number of classes
			f>>num_classes;
			Config::setConfigItem("numclasses", std::to_string(num_classes));
			//Number of bins
			f>>num_bins;
			Config::setConfigItem("numthreads", std::to_string(num_bins));
			std::vector<int> num_trees_bin;
			std::vector<int> num_nodes_bin;
			std::vector<std::vector<int>> bin_tree_start;
			int val;
			//Number of trees in each bin
			for(int i=0; i<num_bins; ++i){
				f>>val;
				num_trees_bin.push_back(val);
			}

			//Number of nodes in each bin
			for(int i=0; i<num_bins; ++i){
				f>>val;
				num_nodes_bin.push_back(val);
			}

			std::vector<int> temp;
			//start position of each bin
			for(int i=0; i<num_bins; ++i){
				for(int j=0; j<num_trees_bin[i]; ++j){
					f>>val;
					temp.push_back(val); 
				}
				bin_tree_start.push_back(temp);
				temp.clear();
			}
			f.close();
			setMembers(num_trees_bin, num_nodes_bin, bin_tree_start);

		}

		/*
		   inline void deserialize() {
		   readMetadata();
		   std::string modelfname = Config::getValue("modelfilename");
		   MemoryMapped mmapped_obj(modelfname, 0);
		   Node<T, F> *data = (Node<T, F>*)mmapped_obj.getData();
//TODO: make this a separate predict bin
std::vector<std::vector<Node<T, F>>> bins;   
int pos = 0;
for (auto i: PacsetBaseModel<T, F>::bin_node_sizes){
std::vector<StatNode<T, F>> nodes;
nodes.assign(data+pos, data+pos+i);
bins.push_back(nodes);
pos = i;
}

}
*/
};

#endif
