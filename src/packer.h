#ifndef PACKER_H
#define PACKER_H

#include <iostream>
#include "stat_node.h"
#include "node.h"
#include "config.h"
#include <algorithm>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <cmath>

template <typename T, typename F>
class Packer{

	int depth_intertwined;
	std::string layout;
	std::vector<StatNode<T, F>> finalbin;
	std::map<int, int> node_to_index;
	std::map<int, int> subtree_count_map;
	std::map<int, int> subtree_class_map;
	std::map<int, int> subtree_nodecount_map;
	std::map<int, int> class_numST_map;

	private:
	//Note: contains helper functions (i.e common routines used for packing)

	//Returns a queue containing the roots of subtrees below the BIN
	//after packing the BIN in an interleaved fashion
	//
	//
	inline std::deque<StatNode<T, F>> packBinHelperRFReg(
			std::vector<StatNode<T, F>>&bin,
			const int num_trees_in_bin,
			std::vector<int> &bin_start) {
		int num_nodes_process = std::pow(2, depth_intertwined) - 1;
		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		std::deque <StatNode<T, F>> temp_q;
		std::deque <StatNode<T, F>> bin_q;
		std::deque <StatNode<T, F>> bin_q_left;
		std::deque <StatNode<T, F>> bin_q_right;

		for(int i=0; i<num_trees_in_bin; ++i){
			temp_q.push_back(bin[bin_start[i]]);
			//trees are interleaved so the starting nodes are adjacent
			//but after the class nodes
			bin_start[i] = num_classes + i;
		}

		// Intertwined levels
		int curr_level = 0;
		int pos_in_level = 0;

		while(curr_level < num_nodes_process * num_trees_in_bin) {
			auto ele = temp_q.front();
			temp_q.pop_front();
			if(ele.getID()>= 0) {
				finalbin.push_back(ele);
				node_to_index.insert(std::pair<int, int>(ele.getID(), finalbin.size()-1));
				if(ele.getLeft() > -1){
					bin_q_left.push_back(bin[ele.getLeft()]);
					bin_q_right.push_back(bin[ele.getRight()]);
				}else{
					bin_q_left.push_back(genBlankNode());
					bin_q_right.push_back(genBlankNode());
				}
			}
			else {

				bin_q_left.push_back(genBlankNode());
				bin_q_right.push_back(genBlankNode());
			}

			if(pos_in_level == num_trees_in_bin - 1){
				while(!bin_q_left.empty()){
					auto ele = bin_q_left.front();
					temp_q.push_back(ele);
					bin_q_left.pop_front();
				}
				while(!bin_q_right.empty()){
					auto ele = bin_q_right.front();
					temp_q.push_back(ele);
					bin_q_right.pop_front();
				}
				pos_in_level = 0;
			}
			else
				pos_in_level++;

			curr_level++;
			if (temp_q.size() == 0)
				break;
		}

		while(!temp_q.empty()){
			auto ele = temp_q.front();
			if(ele.getID() >= 0)
				bin_q.push_back(ele);
			temp_q.pop_front();
		}
		return bin_q;
	}
	inline std::deque<StatNode<T, F>> packBinHelperGB(
			std::vector<StatNode<T, F>>&bin,
			const int num_trees_in_bin,
			std::vector<int> &bin_start) {
		int num_nodes_process = std::pow(2, depth_intertwined) - 1;
		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		std::deque <StatNode<T, F>> temp_q;
		std::deque <StatNode<T, F>> bin_q;
		std::deque <StatNode<T, F>> bin_q_left;
		std::deque <StatNode<T, F>> bin_q_right;

		for(int i=0; i<num_trees_in_bin; ++i){
			temp_q.push_back(bin[bin_start[i]]);
			bin_start[i] = num_classes + i;
		}

		// Intertwined levels
		int curr_level = 0;
		int pos_in_level = 0;

		while(curr_level < num_nodes_process * num_trees_in_bin) {
			auto ele = temp_q.front();
			temp_q.pop_front();
			if(ele.getID()>= 0) {
				finalbin.push_back(ele);
				node_to_index.insert(std::pair<int, int>(ele.getID(), finalbin.size()-1));
				if(ele.getLeft() > -1){
					bin_q_left.push_back(bin[ele.getLeft()]);
					bin_q_right.push_back(bin[ele.getRight()]);
				}else{
					bin_q_left.push_back(genBlankNode());
					bin_q_right.push_back(genBlankNode());
				}
			}
			else {
				bin_q_left.push_back(genBlankNode());
				bin_q_right.push_back(genBlankNode());
			}
			if(pos_in_level == num_trees_in_bin - 1){
				while(!bin_q_left.empty()){
					auto ele = bin_q_left.front();
					temp_q.push_back(ele);
					bin_q_left.pop_front();
				}
				while(!bin_q_right.empty()){
					auto ele = bin_q_right.front();
					temp_q.push_back(ele);
					bin_q_right.pop_front();
				}
				pos_in_level = 0;
			}
			else
				pos_in_level++;

			curr_level++;
			if (temp_q.size() == 0)
				break;
		}

		while(!temp_q.empty()){
			auto ele = temp_q.front();
			if(ele.getID() > -5)
				bin_q.push_back(ele);
			temp_q.pop_front();
		}
		return bin_q;
	}

	inline std::deque<StatNode<T, F>> packBinHelperRFClass(
			std::vector<StatNode<T, F>>&bin, 
			const int num_trees_in_bin, 
			std::vector<int> &bin_start) {
		int num_nodes_process = std::pow(2, depth_intertwined) - 1;
		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		std::deque <StatNode<T, F>> temp_q;
		std::deque <StatNode<T, F>> bin_q;
		std::deque <StatNode<T, F>> bin_q_left;
		std::deque <StatNode<T, F>> bin_q_right;

		for(int i=0; i<num_trees_in_bin; ++i){
			temp_q.push_back(bin[bin_start[i]]);
			//finalbin.push_back(bin[bin_start[i]]);
			//node_to_index.insert(std::pair<int, int>(bin[bin_start[i]].getID(), finalbin.size()-1));
			//trees are interleaved so the starting nodes are adjacent
			//but after the class nodes
			bin_start[i] = num_classes + i;
		}

		// Intertwined levels
		int curr_level = 0; 
		int pos_in_level = 0;

		while(curr_level < num_nodes_process * num_trees_in_bin) {
			auto ele = temp_q.front();
			temp_q.pop_front();
			if(ele.getID()>= 0) {
				finalbin.push_back(ele);
				node_to_index.insert(std::pair<int, int>(ele.getID(), finalbin.size()-1));
				bin_q_left.push_back(bin[ele.getLeft()]); 
				bin_q_right.push_back(bin[ele.getRight()]); 
			}
			else {
				bin_q_left.push_back(bin[ele.getRight()]); 
				bin_q_right.push_back(bin[ele.getRight()]); 
			}

			if(pos_in_level == num_trees_in_bin - 1){
				while(!bin_q_left.empty()){
					auto ele = bin_q_left.front();
					temp_q.push_back(ele);
					bin_q_left.pop_front();
				}
				while(!bin_q_right.empty()){
					auto ele = bin_q_right.front();
					temp_q.push_back(ele);
					bin_q_right.pop_front();
				}
				pos_in_level = 0;
			}
			else
				pos_in_level++;

			curr_level++;
			if (temp_q.size() == 0)
				break;
		}

		while(!temp_q.empty()){
			auto ele = temp_q.front();
			if(ele.getID() >= 0)
				bin_q.push_back(ele);
			temp_q.pop_front();
		}
		return bin_q;
	}

	inline void packSubtreeBFSHelper(
			std::vector<StatNode<T, F>>&bin, 
			const int num_trees_in_bin, 
			std::vector<int> &bin_start, 
			std::deque<StatNode<T, F>> &bin_q) { 
		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		while(!bin_q.empty()){
			std::deque<StatNode<T, F>> bin_st;
			auto ele = bin_q.front();
			bin_q.pop_front();
			bin_st.push_back(ele);
			while(!bin_st.empty()){
				auto ele = bin_st.front();
				bin_st.pop_front(); 
				finalbin.push_back(ele);
				node_to_index.insert(std::pair<int, int>(ele.getID(), finalbin.size()-1));
				if((ele.getLeft() < num_classes) && (ele.getRight() < num_classes))
					continue;

				else if(ele.getLeft() < num_classes)
					bin_st.push_back(bin[ele.getRight()]);

				else if(ele.getRight() < num_classes)
					bin_st.push_back(bin[ele.getLeft()]); 

				else {
					bin_st.push_back(bin[ele.getLeft()]); 
					bin_st.push_back(bin[ele.getRight()]); 
				}
			}
		}		
		if (bin_start.size() > 0)
			bin_start.clear();
		int node_count=0;	
		for (auto node: finalbin){
			if (node.getDepth() == 0 && node.getID()>= num_classes)
				bin_start.push_back(node_count);
			node_count++;
		}
	}


	inline void packSubtreeDFSHelper(
			std::vector<StatNode<T, F>>&bin, 
			const int num_trees_in_bin, 
			std::vector<int> &bin_start, 
			std::deque<StatNode<T, F>> &bin_q) { 

		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		while(!bin_q.empty()){
			std::deque<StatNode<T, F>> bin_st;
			auto ele = bin_q.front();
			bin_q.pop_front();
			bin_st.push_back(ele);
			while(!bin_st.empty()){
				auto ele = bin_st.front();
				bin_st.pop_front(); 
				finalbin.push_back(ele);
				node_to_index.insert(std::pair<int, int>(ele.getID(), 
							finalbin.size()-1));
				if((ele.getLeft() < num_classes) && 
						(ele.getRight() < num_classes))
					continue;

				else if(ele.getLeft() < num_classes)
					bin_st.push_front(bin[ele.getRight()]);

				else if(ele.getRight() < num_classes)
					bin_st.push_front(bin[ele.getLeft()]); 

				else {
					if(layout.find(std::string("stat")) != std::string::npos ||
							layout.find(std::string("weighted")) != std::string::npos){

						if(bin[ele.getLeft()].getCardinality() < 
								bin[ele.getRight()].getCardinality()){
							bin_st.push_front(bin[ele.getLeft()]); 
							bin_st.push_front(bin[ele.getRight()]);
						}
						else{
							bin_st.push_front(bin[ele.getRight()]);
							bin_st.push_front(bin[ele.getLeft()]); 
						}
					}
					else{
						bin_st.push_front(bin[ele.getLeft()]); 
						bin_st.push_front(bin[ele.getRight()]);
					}
				}
			}
		}	
		if (bin_start.size() > 0)
			bin_start.clear();
		int node_count=0;	
		for (auto node: finalbin){
			if (node.getDepth() == 0 && node.getID()>= num_classes)
				bin_start.push_back(node_count);
			node_count++;
		}
	}


	StatNode<T, F> popMaxCardEle(std::deque<StatNode<T, F>> &bin_st){
		int max = -1;
		int positer = 0;
		int ecount = 0;

		StatNode<T, F> ii ;
		StatNode<T, F> ele;
		for(int i=0; i<bin_st.size(); ++i) {
			if((bin_st[i].getCardinality() > max) || (bin_st[i].getCardinality() == max && bin_st[i].getDepth() < ele.getDepth())){
				max = bin_st[i].getCardinality();
				positer = ecount;
				ele = bin_st[i];
			}
			ecount++;
		}
		bin_st.erase(bin_st.begin() + positer, bin_st.begin() + positer +1);
		return ele;
	}



	inline bool compCard(StatNode<T, F> &node1, StatNode<T, F> &node2){
		return node1.getCardinality() > node2.getCardinality();
	}



	inline StatNode<T, F> genBlankNode(){
		auto a =  StatNode<T, F>(-1, -1, -1, -1, -1); 
		a.setDepth(-1);
		a.setSTNum(-1);
		a.setID(-5);
		return a;
	}


	inline std::vector<StatNode<T, F>> extract_delete_ST(std::vector<StatNode<T, F>>&finalbin, const int &st_num){
		std::vector<StatNode<T, F>> nodes;
		while(1){
			if(finalbin.back().getSTNum() != st_num)
				break;
			nodes.push_back(finalbin.back());
			finalbin.pop_back();
		}
		return nodes;
	}


	inline void packSubtreeBlockwiseHelper (
			std::vector<StatNode<T, F>>&bin, 
			const int num_trees_in_bin, 
			std::vector<int> &bin_start, 
			std::deque<StatNode<T, F>> &bin_q,
			bool class_flag ) { 

		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		int block_size = std::atoi(Config::getValue("blocksize").c_str());

		std::vector<int> class_vector(num_classes, 0);
		int num_blank_nodes = 0;
		int actual_pos = finalbin.size();
		int initial_pos_in_block = actual_pos;
		int actual_pos_boundary = ( actual_pos / block_size + 1)*block_size;
		int subtree_end_id = -2;

		std::sort(bin_q.begin(), bin_q.end(), [this](auto l, auto r){return compCard(l, r);} );


		int subtree_count = 0;

		subtree_count_map[0] = 0;
		int st_flag = 0;
		int leaf_encountered = 0;
		int start_flag = 0;
		int pos_in_block = 0;
		int subtree_num = -1;
		pos_in_block = (finalbin.size()-1) % block_size;

		int subtree_flag = 0;
		while(!bin_q.empty()){
			//If we are at the beginning of a block, then we pop from the back of the queue (i.e we take a new "tree")
			auto ele = bin_q.front();
			if(pos_in_block == block_size-1){
				if(subtree_flag == 0) {
					ele = bin_q.front();
					bin_q.pop_front();
					subtree_flag = 1;
				}
				else{
					subtree_flag++;
					int max = -1;
					int positer = 0;
					int ecount = 0;

					for(auto ii: bin_q) {
						if(ii.getCardinality() > max){
							max = ii.getCardinality();
							positer = ecount;
							ele = ii;
						}
						ecount++;
					}
					bin_q.erase(bin_q.begin() + positer, bin_q.begin() + positer +1);
				}
			}
			else{
				ele = bin_q.front();
				bin_q.pop_front();
			}
			finalbin.push_back(ele);
			pos_in_block = (pos_in_block + 1)%block_size;
			node_to_index.insert(std::pair<int, int>(ele.getID(), finalbin.size()-1));        

			if((ele.getLeft() < num_classes) && 
					(ele.getRight() < num_classes))
				continue;
			else if(ele.getLeft() < num_classes)
				bin_q.push_front(bin[ele.getRight()]);

			else if(ele.getRight() < num_classes)
				bin_q.push_front(bin[ele.getLeft()]); 

			else {
				if(layout.find(std::string("stat")) != std::string::npos ||
						layout.find(std::string("weighted")) != std::string::npos){

					if(bin[ele.getLeft()].getCardinality() < 
							bin[ele.getRight()].getCardinality()){
						bin_q.push_front(bin[ele.getLeft()]); 
						bin_q.push_front(bin[ele.getRight()]);
					}
					else{
						bin_q.push_front(bin[ele.getRight()]);
						bin_q.push_front(bin[ele.getLeft()]); 
					}
				}
				else{
					bin_q.push_front(bin[ele.getLeft()]); 
					bin_q.push_front(bin[ele.getRight()]);
				}
			}

		}
		//Populate node index map and bin starts
		//TODO: replace
		std::string algorithm = Config::getValue("algorithm");
		if(algorithm == std::string("gradientboost")){
			if (node_to_index.size() > 0)
				node_to_index.clear();
		}
		if (bin_start.size() > 0)
			bin_start.clear();
		int node_count = 0;
		for (auto node: finalbin){
			if (node.getDepth() == 0 && node.getID()>= num_classes)
				bin_start.push_back(node_count);
			if(algorithm == std::string("gradientboost"))
				node_to_index.insert(std::pair<int, int>(node.getID(), node_count));
			node_count++;
		}
	}




	public:

	Packer(): layout(std::string("bfs")) {depth_intertwined = 0;};
	Packer(std::string layout_str): layout(layout_str){depth_intertwined = 0;}
	Packer(int depth, std::string layout_str): 
		depth_intertwined(depth), layout(layout_str){}


	inline void setDepthIntertwined(const int depth){
		depth_intertwined = depth;
	}


	inline void pack(std::vector<StatNode<T, F>>&bin, 
			const int num_trees_in_bin, std::vector<int> &bin_start){

		const std::string bin_string = "bin";
		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		for(int i=0; i<num_classes; ++i){
			finalbin.push_back(bin[i]);
		}
		if(layout.find(bin_string) != std::string::npos){
			PackLayoutWithBin(bin, num_trees_in_bin, bin_start);
		}
		else{
			PackRegularLayout(bin, num_trees_in_bin, bin_start);
		}
	}


	inline void PackLayoutWithBin(std::vector<StatNode<T, F>> &bin,
			const int num_trees_in_bin, std::vector<int> &bin_start){

		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		std::string algorithm = Config::getValue("algorithm");
		std::string task = Config::getValue("task");
		//Interleaved BIN
		std::deque<StatNode<T, F>> bin_q;

		if (algorithm == std::string("randomforest") && task == std::string("classification")){
			bin_q = packBinHelperRFClass(bin, num_trees_in_bin, bin_start);
		}
		else if (algorithm == std::string("randomforest") && task == std::string("regression")){
			bin_q = packBinHelperRFReg(bin, num_trees_in_bin, bin_start);
		}
		else{	
			bin_q = packBinHelperGB(bin, num_trees_in_bin, bin_start);
		}


		// STAT per (sub)tree layout 
		if(layout.find(std::string("bfs")) != std::string::npos){
			packSubtreeBFSHelper(bin, num_trees_in_bin, bin_start, bin_q);
		}
		else if(layout.find(std::string("dfs")) != std::string::npos){
			packSubtreeDFSHelper(bin, num_trees_in_bin, bin_start, bin_q);
		}
		else if(layout.find(std::string("class")) != std::string::npos){
			packSubtreeBlockwiseHelper(bin, num_trees_in_bin, bin_start, bin_q, true);
		}
		else{
			packSubtreeBlockwiseHelper(bin, num_trees_in_bin, bin_start, bin_q, false);
		}

		// set new IDs
		int siz = finalbin.size();
		for (auto i=num_classes; i<siz; i++){
			if(finalbin[i].getLeft() >= num_classes)
				finalbin[i].setLeft(node_to_index[bin[finalbin[i].getLeft()].getID()]);
			if(finalbin[i].getRight() >= num_classes)
				finalbin[i].setRight(node_to_index[bin[finalbin[i].getRight()].getID()]);
		}

		//replace bin with final bin
		bin.clear();
		for (auto i: finalbin)
			bin.push_back(i);

	}


	inline void PackRegularLayout(std::vector<StatNode<T, F>> &bin,
			const int num_trees_in_bin, std::vector<int> &bin_start){

		std::deque<StatNode<T, F>> bin_q;
		int num_classes = std::atoi(Config::getValue("numclasses").c_str());
		//initialize queue with root node of each tree
		for(int i=0; i<num_trees_in_bin; ++i){
			bin_q.push_back(bin[bin_start[i]]);
		}

		// STAT per (sub)tree layout 
		if(layout.find(std::string("bfs")) != std::string::npos){
			packSubtreeBFSHelper(bin, num_trees_in_bin, bin_start, bin_q);
		}
		else if(layout.find(std::string("dfs")) != std::string::npos){
			packSubtreeDFSHelper(bin, num_trees_in_bin, bin_start, bin_q);
		}
		else if(layout.find(std::string("class")) != std::string::npos){
			packSubtreeBlockwiseHelper(bin, num_trees_in_bin, bin_start, bin_q, true);
		}
		else{
			packSubtreeBlockwiseHelper(bin, num_trees_in_bin, bin_start, bin_q, false);
		}

		// set new IDs
		int siz = finalbin.size();
		for (auto i=num_classes; i<siz; i++){
			if(finalbin[i].getLeft() >= num_classes)
				finalbin[i].setLeft(node_to_index[bin[finalbin[i].getLeft()].getID()]);
			if(finalbin[i].getRight() >= num_classes)
				finalbin[i].setRight(node_to_index[bin[finalbin[i].getRight()].getID()]);
		}

		//replace bin with final bin
		bin.clear();
		for (auto i: finalbin)
			bin.push_back(i);
	}
};
#endif
