#ifndef TVM_TG_AUTOSCHEDULE_SEARCH_TREE_H_
#define TVM_TG_AUTOSCHEDULE_SEARCH_TREE_H_

#include <tvm/tir/expr.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/container.h>

#include "proposer.h"
#include "structure_space.h"
#include "config.h"
#include "utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"

namespace tvm {

namespace tg {


class SearchHistoryNode : public Object{
 public:
  int num_expanded;
  Array<FloatImm> rewards;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_expanded", &num_expanded);
    v->Visit("rewards", &rewards);
  }

  static constexpr const char* _type_key = "tg.SearchHistory";
  TVM_DECLARE_FINAL_OBJECT_INFO(SearchHistoryNode, Object);
};


class SearchHistory : public ObjectRef {
 public:
  SearchHistory(int num_expanded);

  void update(int increase, Array<FloatImm> new_rewards);

  TVM_DEFINE_OBJECT_REF_METHODS(SearchHistory, ObjectRef, SearchHistoryNode);
  TG_DEFINE_OBJECT_SELF_METHOD(SearchHistoryNode);
};


enum SearchTreeNodeState {
  NotExpanded = 0,
  Expanding = 1,
  Expanded = 2
};


class TuneRecord {
 public:
  Config config;
  float gflops;

  TuneRecord(Config config, float gflops) : config(config), gflops(gflops) {}
};


class SearchTreeNode {
 public:
  SearchTreeNodeState state;
  std::shared_ptr<SearchTreeNode> parent = nullptr;
  int child_id = -1;
  std::vector<std::shared_ptr<SearchTreeNode> > children;
  std::vector<SearchHistory> history;
  // only for leaf node
  bool is_leaf;
  std::vector<PartialConfig> leaf_configs;
  std::vector<std::unordered_map<Config, FloatImm, ObjectHash> > propose_records;
  
  SearchTreeNode(SearchTreeNodeState state=SearchTreeNodeState::NotExpanded,
      std::shared_ptr<SearchTreeNode> parent=nullptr, int child_id=-1) :
      state(state), parent(parent), child_id(child_id), is_leaf(false) {}

  SearchTreeNode(SearchTreeNodeState state, std::shared_ptr<SearchTreeNode> parent,
    int child_id, std::vector<std::shared_ptr<SearchTreeNode> > children) :
      state(state), parent(parent), child_id(child_id), children(children), is_leaf(false) {
    
    for (auto v : children) {
      history.push_back(SearchHistory(0));
    }
  }

  void add_child(std::shared_ptr<SearchTreeNode> child);
  void change_state(SearchTreeNodeState state);
  void update_state();
  void update_reward(float reward);
  void update_reward(Array<Config> configs, float reward);
  void set_leaf_node(std::vector<PartialConfig> leaf_configs);
  void random_leaf_propose(LeafProposer &proposer, std::vector<std::vector<Config> > &results);
};


class SearchTree {
 public:
  std::shared_ptr<SearchTreeNode> root;

  SearchTree() : root(std::make_shared<SearchTreeNode>()) {}

  SearchTree(SearchTree &another) {
    root = another.root;
  }

  SearchTree(SearchTree &&another) {
    root = std::move(another.root);
  }

  SearchTree& operator=(SearchTree &another) {
    root = another.root;
    return *this;
  }

  SearchTree& operator=(SearchTree &&another) {
    root = std::move(another.root);
    return *this;
  }
  
  SearchTree(std::shared_ptr<SearchTreeNode> root) : root(root) {}
};


class ExpandPolicy {
 public:
  int choose(TIRGraph subgraph, int op_id, const EndNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const InlineNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const DecomposeSpatialNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const DecomposeReduceNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const DecomposeAllreduceNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const AllreduceNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const CacheReadNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const CacheWriteNode* node, Array<SearchHistory> history);
  int choose(TIRGraph subgraph, int op_id, const UnrollNode* node, Array<SearchHistory> history);

  template<typename T>
  int operator()(TIRGraph subgraph, int op_id, T node, Array<SearchHistory> history) {
    return choose(subgraph, op_id, node, history);
  }
};


class SearchTreeExpander : public StructureSpaceFunctor<void(const StructureSpace&)> {
 private:
  TIRGraph subgraph;
  int op_id;
  std::shared_ptr<SearchTreeNode> current;
  PartialConfig partial_config;
  ExpandPolicy expand_policy;

 public:
  void VisitSpace_(const EndNode* node) override;
  void VisitSpace_(const InlineNode* node) override;
  void VisitSpace_(const DecomposeSpatialNode* node) override;
  void VisitSpace_(const DecomposeReduceNode* node) override;
  void VisitSpace_(const DecomposeAllreduceNode* node) override;
  void VisitSpace_(const AllreduceNode* node) override;
  void VisitSpace_(const CacheReadNode* node) override;
  void VisitSpace_(const CacheWriteNode* node) override;
  void VisitSpace_(const UnrollNode* node) override;
  
  template<typename T>
  int prepare_expand(T node);

  std::pair<std::shared_ptr<SearchTreeNode>, PartialConfig> expand(const StructureSpace &space) {
    VisitSpace(space);
    return std::make_pair(current, partial_config);
  }

  SearchTreeExpander(
    TIRGraph subgraph, int op_id,
    std::shared_ptr<SearchTreeNode> current,
    PartialConfig partial_config, ExpandPolicy expand_policy) :
    subgraph(subgraph), op_id(op_id), current(current),
    partial_config(partial_config), expand_policy(expand_policy) {}
};


std::shared_ptr<SearchTreeNode> expand_tree(
  TIRGraph subgraph, Array<StructureSpace> space_dags, SearchTree &tree);


}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_AUTOSCHEDULE_SEARCH_TREE_H_