#include <random>

#include "search_tree.h"


namespace tvm {

namespace tg {


SearchHistory::SearchHistory(int num_expanded) {
  auto node = make_object<SearchHistoryNode>();
  node->num_expanded = num_expanded;

  data_ = std::move(node);
}


void SearchHistory::update(int increase, Array<FloatImm> new_rewards) {
  auto self = Self();
  self->num_expanded += increase;
  for (auto v : new_rewards) {
    self->rewards.push_back(v);
  }
}


void SearchTreeNode::add_child(std::shared_ptr<SearchTreeNode> child) {
  children.push_back(child);
  history.push_back(SearchHistory(0));
}


void SearchTreeNode::change_state(SearchTreeNodeState state) {
  state = state;
}


void SearchTreeNode::update_state() {
  if (is_leaf || state == SearchTreeNodeState::Expanded) {
    state = SearchTreeNodeState::Expanded;
    if (parent != nullptr) {
      parent->update_state();
    }
  } else {
    for (auto child : children) {
      if (child->state == SearchTreeNodeState::NotExpanded) {
        if (parent != nullptr) {
          parent->update_state();
        }
        return;
      }
    }

    state = SearchTreeNodeState::Expanded;
    if (parent != nullptr) {
      parent->update_state();
    }
  }
}


void SearchTreeNode::update_reward(float reward) {
  if (parent != nullptr && child_id >= 0) {
    parent->history[child_id].update(1, {FloatImm(DataType::Float(32), reward)});
    parent->update_reward(reward);
  }
}


void SearchTreeNode::set_leaf_node(std::vector<PartialConfig> leaf_configs) {
  this->leaf_configs = leaf_configs;
  this->is_leaf = true;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const EndNode* node, Array<SearchHistory> history) {
  
  return 0;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const InlineNode* node, Array<SearchHistory> history) {
  
  const auto* f = runtime::Registry::Get("tg.autoschedule.expand_inline_policy");
  if (f == nullptr) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> distrib(0, 1);
    return distrib(gen);
  } else {
    return (*f)(subgraph, op_id, InlineNode::make(node->next[0], node->next[1]), history);
  }
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const DecomposeSpatialNode* node, Array<SearchHistory> history) {
  
  return 0;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const DecomposeReduceNode* node, Array<SearchHistory> history) {
  
  return 0;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const DecomposeAllreduceNode* node, Array<SearchHistory> history) {
  
  return 0;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const AllreduceNode* node, Array<SearchHistory> history) {
  
  const auto* f = runtime::Registry::Get("tg.autoschedule.expand_allreduce_policy");
  if (f == nullptr) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> distrib(0, 1);
    return distrib(gen);
  } else {
    return (*f)(subgraph, op_id, AllreduceNode::make(node->next[0], node->next[1]), history);
  }
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const CacheReadNode* node, Array<SearchHistory> history) {
  
  return 0;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const CacheWriteNode* node, Array<SearchHistory> history) {
  
  return 0;
}


int ExpandPolicy::choose(
  TIRGraph subgraph, int op_id, const UnrollNode* node, Array<SearchHistory> history) {
  
  return 0;
}


template<typename T>
int SearchTreeExpander::prepare_expand(T node) {
  int num_child = (int)node->next.size();
  if (current->state == SearchTreeNodeState::NotExpanded) {
    for (int i = 0; i < num_child; ++i) {
      current->add_child(std::make_shared<SearchTreeNode>(
        SearchTreeNodeState::NotExpanded,
        current,
        i
      ));
    }
    current->change_state(SearchTreeNodeState::Expanding);
  }

  return num_child;
}


/* end */
void SearchTreeExpander::VisitSpace_(const EndNode* node) {
  return;
}


/* inline */
void SearchTreeExpander::VisitSpace_(const InlineNode* node) {
  int num_child = prepare_expand(node);

  int child_id = expand_policy(subgraph, op_id, node, current->history);
  CHECK(child_id < num_child) << "Expand policy exceeds children limit.";

  current = current->children[child_id];
  if (child_id == 0) {
    // use inline
    partial_config.add_structure_entities(Knob::UseInline, InlineEntityNode::make(true));
  } else {
    partial_config.add_structure_entities(Knob::UseInline, InlineEntityNode::make(false));
  }

  // move on
  VisitSpace(node->next[child_id]);
}


/* decompose spatial */
void SearchTreeExpander::VisitSpace_(const DecomposeSpatialNode* node) {
  int num_child = prepare_expand(node);

  int child_id = 0;

  current = current->children[child_id];
  partial_config.add_param_spaces(
    Knob::Spatial, DecomposeSpatialNode::make(EndNode::make(), node->splits, node->reorder));

  // move on
  VisitSpace(node->next[child_id]);
}


/* decompose reduce */
void SearchTreeExpander::VisitSpace_(const DecomposeReduceNode* node) {
  int num_child = prepare_expand(node);

  int child_id = 0;

  current = current->children[child_id];
  partial_config.add_param_spaces(
    Knob::Reduce, DecomposeReduceNode::make(EndNode::make(), node->splits, node->reorder));

  // move on
  VisitSpace(node->next[child_id]);
}


/* decompose allredcue */
void SearchTreeExpander::VisitSpace_(const DecomposeAllreduceNode* node) {
  int num_child = prepare_expand(node);

  int child_id = 0;

  current = current->children[child_id];
  partial_config.add_param_spaces(
    Knob::Allreduce, DecomposeAllreduceNode::make(EndNode::make(), node->splits, node->use_factor));

  // move on
  VisitSpace(node->next[child_id]);
}


/* allreduce */
void SearchTreeExpander::VisitSpace_(const AllreduceNode* node) {
  int num_child = prepare_expand(node);

  int child_id = expand_policy(subgraph, op_id, node, current->history);
  CHECK(child_id < num_child) << "Expand policy exceeds children limit.";

  current = current->children[child_id];
  if (child_id == 0) {
    // use allreduce
    partial_config.add_structure_entities(Knob::UseAllreduce, AllreduceEntityNode::make(true));
  } else {
    partial_config.add_structure_entities(Knob::UseAllreduce, AllreduceEntityNode::make(false));
  }

  // move on
  VisitSpace(node->next[child_id]);
}


/* cache read */
void SearchTreeExpander::VisitSpace_(const CacheReadNode* node) {
  int num_child = prepare_expand(node);

  int child_id = 0;

  current = current->children[child_id];
  partial_config.add_param_spaces(
    Knob::CacheRead, CacheReadNode::make(EndNode::make(), node->cache_config));

  // move on
  VisitSpace(node->next[child_id]);
}


/* cache write */
void SearchTreeExpander::VisitSpace_(const CacheWriteNode* node) {
  int num_child = prepare_expand(node);

  int child_id = 0;

  current = current->children[child_id];
  partial_config.add_param_spaces(
    Knob::CacheWrite, CacheWriteNode::make(EndNode::make(), node->cache_write));

  // move on
  VisitSpace(node->next[child_id]);
}


/* unroll */
void SearchTreeExpander::VisitSpace_(const UnrollNode* node) {
  int num_child = prepare_expand(node);

  int child_id = 0;

  current = current->children[child_id];
  partial_config.add_param_spaces(
    Knob::Unroll, UnrollNode::make(EndNode::make(), node->unroll));

  // move on
  VisitSpace(node->next[child_id]);
}


SearchTree make_tree(TIRGraph subgraph, Array<StructureSpace> space_dags, bool expand_all) {
  SearchTree tree;
  std::shared_ptr<SearchTreeNode> current = tree.root;
  ExpandPolicy policy;
  std::vector<PartialConfig> partial_configs;
  
  int num_ops = (int)subgraph->operation_list.size();
  for (int i = 0; i < num_ops; ++i) {
    PartialConfig partial_config = make_empty_partial_config();
    auto dag = space_dags[i];
    SearchTreeExpander expander(subgraph, i, current, partial_config, policy);
    std::tie(current, partial_config) = expander.expand(dag);
    partial_configs.push_back(partial_config);
  }

  // record the partial configs in tree leaf node
  current->set_leaf_node(partial_configs);
  // propagate reward to root
  current->update_state();
  return tree;
}

}  // namespace tg

}  // namespace tvm