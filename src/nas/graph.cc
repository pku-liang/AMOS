#include <tvm/nas/graph.h>


namespace tvm {


namespace nas {

TVM_REGISTER_NODE_TYPE(DataBaseNode);
TVM_REGISTER_NODE_TYPE(TensorNode);
TVM_REGISTER_NODE_TYPE(OperationBaseNode);
TVM_REGISTER_NODE_TYPE(StatelessOpNode);
TVM_REGISTER_NODE_TYPE(StateNode);
TVM_REGISTER_NODE_TYPE(StateOpNode);
TVM_REGISTER_NODE_TYPE(ConstantOpNode);
TVM_REGISTER_NODE_TYPE(FunctionOpNode);
TVM_REGISTER_NODE_TYPE(PlaceholderOpNode);
TVM_REGISTER_NODE_TYPE(ComputeOpNode);


DataBase::DataBase(std::string name) {
    auto node = make_object<DataBaseNode>();
    node->name = name;
    data_ = node;
}


Tensor::Tensor(std::string name, DataType dtype, Array<PrimExpr> shape) {
    auto node = make_object<TensorNode>();
    node->name = name;
    node->dtype = dtype;
    node->shape = shape;
    data_ = node;
}


OperationBase::OperationBase(std::string name) {
    auto node = make_object<OperationBaseNode>();
    node->name = name;
    node->input_tensors = Array<Tensor>();
    data_ = node;
}


StatelessOp::StatelessOp(std::string name) {
    auto node = make_object<StatelessOpNode>();
    node->name = name;
    data_ = node;
}


State::State(std::string name, DataType dtype, Array<PrimExpr> shape) {
    auto node = make_object<StateNode>();
    node->name = name;
    node->dtype = dtype;
    node->shape = shape;
    data_ = node;
}


StateOp::StateOp(std::string name, Array<State> states) {
    auto node = make_object<StateOpNode>();
    node->name = name;
    node->states = states;
    data_ = node;
}


ConstantOp::ConstantOp(std::string name, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> array_data) {
    auto node = make_object<ConstantOpNode>();
    node->name = name;
    node->dtype = dtype;
    node->shape = shape;
    node->array_data = array_data;
    data_ = node;
}


FunctionOp::FunctionOp(std::string name, Array<PrimExpr> axis, Array<PrimExpr> reduce_axis, PrimExpr body) {
    auto node = make_object<FunctionOpNode>();
    node->name = name;
    node->axis = axis;
    node->reduce_axis = reduce_axis;
    node->body = body;
    data_ = node;
}


PlaceholderOp::PlaceholderOp(std::string name, Array<State> states, DataType dtype, Array<PrimExpr> shape) {
    auto node = make_object<PlaceholderOpNode>();
    node->name = name;
    node->states = states;
    node->dtype = dtype;
    node->shape = shape;
    data_ = node;
}


ComputeOp::ComputeOp(std::string name, Array<State> states, Array<PrimExpr> axis, Array<PrimExpr> reduce_axis, PrimExpr body) {
    auto node = make_object<ComputeOpNode>();
    node->name = name;
    node->states = states;
    node->axis = axis;
    node->reduce_axis = reduce_axis;
    node->body = body;
    data_ = node;
}


Graph::Graph(std::string name, Array<OperationBase> out_ops) {
    auto node = make_object<GraphNode>();
    node->name = name;
    node->out_ops = out_ops;
    data_ = node;
}


}  // namespace nas


}  // namespace tvm