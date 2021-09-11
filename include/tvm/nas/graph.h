#pragma once

#include <tvm/runtime/object.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/container.h>
#include <tvm/node/node.h>
#include <tvm/ir/expr.h>
#include <string>

namespace tvm {

using namespace tvm::runtime;

namespace nas {

/////////////////////////////////////
// Definitions for Tensors
// Tensor represents for edge
// tensor is related with op through name
// rather than pointer
////////////////////////////////////

/*!
 * \brief A base class for tensor.
 */
class DataBaseNode : public Object {
 public:
 /*! \brief The name of tensor */
  std::string name;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "nast.DataBase";
  TVM_DECLARE_BASE_OBJECT_INFO(DataBaseNode, Object);
};


class DataBase : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of tensor
   */
  TVM_DLL DataBase(std::string name);

  TVM_DEFINE_OBJECT_REF_METHODS(DataBase, ObjectRef, DataBaseNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataBaseNode);
};


/*!
 * \brief A class for tensor.
 */
class TensorNode : public DataBaseNode {
 public:
  /*! \brief The data type of state */
  DataType dtype;
  /*! \brief The shape of state */
  Array<PrimExpr> shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "nast.Tensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorNode, DataBaseNode);
};


class Tensor : public DataBase {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of tensor
   * \param dtype The data type of tensor
   * \param shape The shape of tensor
   */
  TVM_DLL Tensor(std::string name, DataType dtype, Array<PrimExpr> shape);

  TVM_DEFINE_OBJECT_REF_METHODS(Tensor, DataBase, TensorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorNode);
};


/////////////////////////////////////
// Definitions for Operations
// Operation defines single node
////////////////////////////////////

/*!
 * \brief A base class for operation.
 */
class OperationBaseNode : public Object {
 public:
 /*! \brief The name of operation */
  std::string name;
  /*! \brief The input tensors */
  Array<Tensor> input_tensors;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("input_tensors", &input_tensors);
  }

  static constexpr const char* _type_key = "nast.OperationBase";
  TVM_DECLARE_BASE_OBJECT_INFO(OperationBaseNode, Object);
};


class OperationBase : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   */
  TVM_DLL OperationBase(std::string name);

  TVM_DEFINE_OBJECT_REF_METHODS(OperationBase, ObjectRef, OperationBaseNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OperationBaseNode);
};


/*!
 * \brief Operation that has no weights.
 */
class StatelessOpNode : public OperationBaseNode {
 public:
  static constexpr const char* _type_key = "nast.StatelessOp";
  TVM_DECLARE_BASE_OBJECT_INFO(StatelessOpNode, OperationBaseNode);
};


class StatelessOp : public OperationBase {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   */
  TVM_DLL StatelessOp(std::string name);

  TVM_DEFINE_OBJECT_REF_METHODS(StatelessOp, OperationBase, StatelessOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StatelessOpNode);
};


/*!
 * \brief State of an operation, usually a tensor-like array.
 */
class StateNode : public Object {
 public:
 /*! \brief The name of operation */
  std::string name;
  /*! \brief The data type of state */
  DataType dtype;
  /*! \brief The shape of state */
  Array<PrimExpr> shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "nast.State";
  TVM_DECLARE_FINAL_OBJECT_INFO(StateNode, Object);
};


class State : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   * \param dtype data type of state
   * \param shape shape of state
   */
  TVM_DLL State(std::string name, DataType dtype, Array<PrimExpr> shape);

  TVM_DEFINE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateNode);
};


/*!
 * \brief Operation that has weights.
 */
class StateOpNode : public OperationBaseNode {
 public:
 /*! \brief The state of this operation */
  Array<State> states;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("states", &states);
  }

  static constexpr const char* _type_key = "nast.StateOp";
  TVM_DECLARE_BASE_OBJECT_INFO(StateOpNode, OperationBaseNode);
};


class StateOp : public OperationBase {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   * \param states the states in this op
   */
  TVM_DLL StateOp(std::string name, Array<State> states);

  TVM_DEFINE_OBJECT_REF_METHODS(StateOp, OperationBase, StateOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StateOpNode);
};


/*!
 * \brief Operation that defines constant values.
 */
class ConstantOpNode : public StatelessOpNode {
 public:
  /*! \brief The data type of state */
  DataType dtype;
  /*! \brief The shape of state */
  Array<PrimExpr> shape;
  /*! \brief The constant values, length equal to shape */
  Array<PrimExpr> array_data;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
    v->Visit("array_data", &array_data);
  }

  static constexpr const char* _type_key = "nast.ConstantOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantOpNode, StatelessOpNode);
};


class ConstantOp : public StatelessOp {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   */
  TVM_DLL ConstantOp(std::string name, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> array_data);

  TVM_DEFINE_OBJECT_REF_METHODS(ConstantOp, StatelessOp, ConstantOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ConstantOpNode);
};


/*!
 * \brief Operation that performs pure computation.
 */
class FunctionOpNode : public StatelessOpNode {
 public:
 /*! \brief The axis */
  Array<PrimExpr> axis;
  /*! \brief The reduce_axis */
  Array<PrimExpr> reduce_axis;
  /*! \brief The body */
  PrimExpr body;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("axis", &axis);
    v->Visit("reduce_axis", &reduce_axis);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "nast.FunctionOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionOpNode, StatelessOpNode);
};


class FunctionOp : public StatelessOp {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   * \param axis The axis of compute
   * \param reduce_axis the reduce_axis of compute
   * \param body the body of compute
   */
  TVM_DLL FunctionOp(std::string name, Array<PrimExpr> axis, Array<PrimExpr> reduce_axis, PrimExpr body);

  TVM_DEFINE_OBJECT_REF_METHODS(FunctionOp, StatelessOp, FunctionOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionOpNode);
};


/*!
 * \brief Operation that defines non-constant values.
 */
class PlaceholderOpNode : public StateOpNode {
 public:
  /*! \brief The data type of state */
  DataType dtype;
  /*! \brief The shape of state */
  Array<PrimExpr> shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
  }
  
  static constexpr const char* _type_key = "nast.PlaceholderOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(PlaceholderOpNode, StateOpNode);
};


class PlaceholderOp : public StateOp {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   * \param states The states
   * \param dtype The data type
   * \param shape The shape
   */
  TVM_DLL PlaceholderOp(std::string name, Array<State> states, DataType dtype, Array<PrimExpr> shape);

  TVM_DEFINE_OBJECT_REF_METHODS(PlaceholderOp, StateOp, PlaceholderOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(PlaceholderOpNode);
};


/*!
 * \brief Operation for computation with weights.
 */
class ComputeOpNode : public StateOpNode {
 public:
  /*! \brief The axis */
  Array<PrimExpr> axis;
  /*! \brief The reduce_axis */
  Array<PrimExpr> reduce_axis;
  /*! \brief The body */
  PrimExpr body;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("axis", &axis);
    v->Visit("reduce_axis", &reduce_axis);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "nast.ComputeOp";
  TVM_DECLARE_FINAL_OBJECT_INFO(ComputeOpNode, StateOpNode);
};


class ComputeOp : public StateOp {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of operation
   * \param states The states
   * \param axis The axis of compute
   * \param reduce_axis the reduce_axis of compute
   * \param body the body of compute
   */
  TVM_DLL ComputeOp(std::string name, Array<State> states, Array<PrimExpr> axis, Array<PrimExpr> reduce_axis, PrimExpr body);

  TVM_DEFINE_OBJECT_REF_METHODS(ComputeOp, StateOp, ComputeOpNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ComputeOpNode);
};


/////////////////////////////////////
// Definitions for Graph
// Graph = <NodeSet, EdgeSet>
// NodeSet and EdgeSet can be ommited
// we only record the output ops
////////////////////////////////////

/*!
 * \brief A base class for tensor.
 */
class GraphNode : public Object {
 public:
 /*! \brief The name of graph */
  std::string name;
  Array<OperationBase> out_ops;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("out_ops", &out_ops);
  }

  static constexpr const char* _type_key = "nast.Graph";
  TVM_DECLARE_BASE_OBJECT_INFO(GraphNode, Object);
};


class Graph : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of tensor
   */
  TVM_DLL Graph(std::string name, Array<OperationBase> out_ops);

  TVM_DEFINE_OBJECT_REF_METHODS(Graph, ObjectRef, GraphNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GraphNode);
};


}  // namespace nas


}  // namespace tvm