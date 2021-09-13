#pragma once

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/te/tensor.h>
#include <tvm/runtime/object.h>

#include <string>
#include <vector>

namespace tvm {

namespace nas {


class SubstituteTensor : public tir::ExprMutator {
 public:
  using tir::ExprMutator::VisitExpr;

  SubstituteTensor(Array<te::Tensor> org, Array<te::Tensor> replace) : org_(org), replace_(replace) {}
 private:
  Array<te::Tensor> org_;
  Array<te::Tensor> replace_;
 protected:
 using tir::ExprMutator::VisitExpr_;
  // list of functions to override.
  PrimExpr VisitExpr_(const tir::ProducerLoadNode* op) override;
};

// fwd decl for LayerNode
class LayerNode;
// fwd decl for LayerTensor
class LayerTensor;

/*!
 * \brief Layer class.
 */
class Layer : public ObjectRef {
 public:
  // TVM_DEFINE_OBJECT_REF_METHODS(Layer, ObjectRef, LayerNode);
  // TVM_DEFINE_OBJECT_REF_COW_METHOD(LayerNode);

  /*! \brief default constructor  */
  Layer() {}
  explicit Layer(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const LayerNode* operator->() const;
  /*!
   * \brief The constructor.
   * \param name The name of layer
   * \param ops The op of this layer
   * \param inputs The inputs of this layer
   * \param weights The weights of this layer
   * \param const_scalars The constant scalars
   * \param const_tensors The constant tensors
   * \param gradients The gradients of this layer
   */
  TVM_DLL Layer(std::string name, Array<te::Operation> ops, Array<te::Tensor> inputs,
                Array<te::Tensor> weights, Array<PrimExpr> const_scalars,
                Array<te::Tensor> const_tensors, Array<te::Tensor> gradients);
  /*!
   * \brief Self-checking if the given compute is valid.
   */
  void check_validity();
  /*!
   * \brief The constructor.
   * \param inputs The input tensors.
   */
  std::vector<LayerTensor> produce_outputs(std::vector<LayerTensor> layer_inputs);
  /*! \brief specify container node */
  using ContainerType = LayerNode;
};

/////////////////////////////////////
// Definitions for tensor between layers
////////////////////////////////////

/*!
 * \brief LayerTensorNode class.
 */
class LayerTensorNode : public Object {
 public:
  /*! \brief The name of layer (optional) */
  std::string name{"layer_tensor"};
  /*! \brief The layer that produces this tensor, can be nullptr */
  Layer layer{nullptr};
  /*! \brief The real tensor wrapped */
  te::Tensor tensor;
  /*! \brief The ordinal number of this tensor */
  int value_idx{0};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("layer", &layer);
    v->Visit("tensor", &tensor);
    v->Visit("value_idx", &value_idx);
  }

  static constexpr const char* _type_key = "nas.LayerTensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayerTensorNode, Object);
};

class LayerTensor : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of layer
   * \param layer The layer
   * \param tensor The tensor
   * \param value_idx The value index
   */
  TVM_DLL LayerTensor(std::string name, Layer layer, te::Tensor tensor, int value_idx);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LayerTensor, ObjectRef, LayerTensorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LayerTensorNode);
};


/////////////////////////////////////
// Definitions for layer
////////////////////////////////////

/*!
 * \brief LayerNode class.
 */
class LayerNode : public Object {
 public:
  /*! \brief The name of layer (optional) */
  std::string name{"layer"};
  /*! \brief The op within this layer, required */
  Array<te::Operation> ops;
  /*! \brief The inputs of this layer, can by [] */
  Array<te::Tensor> inputs;
  /*! \brief The weights of this layer, can by [] */
  Array<te::Tensor> weights;
  /*! \brief The const scalar values of this layer, can by [] */
  Array<PrimExpr> const_scalars;
  /*! \brief The const tensors of this layer, can by [] */
  Array<te::Tensor> const_tensors;
  /*! \brief The gradients of this layer, can by [] */
  Array<te::Tensor> gradients;
  /*! \brief The input layer tensors */
  Array<LayerTensor> input_layer_tensors_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("ops", &ops);
    v->Visit("inputs", &inputs);
    v->Visit("weights", &weights);
    v->Visit("const_scalars", &const_scalars);
    v->Visit("const_tensors", &const_tensors);
    v->Visit("gradients", &gradients);
    v->Visit("input_layer_tensors", &input_layer_tensors_);
  }
  /*!
   * \brief Get the input tensors.
   */
  Array<LayerTensor> InputTensors() const;

  static constexpr const char* _type_key = "nas.Layer";
  TVM_DECLARE_FINAL_OBJECT_INFO(LayerNode, Object);
};


inline const LayerNode* Layer::operator->() const {
  return static_cast<const LayerNode*>(get());
}

/////////////////////////////////////
// Definitions for Graph
// Graph = <NodeSet, EdgeSet>
// NodeSet and EdgeSet can be ommited
// we only record the output layers
////////////////////////////////////

/*!
 * \brief A base class for tensor.
 */
class GraphNode : public Object {
 public:
  /*! \brief The name of graph */
  std::string name;
  /*! \brief The output tensors */
  Array<LayerTensor> out_tensors;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("out_tensors", &out_tensors);
  }

  static constexpr const char* _type_key = "nas.Graph";
  TVM_DECLARE_BASE_OBJECT_INFO(GraphNode, Object);
};

class Graph : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of tensor
   */
  TVM_DLL Graph(std::string name, Array<LayerTensor> out_tensors);

  TVM_DEFINE_OBJECT_REF_METHODS(Graph, ObjectRef, GraphNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GraphNode);
};

}  // namespace nas

}  // namespace tvm