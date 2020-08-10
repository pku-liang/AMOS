/*!
 * \file feature_visitor.h
 * \brief Base class for feature extractor.
 *        These features are used for machine learning cost model
 */

#ifndef TVM_AUTOTVM_FEATURE_VISITOR_H_
#define TVM_AUTOTVM_FEATURE_VISITOR_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <string>

namespace tvm {
namespace tg {

using namespace tvm::tir;

enum AccessType {
  kNone = 0b00,
  kRead = 0b01,
  kWrite = 0b10,
  kReadWrite = 0b11,
  kAccessTypeNum = 4,
};

/*!
 * \brief Type of for loop, used as one-hot encoding in features
 */
enum AnnotationType {
  kBlockX, kBlockY, kBlockZ, kThreadX, kThreadY, kThreadZ,
  kUnrolled, kVectorized, kParallel, kSerial, kVirtualThread,
  kPragma, kNum,
};

/*!
 * \brief A base class for feature extractor, used for processing
 * for loop and memory access in the IR
 */
class FeatureVisitor : public StmtExprVisitor {
 public:
  // for loop
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

  // memory access
  void VisitExpr_(const LoadNode* op) final;
  void VisitStmt_(const StoreNode* op);

  using StmtExprVisitor::VisitStmt_;
  using StmtExprVisitor::VisitExpr_;

 protected:
  /*!
 * \brief Enter a for loop node
 * \param var The expression to be printed.
 * \return skip Whether skip this node
 */
  virtual bool EnterItervar_(tir::Var var, int64_t min, int64_t length, bool is_attr_stmt, 
                             AnnotationType ann, const char *pragma_key, const PrimExpr *pragma_val) = 0;
  /*! \brief Exit a for loop subtree */
  virtual void ExitItervar_() = 0;
  /*!
   * \brief Enter a memory access node
   * \param buffer_var The buffer to access.
   * \param index Index expression
   */
  virtual void EnterMem_(tir::Var buffer_var, tvm::PrimExpr index, AccessType access_type) = 0;
  /*! \brief Exit a memory access node */
  virtual void ExitMem_() = 0;

  virtual void EnterAllocateNode_(std::string scope) = 0;
  virtual void ExitAllocateNode_() = 0;
};

}  // namespace autotvm
}  // namespace tvm

#endif  // TVM_AUTOTVM_FEATURE_VISITOR_H_
