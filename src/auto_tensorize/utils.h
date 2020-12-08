/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_tensorize/utils.h
 * \brief Common utilities.
 */

#ifndef TVM_AUTO_TENSORIZE_UTILS_H_
#define TVM_AUTO_TENSORIZE_UTILS_H_


#include <tvm/tir/stmt_functor.h>
#include <unordered_set>


namespace tvm {
namespace auto_tensorize {

class BufferSizeCollector : public tir::StmtVisitor {
 public:
  using tir::StmtVisitor::VisitStmt_;
  BufferSizeCollector(String scope) : scope_(scope) {}

  Map<tir::Var, IntImm> collect(const tir::Stmt& n) {
    VisitStmt(n);
    return record_;
  }

  void VisitStmt_(const tir::AttrStmtNode* op) final;
  void VisitStmt_(const tir::AllocateNode* op) final;
 private:
  Map<tir::Var, IntImm> record_;
  String scope_;
  std::unordered_set<const tir::VarNode*> vars_;
};

}  // namespace auto_tensorize
}  // namespace tvm

#endif  // TVM_AUTO_TENSORIZE_UTILS_H_
