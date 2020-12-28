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
 * \file auto_tensorize/utils.cc
 * \brief Utils for auto_tensorize.
 */

#include "utils.h"
#include <tvm/runtime/registry.h>

namespace tvm {
namespace auto_tensorize {

void BufferSizeCollector::VisitStmt_(const tir::AttrStmtNode* op) {
    if (op->attr_key == tir::attr::storage_scope) {
        const tir::VarNode* as_var = op->node.as<tir::VarNode>();
        CHECK(as_var);
        const tir::StringImmNode* as_string = op->value.as<tir::StringImmNode>();
        CHECK(as_string);
        if (as_string->value == std::string(scope_)) {
        vars_.insert(as_var);
        }
    }
    tir::StmtVisitor::VisitStmt_(op);
}


void BufferSizeCollector::VisitStmt_(const tir::AllocateNode* op) {
    if (vars_.count(op->buffer_var.get())) {
        int64_t cap = 1;
        for (auto e : op->extents) {
        const tir::IntImmNode* as_int = e.as<tir::IntImmNode>();
        CHECK(as_int) << as_int << " is not int const";
        cap *= as_int->value;
        cap *= (int)op->dtype.bytes();
      }
      record_.Set(op->buffer_var, IntImm(DataType::Int(64), cap));
    }
    tir::StmtVisitor::VisitStmt_(op);
}


TVM_REGISTER_GLOBAL("auto_tensorize.get_buffer_size").set_body_typed(
    [](
        String scope,
        const tir::Stmt& n
    ) {
  return BufferSizeCollector(scope).collect(n);
});

}  // namespace auto_tensorize

}  // namespace tvm