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
 * \brief Rewrite the memory scope for auto tensorization.
 * \file auto_tensorize_rewrite_memory_scope.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/node/structural_equal.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_util.h"
#include "storage_access.h"

namespace tvm {
namespace tir {

// Get memory scope information by query capsule register pool
class MemoryScopeGetter : public StmtExprVisitor {
 public:
  // MemoryScopeInfo
  class MemoryScopeInfo {
   public:
    Map<String, StringImm> raw;

    MemoryScopeInfo() : raw() {}

    MemoryScopeInfo(Map<String, StringImm> r) : raw(r) {
    //   CHECK(raw.find(String(attr::storage_scope)) != raw.end())
    //     << "Can't find " << std::string(attr::storage_scope) << " attribute.";
    }

    void check() const {
      CHECK(raw.find(String(attr::storage_scope)) != raw.end())
        << "Can't find " << std::string(attr::storage_scope) << " attribute.";
    }

    MemoryScopeInfo& merge(const MemoryScopeInfo& another) {
      for (auto kv : another.raw) {
        if (raw.find(kv.first) == raw.end()) {
          raw.Set(kv.first, kv.second);
        } else {
          CHECK(raw.at(kv.first)->value == kv.second->value);
        }
      }
      return *this;
    }
  };

  const tvm::runtime::PackedFunc* query_capsule_memory_scope =
      runtime::Registry::Get("auto_tensorize.query_capsule_memory_scope");
  std::unordered_map<const VarNode*, std::vector<MemoryScopeInfo>> memory_info;

  MemoryScopeGetter() {
      CHECK(query_capsule_memory_scope)
        << "Can't find auto_tensorize.query_capsule_memory_scope.";
  }

  void VisitStmt_(const AllocateNode* op) final {
    buffer_vars.insert(op->buffer_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);

    if (op->op.same_as(builtin::capsule_compile())) {
      CHECK_GE(op->args.size(), 3U);
      const StringImmNode* target = op->args[0].as<StringImmNode>();
      const StringImmNode* recipe_mnemonic = op->args[1].as<StringImmNode>();
      const StringImmNode* capsule_mnemonic = op->args[2].as<StringImmNode>();
      CHECK(target);
      CHECK(recipe_mnemonic);
      CHECK(capsule_mnemonic);

      Map<PrimExpr, Map<String, StringImm>> ret;
      size_t total_args = op->args.size();
      Array<PrimExpr> other_args;
      for (size_t i = 3; i < total_args; ++i)
        other_args.push_back(op->args[i]);
      for (size_t i = 3; i < total_args; ++i) {
        const VarNode* buffer_var = op->args[i].as<VarNode>();
        if (buffer_var != nullptr && buffer_vars.count(buffer_var)) {
          ret.Set(op->args[i],
                  (*query_capsule_memory_scope)(
                        op->args[0], op->args[1], op->args[2], (int)i-3, other_args));
        }
      }

      for (auto kv : ret) {
        const VarNode* buffer_var = kv.first.as<VarNode>();
        if (buffer_var != nullptr) {
          memory_info[buffer_var].push_back(MemoryScopeInfo(kv.second));
        }
      }
    }
  }

 private:
  std::unordered_set<const VarNode*> buffer_vars;
};

// Check memory info has no conflict
class MemoryScopeChecker : public StmtExprVisitor {
 public:
  MemoryScopeChecker(const MemoryScopeGetter& getter) : getter_(getter) {
    for (auto kv : getter_.memory_info) {
      MemoryScopeGetter::MemoryScopeInfo info;
      for (auto v : kv.second) {
        info.merge(v);
      }
      info.check();
      memory_info[kv.first] = info;
    }
  }

 private:
  const MemoryScopeGetter& getter_;
 public:
  std::unordered_map<const VarNode*, MemoryScopeGetter::MemoryScopeInfo> memory_info;
};

// Add AttrStmt for memory scope info
class RewriteMemoryScoper : public StmtExprMutator {
 public:
  RewriteMemoryScoper(const MemoryScopeChecker& checker) : checker_(checker) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    const VarNode* buffer_var = op->node.as<VarNode>();
    for (auto kv : checker_.memory_info) {
      if (buffer_var == kv.first) {
        for (auto kkv : kv.second.raw) {
          if (kkv.first == op->attr_key) {
            added_attrs[kv.first].insert(kkv.first);
            return AttrStmt(op->node, op->attr_key, kkv.second, VisitStmt(op->body));
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const VarNode* buffer_var = op->buffer_var.get();
    if (checker_.memory_info.find(buffer_var) != checker_.memory_info.end()) {
      MemoryScopeGetter::MemoryScopeInfo info = checker_.memory_info.at(buffer_var);
      for (auto kv : info.raw) {
        if (!added_attrs[buffer_var].count(kv.first)) {
          stmt = AttrStmt(op->buffer_var, kv.first, kv.second, stmt);
        }
      }
    }
    return stmt;
  }

 private:
  const MemoryScopeChecker& checker_;
  std::unordered_map<const VarNode*, std::unordered_set<String>> added_attrs;
};

Stmt RewriteMemoryScope(Stmt stmt) {
  MemoryScopeGetter getter;
  getter(stmt);
  MemoryScopeChecker checker(getter);
  checker(stmt);
  stmt = RewriteMemoryScoper(checker)(std::move(stmt));
  return stmt;
}

namespace transform {

Pass RewriteMemoryScope() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = RewriteMemoryScope(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RewriteMemoryScope", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RewriteMemoryScope").set_body_typed(RewriteMemoryScope);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
