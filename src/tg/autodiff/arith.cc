#include "arg_util.h"

namespace tvm {

namespace tg {

template<typename T>
void Matrix<T>::swap_row(int i, int j) {
  CHECK(i < height_) << "index out of height range: " << i << " vs. " << height_ << "\n";
  CHECK(j < height_) << "index out of height range: " << j << " vs. " << height_ << "\n";
  if (i == j) {
    return;
  }
  for (int l = 0; l < width_; ++l) {
    std::swap(*(ptr + i * width_ + l), *(ptr + j * width_ + l));
  }
}


template <typename T>
void Matrix<T>::swap_col(int i, int j) {
  CHECK(i < width_) << "index out of width range: " << i << " vs. " << width_ << "\n";
  CHECK(j < width_) << "index out of width range: " << j << " vs. " << width_ << "\n";
  if (i == j) {
    return;
  }
  for (int l = 0; l < height_; ++l) {
    std::swap(*(ptr + l * width_ + i), *(ptr + l * width_ + j));
  }
}


template <typename T>
void Matrix<T>::scale_row(int i, T factor) {
  CHECK(i < height_) << "index out of height range: " << i << " vs. " << height_ << "\n";
  for (int l = 0; l < width_; ++l) {
    *(ptr + i * width_ + l) = *(ptr + i * width_ + l) * factor;
  }
}


template <typename T>
void Matrix<T>::scale_col(int j, T factor) {
  CHECK(j < width_) << "index out of height range: " << j << " vs. " << width_ << "\n";
  for (int l = 0; l < height_; ++l) {
    *(ptr + l * width_ + j) = *(ptr + l * width_ + j) * factor;
  }
}


template <typename T>
void Matrix<T>::add_row(int i, int j, T factor) {
  CHECK(i < height_) << "index out of height range: " << i << " vs. " << height_ << "\n";
  CHECK(j < height_) << "index out of height range: " << j << " vs. " << height_ << "\n";
  for (int l = 0; l < width_; ++l) {
    *(ptr + j * width_ + l) = *(ptr + i * width_ + l) * factor + *(ptr + j * width_ + l);
  }
}


template <typename T>
void Matrix<T>::add_col(int i, int j, T factor) {
  CHECK(i < width_) << "index out of width range: " << i << " vs. " << width_ << "\n";
  CHECK(j < width_) << "index out of width range: " << j << " vs. " << width_ << "\n";
  for (int l = 0; l < height_; ++l) {
    *(ptr + l * width_ + j) =  *(ptr + l * width_ + i) * factor + *(ptr + l * width_ + j);
  }
}


template <typename T>
void Matrix<T>::row_transform(int i, int j, T s, T t, T g, T h) {
  CHECK(i < height_) << "index out of height range: " << i << " vs. " << height_ << "\n";
  CHECK(j < height_) << "index out of height range: " << j << " vs. " << height_ << "\n";
  std::vector<T> row_i(width_), row_j(width_);
  for (int l = 0; l < width_; ++l) {
    row_i[l] = (*(ptr + i * width_ + l)) * s + (*(ptr + j * width_ + l)) * t;
    row_j[l] = (*(ptr + i * width_ + l)) * g + (*(ptr + j * width_ + l)) * h;
  }
  for (int l = 0; l < width_; ++l) {
    *(ptr + i * width_ + l) = row_i[l];
    *(ptr + j * width_ + l) = row_j[l];
  }
}


template<typename T>
void Matrix<T>::col_transform(int i, int j, T s, T t, T g, T h) {
  CHECK(i < width_) << "index out of width range: " << i << " vs. " << width_ << "\n";
  CHECK(j < width_) << "index out of width range: " << j << " vs. " << width_ << "\n";
  std::vector<T> col_i(height_), col_j(height_);
  for (int l = 0; l < height_; ++l) {
    col_i[l] = (*(ptr + l * width_ + i)) * s + (*(ptr + l * width_ + j)) * t;
    col_j[l] = (*(ptr + l * width_ + i)) * g + (*(ptr + l * width_ + j)) * h;
  }
  for (int l = 0; l < height_; ++l) {
    *(ptr + l * width_ + i) = col_i[l];
    *(ptr + l * width_ + j) = col_j[l];
  }
}


ExtRange ExtRange::floor_div(int factor) {
  ExtRange ret;
  if (!this->left_inf) {
    ret.left = FloorDivNode::make(this->left, factor);
    ret.left_inf = false;
  }
  if (!this->right_inf) {
    // ceil div
    ret.right = FloorDivNode::make(AddNode::make(this->right, factor - 1), factor);
    ret.right_inf = false;
  }
  return ret;
}


ExtRange ExtRange::floor_mod(int factor) {
  return ExtRange(0, factor, false, false);
}


void RangeInference::VisitExpr_(const VarNode* op) {
  range_map[op->name_hint] = scope_.back();
}


void RangeInference::VisitExpr_(const AddNode* op) {
  const IntImmNode *a_as_int = op->a.as<IntImmNode>();
  const IntImmNode *b_as_int = op->b.as<IntImmNode>();
  ExtRange range = scope_.back();
  if (a_as_int != nullptr) {
    int bias = (int)a_as_int->value;
    if (!range.left_inf) {
      range.left = range.left - bias;
    }
    if (!range.right_inf) {
      range.right = range.right - bias;
    }
    scope_.push_back(range);
    VisitExpr(op->b);
    scope_.pop_back();
  } else if (b_as_int != nullptr) {
    int bias = (int)b_as_int->value;
    if (!range.left_inf) {
      range.left = range.left - bias;
    }
    if (!range.right_inf) {
      range.right = range.right - bias;
    }
    scope_.push_back(range);
    VisitExpr(op->a);
    scope_.pop_back();
  }
}


void RangeInference::VisitExpr_(const SubNode* op) {
  const IntImmNode *a_as_int = op->a.as<IntImmNode>();
  const IntImmNode *b_as_int = op->b.as<IntImmNode>();
  ExtRange range = scope_.back();
  if (a_as_int != nullptr) {
    int bias = (int)a_as_int->value;
    ExtRangeType range_type = range.range_type();
    if (range_type == ExtRangeType::LCRC) {
      range.left = bias - scope_.back().right;
      range.right = bias - scope_.back().left;
    } else if (range_type == ExtRangeType::LCRO) {
      range.left_inf = true;
      range.right_inf = false;
      range.right = bias - range.left;
    } else if (range_type == ExtRangeType::LORC) {
      range.left_inf = false;
      range.right_inf = true;
      range.left = bias - range.right;
    } else {
      // nothing to do
    }
    scope_.push_back(range);
    VisitExpr(op->b);
    scope_.pop_back();
  } else if (b_as_int != nullptr) {
    int bias = (int)b_as_int->value;
    if (!range.left_inf) {
      range.left = range.left + bias;
    }
    if (!range.right_inf) {
      range.right = range.right + bias;
    }
    scope_.push_back(range);
    VisitExpr(op->a);
    scope_.pop_back();
  }
}


void RangeInference::VisitExpr_(const MulNode* op) {
  const IntImmNode *a_as_int = op->a.as<IntImmNode>();
  const IntImmNode *b_as_int = op->b.as<IntImmNode>();
  ExtRange range = scope_.back();
  if (a_as_int != nullptr) {
    int bias = (int)a_as_int->value;
    if (bias == 0) {
      range.left = 0;
      range.left_inf = false;
      range.right = 1;
      range.right_inf = false;
    } else if (bias > 0) {
      ExtRangeType range_type = range.range_type();
      if (range_type == ExtRangeType::LCRC) {
        range.left = FloorDivNode::make(range.left, bias);
        range.right = FloorDivNode::make(AddNode::make(range.right, bias - 1), bias);
      } else if (range_type == ExtRangeType::LCRO) {
        range.left = FloorDivNode::make(range.left, bias);
      } else if (range_type == ExtRangeType::LORC) {
        range.right = FloorDivNode::make(AddNode::make(range.right, bias - 1), bias);
      } else {
        // nothing to do
      }
    } else {
      ExtRangeType range_type = range.range_type();
      if (range_type == ExtRangeType::LCRC) {
        range.left = MulNode::make(SubNode::make(FloorDivNode::make(AddNode::make(scope_.back().right, -bias-1), -bias), 1), -1);
        range.right = MulNode::make(SubNode::make(FloorDivNode::make(scope_.back().left, -bias), 1), -1);
      } else if (range_type == ExtRangeType::LCRO) {
        range.left_inf = true;
        range.right_inf = false;
        range.right = MulNode::make(SubNode::make(FloorDivNode::make(range.left, -bias), 1), -1);
      } else if (range_type == ExtRangeType::LORC) {
        range.left_inf = false;
        range.right_inf = true;
        range.left = MulNode::make(SubNode::make(FloorDivNode::make(AddNode::make(range.right, -bias-1), -bias), 1), -1);
      } else {
        // nothing to do
      }
    }
    
    scope_.push_back(range);
    VisitExpr(op->b);
    scope_.pop_back();
  } else if (b_as_int != nullptr) {
    int bias = (int)b_as_int->value;
    if (bias == 0) {
      range.left = 0;
      range.left_inf = false;
      range.right = 1;
      range.right_inf = false;
    } else if (bias > 0) {
      ExtRangeType range_type = range.range_type();
      if (range_type == ExtRangeType::LCRC) {
        range.left = FloorDivNode::make(range.left, bias);
        range.right = FloorDivNode::make(AddNode::make(range.right, bias - 1), bias);
      } else if (range_type == ExtRangeType::LCRO) {
        range.left = FloorDivNode::make(range.left, bias);
      } else if (range_type == ExtRangeType::LORC) {
        range.right = FloorDivNode::make(AddNode::make(range.right, bias - 1), bias);
      } else {
        // nothing to do
      }
    } else {
      ExtRangeType range_type = range.range_type();
      if (range_type == ExtRangeType::LCRC) {
        range.left = MulNode::make(SubNode::make(FloorDivNode::make(AddNode::make(scope_.back().right, -bias-1), -bias), 1), -1);
        range.right = MulNode::make(SubNode::make(FloorDivNode::make(scope_.back().left, -bias), 1), -1);
      } else if (range_type == ExtRangeType::LCRO) {
        range.left_inf = true;
        range.right_inf = false;
        range.right = MulNode::make(SubNode::make(FloorDivNode::make(range.left, -bias), 1), -1);
      } else if (range_type == ExtRangeType::LORC) {
        range.left_inf = false;
        range.right_inf = true;
        range.left = MulNode::make(SubNode::make(FloorDivNode::make(AddNode::make(range.right, -bias-1), -bias), 1), -1);
      } else {
        // nothing to do
      }
    }
    
    scope_.push_back(range);
    VisitExpr(op->a);
    scope_.pop_back();
  }
}


Array<PrimExpr> relax_matrix_array_product(Matrix<int> &m, Array<PrimExpr> &v) {
  Array<PrimExpr> res;
  int rows = m.height();
  int cols = m.width();
  CHECK(cols <= (int)v.size()) << "Matrix-Array-Mult shape mismatch.\n";
  for (int i = 0; i < rows; ++i) {
    PrimExpr tmp = make_const(DataType::Int(32), 0);
    for (int j = 0; j < cols; ++j) {
      if (m[i][j] != 0) {
        tmp = AddNode::make(tmp, MulNode::make(v[j], m[i][j]));
      }
    }
    res.push_back(tmp);
  }
  return res;
}


bool check_identity(Matrix<int> &m, int dims) {
  int rows = m.height();
  int cols = m.width();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (i == j && i < dims && j < dims) {
        if (m[i][j] != 1) {
          return false;
        }
      } else {
        if (m[i][j] != 0) {
          return false;
        }
      }
    }
  }
  return true;
}


int ext_euclidean(int a, int b, int &x, int &y) {
  int r0 = a;
  int r1 = b;
  int s0 = 1;
  int s1 = 0;
  int t0 = 0;
  int t1 = 1;
  while (r1 != 0) {
    int q = r0 / r1;
    int r = r0 - q * r1;
    int s = s0 - q * s1;
    int t = t0 - q * t1;
    r0 = r1;
    r1 = r;
    s0 = s1;
    s1 = s;
    t0 = t1;
    t1 = t;
  }
  x = s0;
  y = t0;
  return r0;
}


bool divisible(int a, int b) {
  return b % a == 0;
}


int smith_normalize(Matrix<int> &trans, Matrix<int> &U, Matrix<int> &V) {
  int height = trans.height();
  int width = trans.width();
  CHECK(U.height() == height && U.width() == height) << "U matrix wrong shape: ("
        << U.height() << "x" << U.width() << ")\nExpected: (" << height << "x" << height << ")\n";
  CHECK(V.height() == width && V.width() == width) << "V matrix wrong shape: ("
        << V.height() << "x" << V.width() << ")\nExpected: (" << width << "x" << width << ")\n";
  // initialize U and V
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < height; ++j) {
      if (i == j) {
        U[i][j] = 1;
      } else {
        U[i][j] = 0;
      }
    }
  }
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      if (i == j) {
        V[i][j] = 1;
      } else {
        V[i][j] = 0;
      }
    }
  }

  // initialize alpha
  int a = 0;
  // initialize dimension
  int dim = 0;
  // helper functions
  std::function<bool(int &, int &)> has_next;
  has_next = [&a, &height, &width, &trans](int &i, int &j) {
    for (i = a; i < height; ++i) {
      for (j = a; j < width; ++j) {
        if (trans[i][j] != 0) {
          return true;
        }
      }
    }
    return false;
  };

  std::function<bool(int &)> col_non_div;
  col_non_div = [&a, &height, &trans](int &i) {
    for (i = a + 1; i < height; ++i) {
      if (!divisible(trans[a][a], trans[i][a])) {
        return true;
      }
    }
    return false;
  };

  std::function<bool(int &)> col_non_zero;
  col_non_zero = [&a, &height, &trans](int &i) {
    for (i = a + 1; i < height; ++i) {
      if (trans[i][a] != 0) {
        return true;
      }
    }
    return false;
  };

  std::function<bool(int &)> row_non_div;
  row_non_div = [&a, &width, &trans](int &j) {
    for (j = a + 1; j < width; ++j) {
      if (!divisible(trans[a][a], trans[a][j])) {
        return true;
      }
    }
    return false;
  };

  std::function<bool(int &)> row_non_zero;
  row_non_zero = [&a, &width, &trans](int &j) {
    for (j = a + 1; j < width; ++j) {
      if (trans[a][j] != 0) {
        return true;
      }
    }
    return false;
  };
  // outer most iteration
  bool stop = false;
  while (!stop) {
    stop = true;
    // index to non-zero value
    int pi, pj;
    // inner iteration
    while (has_next(pi, pj)) {
      trans.swap_row(a, pi);
      U.swap_row(a, pi);
      trans.swap_col(a, pj);
      V.swap_col(a, pj);

      bool changed = true;
      // clear row a and col a
      while (changed) {
        changed = false;
        int ppi, ppj;
        while (col_non_div(ppi)) {
          changed = true;
          // col a, row ppi, non-divisible
          int s, t;
          int z = ext_euclidean(trans[a][a], trans[ppi][a], s, t);
          int g = -trans[ppi][a] / z;
          int h = trans[a][a] / z;
          trans.row_transform(a, ppi, s, t, g, h);
          U.row_transform(a, ppi, s, t, g, h);
        }
        while (col_non_zero(ppi)) {
          changed = true;
          // col a, row ppi, non-zero
          int f = trans[ppi][a] / trans[a][a];
          trans.add_row(a, ppi, -f);
          U.add_row(a, ppi, -f);
        }
        while (row_non_div(ppj)) {
          changed = true;
          // col ppj, row a, non-divisible
          int s, t;
          int z = ext_euclidean(trans[a][a], trans[a][ppj], s, t);
          int g = -trans[a][ppj] / z;
          int h = trans[a][a] / z;
          trans.col_transform(a, ppj, s, t, g, h);
          V.col_transform(a, ppj, s, t, g, h);
        }
        while (row_non_zero(ppj)) {
          changed = true;
          // col ppj, row a, non-zero
          int f = trans[a][ppj] / trans[a][a];
          trans.add_col(a, ppj, -f);
          V.add_col(a, ppj, -f);
        }
      }
      // move to next row/col
      a = a + 1;
    }
    // record dimension
    dim = a;

    for (a = 0; a < dim; ++a) {
      if (trans[a][a] < 0) {
        trans.scale_col(a, -1);
        V.scale_col(a, -1);
      }
      if (a < dim - 1 && !divisible(trans[a][a], trans[a+1][a+1])) {
        trans.add_col(a + 1, a, 1);
        V.add_col(a + 1, a, 1);
        stop = false;
        break;
      }
    }
  }

  return dim;
}


}  // namespace tg

}  // namespace tvm