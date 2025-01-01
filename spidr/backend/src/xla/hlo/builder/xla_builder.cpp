/*
Copyright 2021 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <algorithm>
#include <cstring>
#include <string>

#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

#include "../../literal.h"
#include "../../shape.h"
#include "../../xla_data.pb.h"
#include "xla_builder.h"
#include "xla_computation.h"

const char* c_string_copy(std::string str) {
    char *res = NULL;
    auto len = str.length();
    res = (char *) malloc(len + 1);
    strncpy(res, str.c_str(), len);
    res[len] = '\0';
    return res;
}

extern "C" {
    int sizeof_XlaOp() {
        return sizeof(xla::XlaOp);
    }

    void set_array_XlaOp(XlaOp* arr, int idx, XlaOp* op) {
        reinterpret_cast<xla::XlaOp*>(arr)[idx] = *reinterpret_cast<xla::XlaOp*>(op);
    }

    void XlaOp_delete(XlaOp* s) {
        delete reinterpret_cast<xla::XlaOp*>(s);
    }

    XlaBuilder* XlaBuilder_new(const char* computation_name) {
        auto builder = new xla::XlaBuilder(computation_name);
        return reinterpret_cast<XlaBuilder*>(builder);
    }

    void XlaBuilder_delete(XlaBuilder* s) {
        delete reinterpret_cast<xla::XlaBuilder*>(s);
    }

    const char* XlaBuilder_name(XlaBuilder& s) {
        return c_string_copy(reinterpret_cast<xla::XlaBuilder&>(s).name());
    }

    XlaBuilder* CreateSubBuilder(XlaBuilder& s, const char* computation_name) {
        xla::XlaBuilder& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        std::unique_ptr<xla::XlaBuilder> sub_builder = s_.CreateSubBuilder(computation_name);
        return reinterpret_cast<XlaBuilder*>(sub_builder.release());
    }

    XlaComputation* XlaBuilder_Build(XlaBuilder& s, XlaOp& root) {
        auto& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        auto& root_ = reinterpret_cast<xla::XlaOp&>(root);
        xla::XlaComputation computation = *s_.Build(root_);
        xla::XlaComputation* non_stack = new xla::XlaComputation(std::move(computation));
        return reinterpret_cast<XlaComputation*>(non_stack);
    }

    const char* XlaBuilder_OpToString(XlaBuilder& s, XlaOp& op) {
        auto& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        auto& op_ = reinterpret_cast<xla::XlaOp&>(op);
        auto op_str = s_.OpToString(op_);
        return c_string_copy(op_str);
    }

    XlaOp* Parameter(XlaBuilder& builder, int parameter_number, Shape& shape, const char* name) {
        xla::XlaBuilder& builder_ = reinterpret_cast<xla::XlaBuilder&>(builder);
        xla::Shape& shape_ = reinterpret_cast<xla::Shape&>(shape);
        xla::XlaOp parameter = xla::Parameter(&builder_, parameter_number, shape_, name);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(parameter));
    }

    XlaOp* ConstantLiteral(XlaBuilder& builder, Literal& data) {
        xla::XlaBuilder& builder_ = reinterpret_cast<xla::XlaBuilder&>(builder);
        xla::Literal& data_ = reinterpret_cast<xla::Literal&>(data);
        xla::XlaOp op = ConstantLiteral(&builder_, data_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(op));
    }

    XlaOp* Broadcast(XlaOp& s, int* broadcast_sizes, int len) {
        xla::XlaOp s_ = reinterpret_cast<xla::XlaOp&>(s);
        int64_t bcs64[len];
        std::copy(broadcast_sizes, broadcast_sizes + len, bcs64);
        xla::XlaOp res = Broadcast(s_, absl::Span<const int64_t>(bcs64, len));
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* BroadcastInDim(
        XlaOp& s, int* out_dim_size, int ods_len, int* broadcast_dimensions, int bcd_len
    ) {
        int64_t ods64[ods_len];
        std::copy(out_dim_size, out_dim_size + ods_len, ods64);

        int64_t bcd64[bcd_len];
        std::copy(broadcast_dimensions, broadcast_dimensions + bcd_len, bcd64);

        xla::XlaOp res = BroadcastInDim(
            reinterpret_cast<xla::XlaOp&>(s),
            absl::Span<const int64_t>(ods64, ods_len),
            absl::Span<const int64_t>(bcd64, bcd_len)
        );
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Reshape(
        XlaOp& operand, int* dimensions, int dimensions_len, int* new_sizes, int new_sizes_len
    ) {
        auto operand_ = reinterpret_cast<xla::XlaOp&>(operand);

        int64_t dimensions64[dimensions_len];
        std::copy(dimensions, dimensions + dimensions_len, dimensions64);

        int64_t new_sizes64[new_sizes_len];
        std::copy(new_sizes, new_sizes + new_sizes_len, new_sizes64);

        xla::XlaOp res = xla::Reshape(
            operand_,
            absl::Span<const int64_t>(dimensions64, dimensions_len),
            absl::Span<const int64_t>(new_sizes64, new_sizes_len)
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Slice(
        XlaOp& operand,
        int* start_indices,
        int start_indices_len,
        int* limit_indices,
        int limit_indices_len,
        int* strides,
        int strides_len
    ) {
        auto operand_ = reinterpret_cast<xla::XlaOp&>(operand);

        int64_t start_indices64[start_indices_len];
        std::copy(start_indices, start_indices + start_indices_len, start_indices64);

        int64_t limit_indices64[limit_indices_len];
        std::copy(limit_indices, limit_indices + limit_indices_len, limit_indices64);

        int64_t strides64[strides_len];
        std::copy(strides, strides + strides_len, strides64);

        xla::XlaOp res = xla::Slice(
            operand_,
            absl::Span<const int64_t>(start_indices64, start_indices_len),
            absl::Span<const int64_t>(limit_indices64, limit_indices_len),
            absl::Span<const int64_t>(strides64, strides_len)
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* DynamicSlice(
        XlaOp& operand, XlaOp* start_indices, int start_indices_len, int* slice_sizes, int slice_sizes_len
    ) {
        auto operand_ = reinterpret_cast<xla::XlaOp&>(operand);
        auto start_indices_ = reinterpret_cast<xla::XlaOp*>(start_indices);

        int64_t slice_sizes64[slice_sizes_len];
        std::copy(slice_sizes, slice_sizes + slice_sizes_len, slice_sizes64);

        xla::XlaOp res = xla::DynamicSlice(
            operand_,
            absl::Span<xla::XlaOp>(start_indices_, start_indices_len),
            absl::Span<const int64_t>(slice_sizes64, slice_sizes_len)
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* ConcatInDim(XlaBuilder* builder, XlaOp* operands, int operands_len, int dimension) {
        auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        auto operands_ = reinterpret_cast<xla::XlaOp*>(operands);
        auto operands_span = absl::Span<const xla::XlaOp>(operands_, operands_len);

        xla::XlaOp res = xla::ConcatInDim(builder_, operands_span, (int64_t) dimension);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Select(XlaOp& pred, XlaOp& on_true, XlaOp& on_false) {
        auto& pred_ = reinterpret_cast<xla::XlaOp&>(pred);
        auto& on_true_ = reinterpret_cast<xla::XlaOp&>(on_true);
        auto& on_false_ = reinterpret_cast<xla::XlaOp&>(on_false);

        xla::XlaOp res = xla::Select(pred_, on_true_, on_false_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Tuple(XlaBuilder* builder, XlaOp* elements, int elements_len) {
        auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        auto elements_ = reinterpret_cast<xla::XlaOp*>(elements);
        auto elements_span = absl::Span<const xla::XlaOp>(elements_, elements_len);

        xla::XlaOp res = xla::Tuple(builder_, elements_span);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* GetTupleElement(XlaOp& tuple_data, int index) {
        auto& tuple_data_ = reinterpret_cast<xla::XlaOp&>(tuple_data);
        xla::XlaOp res = xla::GetTupleElement(tuple_data_, (int64_t) index);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}

XlaOp* unaryOp(std::function<xla::XlaOp(xla::XlaOp)> op, XlaOp& operand) {
    xla::XlaOp res = op(reinterpret_cast<xla::XlaOp&>(operand));
    return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
}

XlaOp* binOp(
    std::function<xla::XlaOp(
        xla::XlaOp, xla::XlaOp, absl::Span<const int64_t> broadcast_dimensions
    )> op,
    XlaOp& lhs,
    XlaOp& rhs
) {
    auto& lhs_ = reinterpret_cast<xla::XlaOp&>(lhs);
    auto& rhs_ = reinterpret_cast<xla::XlaOp&>(rhs);
    xla::XlaOp res = op(lhs_, rhs_, {});
    return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
}

extern "C" {
    XlaOp* Eq(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Eq, lhs, rhs); }
    XlaOp* Ne(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Ne, lhs, rhs); }
    XlaOp* Ge(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Ge, lhs, rhs); }
    XlaOp* Gt(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Gt, lhs, rhs); }
    XlaOp* Lt(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Lt, lhs, rhs); }
    XlaOp* Le(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Le, lhs, rhs); }

    XlaOp* Dot(XlaOp& lhs, XlaOp& rhs) {
        auto& lhs_ = reinterpret_cast<xla::XlaOp&>(lhs);
        auto& rhs_ = reinterpret_cast<xla::XlaOp&>(rhs);
        xla::XlaOp res = xla::Dot(lhs_, rhs_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* DotGeneral(XlaOp& lhs, XlaOp& rhs, DotDimensionNumbers& dimension_numbers) {
        auto& lhs_ = reinterpret_cast<xla::XlaOp&>(lhs);
        auto& rhs_ = reinterpret_cast<xla::XlaOp&>(rhs);
        auto& dimension_numbers_ = reinterpret_cast<xla::DotDimensionNumbers&>(dimension_numbers);
        xla::XlaOp res = xla::DotGeneral(lhs_, rhs_, dimension_numbers_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* TriangularSolve(
        XlaOp& a, XlaOp& b, int left_side, int lower, int unit_diagonal, int transpose_a
    ) {
        auto& a_ = reinterpret_cast<xla::XlaOp&>(a);
        auto& b_ = reinterpret_cast<xla::XlaOp&>(b);

        xla::XlaOp res = xla::TriangularSolve(
            a_,
            b_,
            (bool) left_side,
            (bool) lower,
            (bool) unit_diagonal,
            (xla::TriangularSolveOptions::Transpose) transpose_a
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Cholesky(XlaOp& a, int lower) {
        auto& a_ = reinterpret_cast<xla::XlaOp&>(a);
        xla::XlaOp res = xla::Cholesky(a_, (bool) lower);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Add(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Add, lhs, rhs); }
    XlaOp* Sub(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Sub, lhs, rhs); }
    XlaOp* Mul(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Mul, lhs, rhs); }
    XlaOp* Div(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Div, lhs, rhs); }
    XlaOp* Rem(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Rem, lhs, rhs); }
    XlaOp* Max(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Max, lhs, rhs); }
    XlaOp* Min(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Min, lhs, rhs); }

    XlaOp* And(XlaOp& lhs, XlaOp& rhs) {
        auto& lhs_ = reinterpret_cast<xla::XlaOp&>(lhs);
        auto& rhs_ = reinterpret_cast<xla::XlaOp&>(rhs);
        xla::XlaOp res = xla::And(lhs_, rhs_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Or(XlaOp& lhs, XlaOp& rhs) {
        auto& lhs_ = reinterpret_cast<xla::XlaOp&>(lhs);
        auto& rhs_ = reinterpret_cast<xla::XlaOp&>(rhs);
        xla::XlaOp res = xla::Or(lhs_, rhs_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Not(XlaOp& operand) { return unaryOp(xla::Not, operand); }

    XlaOp* Reduce(
        XlaOp& operand,
        XlaOp& init_value,
        const XlaComputation& computation,
        int* dimensions_to_reduce,
        int dimensions_to_reduce_len
    ) {
        int64_t dimensions_to_reduce64[dimensions_to_reduce_len];
        std::copy(
            dimensions_to_reduce,
            dimensions_to_reduce + dimensions_to_reduce_len,
            dimensions_to_reduce64
        );

        xla::XlaOp res = xla::Reduce(
            reinterpret_cast<xla::XlaOp&>(operand),
            reinterpret_cast<xla::XlaOp&>(init_value),
            reinterpret_cast<const xla::XlaComputation&>(computation),
            absl::Span<const int64_t>(dimensions_to_reduce64, dimensions_to_reduce_len)
        );
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Abs(XlaOp& operand) { return unaryOp(xla::Abs, operand); }
    XlaOp* Exp(XlaOp& operand) { return unaryOp(xla::Exp, operand); }
    XlaOp* Floor(XlaOp& operand) { return unaryOp(xla::Floor, operand); }
    XlaOp* Ceil(XlaOp& operand) { return unaryOp(xla::Ceil, operand); }
    XlaOp* Log(XlaOp& operand) { return unaryOp(xla::Log, operand); }
    XlaOp* Logistic(XlaOp& operand) { return unaryOp(xla::Logistic, operand); }
    XlaOp* Cos(XlaOp& operand) { return unaryOp(xla::Cos, operand); }
    XlaOp* Sin(XlaOp& operand) { return unaryOp(xla::Sin, operand); }
    XlaOp* Tanh(XlaOp& operand) { return unaryOp(xla::Tanh, operand); }
    XlaOp* Sqrt(XlaOp& operand) { return unaryOp(xla::Sqrt, operand); }

    XlaOp* Pow(XlaOp& lhs, XlaOp& rhs) { return binOp(xla::Pow, lhs, rhs); }

    XlaOp* Iota(XlaBuilder* builder, Shape& shape, int iota_dimension) {
        auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        auto& shape_ = reinterpret_cast<xla::Shape&>(shape);
        xla::XlaOp res = xla::Iota(builder_, shape_, iota_dimension);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* ConvertElementType(XlaOp& operand, int new_element_type) {
        auto& operand_ = reinterpret_cast<xla::XlaOp&>(operand);
        auto new_element_type_ = (xla::PrimitiveType) new_element_type;
        xla::XlaOp res = xla::ConvertElementType(operand_, new_element_type_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Neg(XlaOp& operand) { return unaryOp(xla::Neg, operand); }

    XlaOp* Transpose(XlaOp& operand, int* permutation, int rank) {
        auto& operand_ = reinterpret_cast<xla::XlaOp&>(operand);
        int64_t permutation64[rank];
        std::copy(permutation, permutation + rank, permutation64);
        auto permutation_span = absl::Span<const int64_t>(permutation64, rank);

        xla::XlaOp res = xla::Transpose(operand_, permutation_span);

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Rev(XlaOp& operand, int* dimensions, int dimensions_len) {
        auto& operand_ = reinterpret_cast<xla::XlaOp&>(operand);
        int64_t dimensions64[dimensions_len];
        std::copy(dimensions, dimensions + dimensions_len, dimensions64);
        auto dimensions_span = absl::Span<const int64_t>(dimensions64, dimensions_len);

        xla::XlaOp res = xla::Rev(operand_, dimensions_span);

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Sort(
        XlaOp* operands, int operands_len, XlaComputation& comparator, int dimension, int is_stable
    ) {
        xla::XlaOp* operands_ = reinterpret_cast<xla::XlaOp*>(operands);
        auto operands_span = absl::Span<const xla::XlaOp>(operands_, operands_len);
        auto& comparator_ = reinterpret_cast<xla::XlaComputation&>(comparator);
        xla::XlaOp res = xla::Sort(operands_span, comparator_, dimension, (bool) is_stable);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Map(
        XlaBuilder* builder,
        XlaOp* operands,
        int operands_len,
        XlaComputation& computation,
        int* dimensions,
        int dimensions_len,
        XlaOp* static_operands,
        int static_operands_len
    ) {
        xla::XlaBuilder* builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        xla::XlaOp* operands_ = reinterpret_cast<xla::XlaOp*>(operands);
        xla::XlaComputation& computation_ = reinterpret_cast<xla::XlaComputation&>(computation);
        xla::XlaOp* static_operands_ = reinterpret_cast<xla::XlaOp*>(static_operands);

        int64_t dimensions64[dimensions_len];
        std::copy(dimensions, dimensions + dimensions_len, dimensions64);

        auto operands_span = absl::Span<const xla::XlaOp>(operands_, operands_len);
        auto dimensions_span = absl::Span<const int64_t>(dimensions64, dimensions_len);
        auto static_operands_span =
            absl::Span<const xla::XlaOp>(static_operands_, static_operands_len);

        xla::XlaOp res = xla::Map(
            builder_, operands_span, computation_, dimensions_span, static_operands_span
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* RngBitGenerator(int algorithm, XlaOp& initial_state, Shape& shape) {
        auto algorithm_ = (xla::RandomAlgorithm) algorithm;
        auto initial_state_ = reinterpret_cast<xla::XlaOp&>(initial_state);
        auto shape_ = reinterpret_cast<xla::Shape&>(shape);
        xla::XlaOp res = xla::RngBitGenerator(algorithm_, initial_state_, shape_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Conditional(
        XlaOp& predicate,
        XlaOp& true_operand,
        const XlaComputation& true_computation,
        XlaOp& false_operand,
        const XlaComputation& false_computation
    ) {
        auto& predicate_ = reinterpret_cast<xla::XlaOp&>(predicate);
        auto& true_operand_ = reinterpret_cast<xla::XlaOp&>(true_operand);
        auto& true_computation_ = reinterpret_cast<const xla::XlaComputation&>(true_computation);
        auto& false_operand_ = reinterpret_cast<xla::XlaOp&>(false_operand);
        auto& false_computation_ = reinterpret_cast<const xla::XlaComputation&>(false_computation);

        xla::XlaOp res = xla::Conditional(
            predicate_,
            true_operand_,
            true_computation_,
            false_operand_,
            false_computation_
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}
