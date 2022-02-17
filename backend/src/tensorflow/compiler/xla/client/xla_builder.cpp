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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "../literal.h"
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

    XlaComputation* XlaBuilder_Build(XlaBuilder& s) {
        xla::XlaBuilder& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        xla::XlaComputation computation = s_.Build().ConsumeValueOrDie();
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
        xla::int64 bcs64[len];
        std::copy(broadcast_sizes, broadcast_sizes + len, bcs64);
        xla::XlaOp res = Broadcast(s_, absl::Span<const xla::int64>(bcs64, len));
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* BroadcastInDim(
        XlaOp& s, int* out_dim_size, int ods_len, int* broadcast_dimensions, int bcd_len
    ) {
        xla::int64 ods64[ods_len];
        std::copy(out_dim_size, out_dim_size + ods_len, ods64);

        xla::int64 bcd64[bcd_len];
        std::copy(broadcast_dimensions, broadcast_dimensions + bcd_len, bcd64);

        xla::XlaOp res = BroadcastInDim(
            reinterpret_cast<xla::XlaOp&>(s),
            absl::Span<const xla::int64>(ods64, ods_len),
            absl::Span<const xla::int64>(bcd64, bcd_len)
        );
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Reshape(
        XlaOp& operand, int* dimensions, int dimensions_len, int* new_sizes, int new_sizes_len
    ) {
        auto operand_ = reinterpret_cast<xla::XlaOp&>(operand);

        xla::int64 dimensions64[dimensions_len];
        std::copy(dimensions, dimensions + dimensions_len, dimensions64);

        xla::int64 new_sizes64[new_sizes_len];
        std::copy(new_sizes, new_sizes + new_sizes_len, new_sizes64);

        xla::XlaOp res = xla::Reshape(
            operand_,
            absl::Span<const xla::int64>(dimensions64, dimensions_len),
            absl::Span<const xla::int64>(new_sizes64, new_sizes_len)
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

        xla::int64 start_indices64[start_indices_len];
        std::copy(start_indices, start_indices + start_indices_len, start_indices64);

        xla::int64 limit_indices64[limit_indices_len];
        std::copy(limit_indices, limit_indices + limit_indices_len, limit_indices64);

        xla::int64 strides64[strides_len];
        std::copy(strides, strides + strides_len, strides64);

        xla::XlaOp res = xla::Slice(
            operand_,
            absl::Span<const xla::int64>(start_indices64, start_indices_len),
            absl::Span<const xla::int64>(limit_indices64, limit_indices_len),
            absl::Span<const xla::int64>(strides64, strides_len)
        );

        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}

XlaOp* unaryOp(std::function<xla::XlaOp(xla::XlaOp)> op, XlaOp& operand) {
    xla::XlaOp res = op(reinterpret_cast<xla::XlaOp&>(operand));
    return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
}

XlaOp* binOp(
    std::function<xla::XlaOp(
        xla::XlaOp, xla::XlaOp, absl::Span<const xla::int64> broadcast_dimensions
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
        xla::int64 dimensions_to_reduce64[dimensions_to_reduce_len];
        std::copy(
            dimensions_to_reduce,
            dimensions_to_reduce + dimensions_to_reduce_len,
            dimensions_to_reduce64
        );

        xla::XlaOp res = xla::Reduce(
            reinterpret_cast<xla::XlaOp&>(operand),
            reinterpret_cast<xla::XlaOp&>(init_value),
            reinterpret_cast<const xla::XlaComputation&>(computation),
            absl::Span<const xla::int64>(dimensions_to_reduce64, dimensions_to_reduce_len)
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
    XlaOp* Neg(XlaOp& operand) { return unaryOp(xla::Neg, operand); }

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

        xla::int64 dimensions64[dimensions_len];
        std::copy(dimensions, dimensions + dimensions_len, dimensions64);

        auto operands_span = absl::Span<const xla::XlaOp>(operands_, operands_len);
        auto dimensions_span = absl::Span<const xla::int64>(dimensions64, dimensions_len);
        auto static_operands_span = absl::Span<const xla::XlaOp>(static_operands_, static_operands_len);

        xla::XlaOp res = xla::Map(builder_, operands_span, computation_, dimensions_span, static_operands_span);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}
