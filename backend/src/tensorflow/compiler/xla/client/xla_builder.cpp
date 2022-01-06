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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include "src/ffi.h"
#include "src/tensorflow/compiler/xla/literal.h"

#include "xla_builder.h"
#include "xla_computation.h"

extern "C" {
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

    XlaComputation* XlaBuilder_Build(XlaBuilder& s) {
        xla::XlaBuilder& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        xla::XlaComputation* res = new xla::XlaComputation();
        *res = s_.Build().ConsumeValueOrDie();
        return reinterpret_cast<XlaComputation*>(res);
    }

    const char* XlaBuilder_OpToString(XlaBuilder& s, XlaOp& op) {
        auto& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        auto& op_ = reinterpret_cast<xla::XlaOp&>(op);
        auto op_str = s_.OpToString(op_);
        return c_string_copy(op_str);
    }

    XlaOp* Broadcast(XlaOp& s, int* broadcast_sizes, int len) {
        xla::XlaOp s_ = reinterpret_cast<xla::XlaOp&>(s);
        xla::int64 bcs64[len];
        std::copy(broadcast_sizes, broadcast_sizes + len, bcs64);
        auto res = new xla::XlaOp();
        *res = Broadcast(s_, absl::Span<const xla::int64>(bcs64, len));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* BroadcastInDim(
        XlaOp& s, int* out_dim_size, int ods_len, int* broadcast_dimensions, int bcd_len
    ) {
        xla::int64 ods64[ods_len];
        std::copy(out_dim_size, out_dim_size + ods_len, ods64);

        xla::int64 bcd64[bcd_len];
        std::copy(broadcast_dimensions, broadcast_dimensions + bcd_len, bcd64);

        auto res = new xla::XlaOp();
        *res = BroadcastInDim(
            reinterpret_cast<xla::XlaOp&>(s),
            absl::Span<const xla::int64>(ods64, ods_len),
            absl::Span<const xla::int64>(bcd64, bcd_len)
        );
        return reinterpret_cast<XlaOp*>(res);
    }
}

XlaOp* unaryOp(std::function<xla::XlaOp(xla::XlaOp)> op, XlaOp& operand) {
    auto res = new xla::XlaOp();
    *res = op(reinterpret_cast<xla::XlaOp&>(operand));
    return reinterpret_cast<XlaOp*>(res);
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
    auto res = new xla::XlaOp();
    *res = op(lhs_, rhs_, {});
    return reinterpret_cast<XlaOp*>(res);
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

    XlaOp* Neg(XlaOp& operand) { return unaryOp(xla::Neg, operand); }
    XlaOp* Abs(XlaOp& operand) { return unaryOp(xla::Abs, operand); }

    XlaOp* ConstantLiteral(XlaBuilder& builder, Literal& data) {
        xla::XlaBuilder& builder_ = reinterpret_cast<xla::XlaBuilder&>(builder);
        xla::Literal& data_ = reinterpret_cast<xla::Literal&>(data);
        xla::XlaOp* op = new xla::XlaOp();
        *op = ConstantLiteral(&builder_, data_);
        return reinterpret_cast<XlaOp*>(op);
    }
}
