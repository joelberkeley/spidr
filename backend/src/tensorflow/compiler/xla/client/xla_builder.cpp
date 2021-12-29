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

#include <absl/types/span.h>
#include <tensorflow/compiler/xla/client/client_library.h>
#include <tensorflow/compiler/xla/client/local_client.h>
#include <tensorflow/compiler/xla/client/xla_builder.h>
#include <tensorflow/compiler/xla/literal.h>
#include <tensorflow/compiler/xla/shape.h>
#include <tensorflow/compiler/xla/shape_util.h>
#include <tensorflow/compiler/xla/xla_data.pb.h>

// Return a pointer to a new, heap-allocated, null-terminated C string.
const char* c_string_copy(std::string str) {
    char *res = NULL;
    auto len = str.length();
    res = (char *) malloc(len + 1);
    strncpy(res, str.c_str(), len);
    res[len] = '\0';
    return res;
}



extern "C" {
    /*
     *
     *
     * FFI
     *
     *
     */

    void free_int_array(int* arr) {
        free(arr);
    }

    int* alloc_int_array(int len) {
        int* arr = new int[len];
        return arr;
    }

    void set_array_int(int* arr, int idx, int value) {
        arr[idx] = value;
    }

    /*
     *
     *
     * XlaOp
     *
     *
     */

    struct XlaOp;

    void XlaOp_delete(XlaOp* s) {
        delete reinterpret_cast<xla::XlaOp*>(s);
    }

    /*
     *
     *
     * XlaBuilder
     *
     *
     */

    struct XlaBuilder;

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

    const char* XlaBuilder_OpToString(XlaBuilder& s, XlaOp& op) {
        auto& s_ = reinterpret_cast<xla::XlaBuilder&>(s);
        auto& op_ = reinterpret_cast<xla::XlaOp&>(op);
        auto op_str = s_.OpToString(op_);
        return c_string_copy(op_str);
    }

    /*
     *
     *
     * Literal
     *
     *
     */

    struct Literal;

    Literal* Literal_new(int* shape, int rank, int primitive_type) {
        xla::int64 shape64[rank];
        std::copy(shape, shape + rank, shape64);

        const std::vector<bool> dynamic_dimensions(rank, false);

        xla::Shape xla_shape = xla::ShapeUtil::MakeShape(
            (xla::PrimitiveType) primitive_type,
            absl::Span<const xla::int64>(shape64, rank),
            dynamic_dimensions
        );

        xla::Literal* lit = new xla::Literal(xla_shape, true);
        return reinterpret_cast<Literal*>(lit);
    }

    void Literal_delete(Literal* lit) {
        delete reinterpret_cast<xla::Literal*>(lit);
    }
}

template <typename NativeT>
NativeT Literal_Get(Literal& lit, int* indices) {
    xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
    xla::int64 rank = lit_.shape().rank();
    xla::int64 multi_index[rank];
    std::copy(indices, indices + rank, multi_index);
    return lit_.Get<NativeT>(absl::Span<const xla::int64>(multi_index, rank));
};

template <typename NativeT>
void Literal_Set(Literal& lit, int* indices, NativeT value) {
    xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
    xla::int64 rank = lit_.shape().rank();
    xla::int64 multi_index[rank];
    std::copy(indices, indices + rank, multi_index);
    lit_.Set<NativeT>(absl::Span<const xla::int64>(multi_index, rank), value);
};

extern "C" {
    int Literal_Get_bool(Literal& lit, int* indices) {
        return (int) Literal_Get<bool>(lit, indices);
    }

    int Literal_Get_int(Literal& lit, int* indices) {
        return Literal_Get<int>(lit, indices);
    }

    double Literal_Get_double(Literal& lit, int* indices) {
        return Literal_Get<double>(lit, indices);
    }

    void Literal_Set_bool(Literal& lit, int* indices, int value) {
        Literal_Set<bool>(lit, indices, (bool) value);
    }

    void Literal_Set_int(Literal& lit, int* indices, int value) {
        Literal_Set<int>(lit, indices, value);
    }

    void Literal_Set_double(Literal& lit, int* indices, double value) {
        Literal_Set<double>(lit, indices, value);
    }

    /*
     *
     *
     * Free functions
     *
     *
     */

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

    XlaOp* Neg(XlaOp& s) {
        auto res = new xla::XlaOp();
        *res = Neg(reinterpret_cast<xla::XlaOp&>(s));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* Add(XlaOp& x, XlaOp& y) {
        auto res = new xla::XlaOp();
        *res = Add(reinterpret_cast<xla::XlaOp&>(x), reinterpret_cast<xla::XlaOp&>(y));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* Sub(XlaOp& x, XlaOp& y) {
        auto res = new xla::XlaOp();
        *res = Sub(reinterpret_cast<xla::XlaOp&>(x), reinterpret_cast<xla::XlaOp&>(y));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* Mul(XlaOp& x, XlaOp& y) {
        auto res = new xla::XlaOp();
        *res = Mul(reinterpret_cast<xla::XlaOp&>(x), reinterpret_cast<xla::XlaOp&>(y));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* Div(XlaOp& x, XlaOp& y) {
        auto res = new xla::XlaOp();
        *res = Div(reinterpret_cast<xla::XlaOp&>(x), reinterpret_cast<xla::XlaOp&>(y));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* Rem(XlaOp& x, XlaOp& y) {
        auto res = new xla::XlaOp();
        *res = Rem(reinterpret_cast<xla::XlaOp&>(x), reinterpret_cast<xla::XlaOp&>(y));
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* Eq(XlaOp& lhs, XlaOp& rhs) {
        auto& lhs_ = reinterpret_cast<xla::XlaOp&>(lhs);
        auto& rhs_ = reinterpret_cast<xla::XlaOp&>(rhs);
        auto res = new xla::XlaOp();
        *res = Eq(lhs_, rhs_);
        return reinterpret_cast<XlaOp*>(res);
    }

    XlaOp* ConstantLiteral(XlaBuilder& builder, Literal& data) {
        xla::XlaBuilder& builder_ = reinterpret_cast<xla::XlaBuilder&>(builder);
        xla::Literal& data_ = reinterpret_cast<xla::Literal&>(data);
        xla::XlaOp* op = new xla::XlaOp();
        *op = ConstantLiteral(&builder_, data_);
        return reinterpret_cast<XlaOp*>(op);
    }

    /*
     *
     *
     * Custom utility functions
     *
     * Unlike the functions above, these are not just a minimal C layer round the XLA API
     *
     *
     */

    Literal* eval(XlaOp& op) {
        xla::XlaOp& op_ = reinterpret_cast<xla::XlaOp&>(op);

        xla::XlaComputation computation = op_.builder()->Build().ConsumeValueOrDie();
        xla::ExecutionProfile profile;
        xla::Literal lit = xla::ClientLibrary::LocalClientOrDie()
            ->ExecuteAndTransfer(computation, {}, nullptr, &profile)
            .ConsumeValueOrDie();

        xla::Literal* res = new xla::Literal(lit.shape(), true);
        *res = lit.Clone();
        return reinterpret_cast<Literal*>(res);
    }
}
