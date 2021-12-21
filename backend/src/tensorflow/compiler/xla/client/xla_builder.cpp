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
#include <absl/types/span.h>
#include <tensorflow/compiler/xla/array.h>
#include <tensorflow/compiler/xla/client/client_library.h>
#include <tensorflow/compiler/xla/client/local_client.h>
#include <tensorflow/compiler/xla/client/xla_builder.h>
#include <tensorflow/compiler/xla/literal_util.h>
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
    using namespace xla;

    /*
     *
     *
     * FFI
     *
     *
     */

    void free_shape(int* shape) {
        free(shape);
    }

    int* alloc_shape(int rank) {
        int* shape = new int[rank];
        return shape;
    }

    void array_set_i32(int* arr, int idx, int value) {
        arr[idx] = value;
    }

    /*
     *
     *
     * XlaOp
     *
     *
     */

    struct c__XlaOp;

    void c__XlaOp_del(c__XlaOp* s) {
        delete reinterpret_cast<XlaOp*>(s);
    }

    c__XlaOp* c__XlaOp_operator_neg(c__XlaOp& s) {
        auto res = new XlaOp();
        *res = -reinterpret_cast<XlaOp&>(s);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    c__XlaOp* c__XlaOp_operator_add(c__XlaOp& x, c__XlaOp& y) {
        auto res = new XlaOp();
        *res = Add(reinterpret_cast<XlaOp&>(x), reinterpret_cast<XlaOp&>(y));
        return reinterpret_cast<c__XlaOp*>(res);
    }

    c__XlaOp* c__XlaOp_operator_sub(c__XlaOp& x, c__XlaOp& y) {
        auto res = new XlaOp();
        *res = reinterpret_cast<XlaOp&>(x) - reinterpret_cast<XlaOp&>(y);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    c__XlaOp* c__XlaOp_operator_mul(c__XlaOp& x, c__XlaOp& y) {
        auto res = new XlaOp();
        *res = reinterpret_cast<XlaOp&>(x) * reinterpret_cast<XlaOp&>(y);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    c__XlaOp* c__XlaOp_operator_div(c__XlaOp& x, c__XlaOp& y) {
        auto res = new XlaOp();
        *res = reinterpret_cast<XlaOp&>(x) / reinterpret_cast<XlaOp&>(y);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    c__XlaOp* c__XlaOp_operator_mod(c__XlaOp& x, c__XlaOp& y) {
        auto res = new XlaOp();
        *res = reinterpret_cast<XlaOp&>(x) % reinterpret_cast<XlaOp&>(y);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    /*
     *
     *
     * XlaBuilder
     *
     *
     */

    struct c__XlaBuilder;

    c__XlaBuilder* c__XlaBuilder_new(const char* computation_name) {
        auto builder = new XlaBuilder(computation_name);
        return reinterpret_cast<c__XlaBuilder*>(builder);
    }

    void c__XlaBuilder_del(c__XlaBuilder* s) {
        delete reinterpret_cast<XlaBuilder*>(s);
    }

    const char* c__XlaBuilder_name(c__XlaBuilder& s) {
        return c_string_copy(reinterpret_cast<XlaBuilder&>(s).name());
    }

    const char* c__XlaBuilder_OpToString(c__XlaBuilder& s, c__XlaOp& op) {
        auto& s_ = reinterpret_cast<XlaBuilder&>(s);
        auto& op_ = reinterpret_cast<XlaOp&>(op);
        auto op_str = s_.OpToString(op_);
        return c_string_copy(op_str);
    }

    /*
     *
     *
     * Free functions
     *
     *
     */

    c__XlaOp* c__Eq(c__XlaOp& lhs, c__XlaOp& rhs) {
        auto& lhs_ = reinterpret_cast<XlaOp&>(lhs);
        auto& rhs_ = reinterpret_cast<XlaOp&>(rhs);
        auto res = new XlaOp();
        *res = Eq(lhs_, rhs_);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    struct c__Literal;

    c__XlaOp* c__ConstantLiteral(c__XlaBuilder& builder, c__Literal& data) {
        XlaBuilder& builder_ = reinterpret_cast<XlaBuilder&>(builder);
        xla::Literal& data_ = reinterpret_cast<xla::Literal&>(data);
        XlaOp* op = new XlaOp();
        *op = ConstantLiteral(&builder_, data_);
        return reinterpret_cast<c__XlaOp*>(op);
    }

    /*
     *
     *
     * Literal
     *
     *
     */

    void c__Literal_delete(c__Literal* lit) {
        delete reinterpret_cast<Literal*>(lit);
    }

    int c__Literal_Get_int(
        c__Literal& lit,
        int* indices  // TODO should be a Span
    ) {
        Literal& lit_ = reinterpret_cast<Literal&>(lit);
        int64 rank = lit_.shape().rank();
        tensorflow::int64 indices64[rank];
        for (int i = 0; i < rank; i++) {
            indices64[i] = indices[i];
        }
        auto multi_index = absl::Span<const tensorflow::int64>(indices64, rank);
        return lit_.Get<int>(multi_index);
    }

    void c__Literal_Set_int(
        c__Literal& lit,
        int* indices,  // TODO should be a Span
        int value
    ) {
        Literal& lit_ = reinterpret_cast<Literal&>(lit);
        int64 rank = lit_.shape().rank();
        tensorflow::int64 indices64[rank];
        for (int i = 0; i < rank; i++) {
            indices64[i] = indices[i];
        }

        auto multi_index = absl::Span<const tensorflow::int64>(indices64, rank);
        lit_.Set<int>(multi_index, value);
    }

    int to_int(c__Literal& lit) {
        return *(int*) reinterpret_cast<Literal&>(lit).untyped_data();
    }

    double to_double(c__Literal& lit) {
        return *(double*) reinterpret_cast<Literal&>(lit).untyped_data();
    }

    c__Literal* c__Literal_new(int* shape, int rank) {
        int64 shape64[rank];
        for (int i = 0; i < rank; i++) {
            shape64[i] = shape[i];
        }

        const std::vector<bool> dynamic_dimensions(rank, false);

        auto xla_shape = ShapeUtil::MakeShape(
            PrimitiveType::S32,
            absl::Span<const int64>(shape64, rank),
            dynamic_dimensions
        );

        xla::Literal* lit = new xla::Literal(xla_shape, true);
        return reinterpret_cast<c__Literal*>(lit);
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

    c__Literal* eval(c__XlaOp& op) {
        XlaOp& op_ = reinterpret_cast<XlaOp&>(op);

        XlaComputation computation = op_.builder()->Build().ConsumeValueOrDie();
        ExecutionProfile profile;
        Literal lit = ClientLibrary::LocalClientOrDie()
            ->ExecuteAndTransfer(computation, {}, nullptr, &profile)
            .ConsumeValueOrDie();

        xla::Literal* res = new xla::Literal(lit.shape(), true);
        *res = lit.Clone();
        return reinterpret_cast<c__Literal*>(res);
    }
}
