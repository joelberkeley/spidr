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
#include "absl/types/span.h"
#include <tensorflow/compiler/xla/array.h>
#include <tensorflow/compiler/xla/client/client_library.h>
#include <tensorflow/compiler/xla/client/local_client.h>
#include <tensorflow/compiler/xla/client/xla_builder.h>
#include <tensorflow/compiler/xla/literal_util.h>

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
        *res = reinterpret_cast<XlaOp&>(x) + reinterpret_cast<XlaOp&>(y);
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

    /*
     *
     *
     * Custom utility functions
     *
     * Unlike the functions above, these are not just a minimal C layer round the XLA API
     *
     *
     */

    c__XlaOp* constant(c__XlaBuilder& builder, void* data, int64* shape) {
        XlaBuilder& builder_ = reinterpret_cast<XlaBuilder&>(builder);

        int size = 1;
        for (int i = 0; i < sizeof(shape) / sizeof(int); i++) {
            size *= shape[i];
        }

        auto shape_ = absl::Span<const int64>(shape, size);
        auto array = Array<int32>(shape_);
        // todo this next line doesn't appear possible - looks like we can only construct up to 4D arrays.
        // If this is true, which seems unlikely, there are probably simpler ways to do this.
        array.SetValues(data);
        Literal lit = LiteralUtil::CreateFromArray(array);
        XlaOp* res = new XlaOp();
        *res = ConstantLiteral(&builder_, lit);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    int32* alloc_shape(int32 rank) {
        int32* shape = new int32[rank];
        return shape;
    }

    void array_set_i32(int32* arr, int idx, int32 value) {
        arr[idx] = value;
    }

    void array_set_f64(double* arr, int idx, double value) {
        arr[idx] = value;
    }

    void* array_alloc_i32(int32* shape) {
        size_t rank = sizeof(shape) / sizeof(shape[0]);

        if (rank == 1) {
            int32* res = new int32[shape[0]];
            return res;
        } else if (rank > 1) {
            void** res = new void*[shape[0]];
            int32 trailing_shape[rank - 1];
            for (int i = 0; i < rank - 1; i++) {
                trailing_shape[i] = shape[i + 1];
            }
            for (int i = 0; i < shape[0] - 1; i++) {
                res[i] = array_alloc_i32(trailing_shape);
            }
            return res;
        } else {
            std::cout << "Invalid system state: memory cannot be allocated for array with rank 0" << std::endl;
            return NULL;
        }
    }

    void test_put(int32** arr) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << arr[i][j] << std::endl;
            }
        }
    }

    void* eval(c__XlaOp& op) {
        XlaOp& op_ = reinterpret_cast<XlaOp&>(op);

        XlaComputation computation = op_.builder()->Build().ConsumeValueOrDie();
        ExecutionProfile profile;
        Literal lit = ClientLibrary::LocalClientOrDie()
            ->ExecuteAndTransfer(computation, {}, nullptr, &profile)
            .ConsumeValueOrDie();

        // todo is this a shallow copy when we need a deep copy?
        int64 size = lit.size_bytes();
        void* res = malloc(size);;
        memcpy(res, lit.untyped_data(), size);

        return res;
    }

    int32 eval_i32(c__XlaOp& op) {
        return *(int32*) eval(op);
    }

    double eval_f64(c__XlaOp& op) {
        return *(double*) eval(op);
    }

    int32 index_i32(void* ptr, int32 idx) {
        return ((int32*) ptr)[idx];
    }

    double index_f64(void* ptr, int32 idx) {
        return ((double*) ptr)[idx];
    }

    void* index_void_ptr(void** ptr, int32 idx) {
        return ptr[idx];
    }
}
