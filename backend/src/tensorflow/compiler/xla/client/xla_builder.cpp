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
#include "../../../../ffi.cpp"
#include "../literal.cpp"

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

    int* alloc_shape(int rank) {
        std::cout << "alloc_shape ..." << std::endl;
        std::cout << "rank " << rank << std::endl;
        int* shape = new int[rank];
        std::cout << "alloc_shape ... returning shape" << std::endl;
        return shape;
    }

    // todo all types should be FFI-compatible primitives e.g. int, no?
    void array_set_i32(int* arr, int idx, int value) {
        std::cout << "array_set_i32 " << idx << " " << value << std::endl;

        std::cout << "array before set: ";
        for (int i = 0; i < idx; i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;

        arr[idx] = value;
        std::cout << "indexing worked " << std::endl;
        std::cout << "array after set: ";
        for (int i = 0; i <= idx; i++) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    void array_set_f64(double* arr, int idx, double value) {
        arr[idx] = value;
    }

    void* array_alloc_i32(const int* shape, int rank) {
        std::cout << "array_alloc_i32 ... " << std::endl;
        std::cout << "rank " << rank << std::endl;
        std::cout << "shape ";
        for (int i = 0; i < rank; i++) {
            std::cout << shape[i] << " ";
        }
        std::cout << std::endl;

        if (rank == 1) {
            std::cout << "array_alloc_i32 ... rank 1" << std::endl;
            int* res = new int[shape[0]];
            return res;
        } else if (rank > 1) {
            std::cout << "array_alloc_i32 ... rank " << rank << std::endl;
            void** res = new void*[shape[0]];
            int trailing_shape[rank - 1];
            for (int i = 0; i < rank - 1; i++) {
                trailing_shape[i] = shape[i + 1];
            }
            for (int i = 0; i < shape[0]; i++) {
                res[i] = array_alloc_i32(trailing_shape, rank - 1);
            }
            return res;
        } else {
            std::cout << "Invalid system state: memory cannot be allocated for array with rank 0" << std::endl;
            return NULL;
        }
    }

    int index_i32(void* ptr, int idx) {
        std::cout << "index_i32 ..." << std::endl;
        std::cout << "ptr " << ptr << std::endl;
        std::cout << "idx " << idx << std::endl;
        auto res = ((int*) ptr)[idx];
        std::cout << "index_void_ptr ... return int " << res << std::endl;
        return res;
    }

    double index_f64(void* ptr, int idx) {
        return ((double*) ptr)[idx];
    }

    void* index_void_ptr(void** ptr, int idx) {
        std::cout << "index_void_ptr ..." << std::endl;
        std::cout << "ptr " << ptr << std::endl;
        std::cout << "idx " << idx << std::endl;
        auto res = ptr[idx];
        std::cout << "index_void_ptr ... return ptr " << res << std::endl;
        return res;
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
        // std::cout << "c__XlaOp_operator_add x handle " << reinterpret_cast<XlaOp&>(x) << std::endl;
        // std::cout << "c__XlaOp_operator_add y handle " << reinterpret_cast<XlaOp&>(y) << std::endl;
        *res = Add(reinterpret_cast<XlaOp&>(x), reinterpret_cast<XlaOp&>(y));
        // std::cout << "c__XlaOp_operator_add res handle " << *res << std::endl;
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
     * Literal
     *
     *
     */

    struct c__Literal;

    int c__Literal_Get_int(c__Literal& lit, int* indices) {
        Literal& lit_ = reinterpret_cast<Literal&>(lit);
        int64 rank = lit_.shape().rank();
        tensorflow::int64 indices64[rank];
        for (int i = 0; i < rank; i++) {
            indices64[i] = indices[i];
        }
        auto multi_index = absl::Span<const tensorflow::int64>(indices64, rank);
        return lit_.Get<int>(multi_index);
    }

    static void write_literal_to_array_int_impl(
        xla::Literal& lit,
        void* arr,
        int* shape,
        int rank,
        int num_remaining_dims,
        int* current_indices
    ) {
        std::cout << "write_literal_to_array_int_impl ..." << std::endl;

        for (int i = 0; i < shape[rank - num_remaining_dims]; i++) {
            std::cout << "write_literal_to_array_int_impl ... i " << i << std::endl;
            int new_current_indices[rank];
            for (int j = 0; j < rank - num_remaining_dims; j++) {
                new_current_indices[j] = current_indices[j];
            }
            new_current_indices[rank - num_remaining_dims] = i;

            if (num_remaining_dims == 1) {
                std::cout << "write_literal_to_array_int_impl ... get array element" << std::endl;
                int res = c__Literal_Get_int(reinterpret_cast<c__Literal&>(lit), new_current_indices);
                std::cout << "write_literal_to_array_int_impl ... write array element" << std::endl;
                ((int*) arr)[i] = res;
            } else {
                std::cout << "write_literal_to_array_int_impl ... recurse" << std::endl;
                write_literal_to_array_int_impl(
                    lit,
                    ((void**) arr)[i],
                    shape,
                    rank,
                    num_remaining_dims - 1,
                    new_current_indices
                );
            }
        }
    }

    void* to_array_int(c__Literal& lit) {
        std::cout << "to_array_int ..." << std::endl;
        xla::Literal& lit_ = reinterpret_cast<xla::Literal&>(lit);
        std::cout << "lit " << &lit_ << std::endl;

        std::cout << "to_array_int ... get lit_shape" << std::endl;
        Shape lit_shape = lit_.shape();
        std::cout << "to_array_int ... get rank" << std::endl;
        int64 rank = lit_shape.rank();
        std::cout << "to_array_int rank " << rank << std::endl;

        std::cout << "to_array_int ... get shape" << std::endl;
        int shape[rank];
        const int64* shape64 = lit_.shape().dimensions().data();
        for (int i = 0; i < rank; i++) {
            shape[i] = shape64[i];
        }

        std::cout << "to_array_int ... allocate array" << std::endl;
        void* arr = array_alloc_i32(shape, rank);
        int current_indices[rank] = {0};
        std::cout << "to_array_int ... write to array" << std::endl;
        write_literal_to_array_int_impl(lit_, arr, shape, rank, rank, current_indices);
        std::cout << "to_array_int ... return array" << std::endl;
        return arr;
    }

    int to_int(c__Literal& lit) {
        return *(int*) reinterpret_cast<Literal&>(lit).untyped_data();
    }

    double to_double(c__Literal& lit) {
        return *(double*) reinterpret_cast<Literal&>(lit).untyped_data();
    }

    static void write_array_to_literal_impl(
        xla::Literal& lit,
        void* data,
        int* shape,
        int rank,
        int num_remaining_dims,
        tensorflow::int64* current_indices
    ) {
        std::cout << "set_literal ..." << std::endl;
        std::cout << "rank " << rank << std::endl;
        std::cout << "shape ";
        for (int k = 0; k < rank; k++) {
            std::cout << shape[k] << " ";
        }
        std::cout << std::endl;
        std::cout << "num_remaining_dims " << num_remaining_dims << std::endl;
        std::cout << "current_indices ";
        for (int k = 0; k < rank; k++) {
            std::cout << current_indices[k] << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < shape[rank - num_remaining_dims]; i++) {
            tensorflow::int64 new_current_indices[rank];
            for (int j = 0; j < rank - num_remaining_dims; j++) {
                new_current_indices[i] = current_indices[i];
            }
            new_current_indices[rank - num_remaining_dims] = i;
            
            if (num_remaining_dims == 1) {
                std::cout << "set_literal ... constructing multi_index" << std::endl;
                absl::Span<const tensorflow::int64> multi_index = absl::Span<const tensorflow::int64>(new_current_indices, rank);

                std::cout << "set_literal ... new_current_indices" << std::endl;
                for (int k = 0; k < rank; k++) {
                    std::cout << new_current_indices[k] << " ";
                }
                std::cout << std::endl;

                std::cout << "set_literal ... indexing leaf element" << std::endl;
                std::cout << "set_literal ... setting literal value" << std::endl;
                lit.Set(multi_index, ((int*) data)[i]);
                std::cout << "set_literal ... returning" << std::endl;
            } else {
                write_array_to_literal_impl(
                    lit,
                    ((void**) data)[i],
                    shape,
                    rank,
                    num_remaining_dims - 1,
                    new_current_indices
                );
            }
        }
    }

    c__Literal* array_to_literal(void* data, int* shape, int rank) {
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
        tensorflow::int64 current_indices[rank] = {0};
        write_array_to_literal_impl(*lit, data, shape, rank, rank, current_indices);

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

    // c__XlaOp* constant(c__XlaBuilder& builder, void* data, int* shape, int rank) {
    c__XlaOp* constant(c__XlaBuilder& builder, c__Literal& data) {
        std::cout << "constant ..." << std::endl;

        XlaBuilder& builder_ = reinterpret_cast<XlaBuilder&>(builder);
        xla::Literal& data_ = reinterpret_cast<xla::Literal&>(data);

        // std::cout << "constant rank: " << rank << std::endl;

            // int64 shape64[rank];
            // for (int i = 0; i < rank; i++) {
            //     shape64[i] = shape[i];
            // }
        // int size = 1;
        // for (int i = 0; i < rank; i++) {
        //     size *= shape[i];
        // }

        // std::cout << ((int*) data)[0] << std::endl;
        // std::cout << ((int*) data)[1] << std::endl;
        // std::cout << ((int*) data)[2] << std::endl;

        // auto array = Array<int32>(shape_);

        // bool dynamic_dimensions[rank];
        // for (int i = 0; i < rank; i++) {
        //     dynamic_dimensions[i] = false;
        // }

        // for (int i = 0; i < rank; i++) {
        //     std::cout << shape[i] << " ";
        // }
        // std::cout << std::endl;;

            // const std::vector<bool> dynamic_dimensions(rank, false);

            // auto xla_shape = ShapeUtil::MakeShape(
            //     PrimitiveType::S32,
            //     absl::Span<const int64>(shape64, rank),
            //     dynamic_dimensions
            // );

        // std::cout << "constant shape: " << xla_shape.ToString(true) << std::endl;
        // std::cout << "constant shape has_layout: " << xla_shape.has_layout() << std::endl;
        // std::cout << "constant shape DebugString:" << std::endl << xla_shape.DebugString() << std::endl;

            // std::cout << "constant ... constructing literal" << std::endl;

            // xla::Literal lit = xla::Literal(xla_shape, true);

            // std::cout << "constant ... populating literal" << std::endl;

            // write_array_to_literal(lit, data, shape, rank);

        // switch (rank) {
        //     case 1:
        //         auto span = absl::Span<const int>((int*) data, size);
        //         lit = LiteralUtil::CreateR1<int>(span);
        //         break;
        //     case 2:
        //         lit = LiteralUtil::CreateR2<int>((int**) data);
        //         break;
        //     case 3:
        //         lit = LiteralUtil::CreateR3<int>((int***) data);
        //         break;
        //     case 4:
        //         lit = LiteralUtil::CreateR4<int>((int****) data);
        //         break;
        //     default:
        //         std::cout << "rank greater than 4 not supported" << std::endl;
        // }

        std::cout << "constant ... constructing XlaOp" << std::endl;

        // auto lit_data = (int*)lit.untyped_data();
        // std::cout << lit_data[0] << " " << lit_data[1] << " " << lit_data[2] << " " << std::endl;
        XlaOp* op = new XlaOp();
        *op = ConstantLiteral(&builder_, data_);

        // std::cout << "constant 5" << std::endl;
        // std::cout << "constant op: " << *op << std::endl;

        std::cout << "constant ... return XlaOp" << std::endl;

        return reinterpret_cast<c__XlaOp*>(op);
    }

    c__Literal* eval(c__XlaOp& op) {
        XlaOp& op_ = reinterpret_cast<XlaOp&>(op);

        std::cout << "eval ... build op" << std::endl;
        // std::cout << "eval op handle " << op_ << std::endl;
        XlaComputation computation = op_.builder()->Build().ConsumeValueOrDie();
        // std::cout << "eval 2" << std::endl;
        // std::cout << "eval ... build literal" << std::endl;

        // todo this next section copied from above - needs refactoring
        // int64 shape64[rank];
        // for (int i = 0; i < rank; i++) {
        //     std::cout << "i = " << i << std::endl;
        //     shape64[i] = shape[i];
        // }

        // std::cout << "eval ... build literal ... initialise dynamic_dimensions" << std::endl;

        // const std::vector<bool> dynamic_dimensions(rank, false);

        // std::cout << "eval ... build literal ... make shape" << std::endl;

        // auto xla_shape = ShapeUtil::MakeShape(
        //     PrimitiveType::S32,
        //     absl::Span<const int64>(shape64, rank),
        //     dynamic_dimensions
        // );

        // std::cout << "eval ... build literal ... instantiate literal" << std::endl;

        

        std::cout << "eval ... run client" << std::endl;

        ExecutionProfile profile;
        Literal lit = ClientLibrary::LocalClientOrDie()
            ->ExecuteAndTransfer(computation, {}, nullptr, &profile)
            .ConsumeValueOrDie();

        xla::Literal* res = new xla::Literal(lit.shape(), true);
        *res = lit.Clone();

        std::cout << "eval ... copy data" << std::endl;
        // todo is this a shallow copy when we need a deep copy?
        // int64 size = lit.size_bytes();
        // void* res = malloc(size);;
        // memcpy(res, lit.untyped_data(), size);

        std::cout << *res << std::endl;

        std::cout << "eval ... return data " << res << std::endl;
        // TODO create new array with lit contents (with lit.Get?) and return
        return reinterpret_cast<c__Literal*>(res);
    }
}
