/*
Copyright 2022 Joel Berkeley

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
#include "../literal.h"
#include "xla_computation.h"

extern "C" {
    /*
     *
     *
     * XlaOp
     *
     *
     */

    struct XlaOp;
    int sizeof_XlaOp();
    void set_array_XlaOp(XlaOp* arr, int idx, XlaOp* op);
    void XlaOp_delete(XlaOp* s);

    /*
     *
     *
     * XlaBuilder
     *
     *
     */

    struct XlaBuilder;
    XlaBuilder* XlaBuilder_new(const char* computation_name);
    void XlaBuilder_delete(XlaBuilder* s);
    const char* XlaBuilder_name(XlaBuilder& s);
    XlaBuilder* CreateSubBuilder(XlaBuilder& s, const char* computation_name);
    XlaComputation* XlaBuilder_Build(XlaBuilder& s);
    const char* XlaBuilder_OpToString(XlaBuilder& s, XlaOp& op);

    /*
     *
     *
     * Free functions
     *
     *
     */

    XlaOp* Parameter(XlaBuilder& builder, int parameter_number, Shape& shape, const char* name);

    XlaOp* ConstantLiteral(XlaBuilder& builder, Literal& data);

    XlaOp* Broadcast(XlaOp& s, int* broadcast_sizes, int len);

    XlaOp* BroadcastInDim(
        XlaOp& s, int* out_dim_size, int ods_len, int* broadcast_dimensions, int bcd_len
    );

    XlaOp* Reshape(
        XlaOp& operand, int* dimensions, int dimensions_len, int* new_sizes, int new_sizes_len
    );

    XlaOp* Slice(
        XlaOp& operand,
        int* start_indices,
        int start_indices_len,
        int* limit_indices,
        int limit_indices_len,
        int* strides,
        int strides_len
    );

    XlaOp* ConcatInDim(XlaBuilder* builder, XlaOp* operands, int operands_len, int dimension);

    XlaOp* Eq(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Ne(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Ge(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Gt(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Lt(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Le(XlaOp& lhs, XlaOp& rhs);

    XlaOp* Dot(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Cholesky(XlaOp& a, int lower);

    XlaOp* Add(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Sub(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Mul(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Div(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Rem(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Max(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Min(XlaOp& lhs, XlaOp& rhs);
    XlaOp* And(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Or(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Not(XlaOp& operand);

    XlaOp* Reduce(
        XlaOp& operand,
        XlaOp& init_value,
        const XlaComputation& computation,
        int* dimensions_to_reduce,
        int dimensions_to_reduce_len
    );

    XlaOp* Abs(XlaOp& operand);
    XlaOp* Exp(XlaOp& operand);
    XlaOp* Floor(XlaOp& operand);
    XlaOp* Ceil(XlaOp& operand);
    XlaOp* Log(XlaOp& operand);
    XlaOp* Logistic(XlaOp& operand);
    XlaOp* Cos(XlaOp& operand);
    XlaOp* Sin(XlaOp& operand);
    XlaOp* Tanh(XlaOp& operand);
    XlaOp* Sqrt(XlaOp& operand);
    XlaOp* Neg(XlaOp& operand);

    XlaOp* Map(
        XlaBuilder* builder,
        XlaOp* operands,
        int operands_len,
        XlaComputation& computation,
        int* dimensions,
        int dimensions_len,
        XlaOp* static_operands,
        int static_operands_len
    );
}
