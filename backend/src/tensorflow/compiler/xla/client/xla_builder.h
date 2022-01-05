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
#include "src/tensorflow/compiler/xla/literal.h"

extern "C" {
    /*
     *
     *
     * FFI
     *
     *
     */

    void free_int_array(int* arr);
    int* alloc_int_array(int len);
    void set_array_int(int* arr, int idx, int value);

    /*
     *
     *
     * XlaOp
     *
     *
     */

    struct XlaOp;
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

    const char* XlaBuilder_OpToString(XlaBuilder& s, XlaOp& op);

    XlaOp* Broadcast(XlaOp& s, int* broadcast_sizes, int len);

    XlaOp* BroadcastInDim(
        XlaOp& s, int* out_dim_size, int ods_len, int* broadcast_dimensions, int bcd_len
    );

    XlaOp* Eq(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Ne(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Ge(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Gt(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Lt(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Le(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Add(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Sub(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Mul(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Div(XlaOp& lhs, XlaOp& rhs);
    XlaOp* Rem(XlaOp& lhs, XlaOp& rhs);

    XlaOp* Neg(XlaOp& operand);
    XlaOp* Abs(XlaOp& operand);

    XlaOp* ConstantLiteral(XlaBuilder& builder, Literal& data);

    Literal* eval(XlaOp& op);
}
