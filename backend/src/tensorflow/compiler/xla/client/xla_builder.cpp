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
/* This file contains the pure C API to XLA. */
#include <tensorflow/compiler/xla/client/xla_builder.h>
#include <tensorflow/compiler/xla/client/value_inference.h>

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

    c__XlaBuilder* c__XlaBuilder_new() {
        auto builder = new XlaBuilder("XlaBuilder");
        return reinterpret_cast<c__XlaBuilder*>(builder);
    }

    void c__XlaBuilder_del(c__XlaBuilder* s) {
        delete reinterpret_cast<XlaBuilder*>(s);
    }

    const char* c__XlaBuilder_OpToString(c__XlaBuilder& s, c__XlaOp& op) {
        auto& s_ = reinterpret_cast<XlaBuilder&>(s);
        auto& op_ = reinterpret_cast<XlaOp&>(op);
        auto op_str = s_.OpToString(op_);
        char *res = NULL;
        res = (char *) malloc(op_str.length() + 1);
        strncpy(res, op_str.c_str(), op_str.length());
        return res;
    }

    /*
     *
     *
     * Free functions
     * 
     * 
     */

    c__XlaOp* c__ConstantR0(c__XlaBuilder& builder, int32 value) {
        auto& builder_ = reinterpret_cast<XlaBuilder&>(builder);
        auto res = new XlaOp();
        *res = ConstantR0(&builder_, value);
        return reinterpret_cast<c__XlaOp*>(res);
    }

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

    void* eval (c__XlaBuilder& builder, c__XlaOp& op) {
        auto& op_ = reinterpret_cast<XlaOp&>(op);
        auto& builder_ = reinterpret_cast<XlaBuilder&>(builder);

        ValueInferenceMode value_inf_mode;
        Literal lit = (new ValueInference(&builder_))
            ->AnalyzeConstant(op_, value_inf_mode)
            ->GetValue()
            ->Clone();  // todo we get memory errors if we don't do this. Not sure why.

        auto size = lit.size_bytes();
        auto data = lit.untyped_data();

        void* res = malloc(size);;
        memcpy(res, data, size);

        return res;
    }

    int32 eval_int32(c__XlaBuilder& builder, c__XlaOp& op) {
        return *(int32*) eval(builder, op);
    }
}
