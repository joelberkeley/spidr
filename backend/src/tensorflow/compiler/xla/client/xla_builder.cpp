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
// #include <stdio.h>
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
        // todo is this a memory leak?
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

    struct c__XlaBuilder* c__XlaBuilder_new() {
        return reinterpret_cast<c__XlaBuilder*>(new XlaBuilder("XlaBuilder"));
    }

    void c__XlaBuilder_del(c__XlaBuilder* s) {
        delete reinterpret_cast<XlaBuilder*>(s);
    }

    const char* c__XlaBuilder_OpToString(c__XlaBuilder* s, c__XlaOp* op) {
        auto s_ = reinterpret_cast<XlaBuilder*>(s);
        auto op_ = reinterpret_cast<XlaOp*>(op);
        auto op_str = s_->OpToString(*op_);
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

    c__XlaOp* c__ConstantR0(c__XlaBuilder* builder, int32 value) {
        auto res = new XlaOp();
        *res = ConstantR0(reinterpret_cast<XlaBuilder*>(builder), value);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    c__XlaOp* c__Eq(c__XlaOp* lhs, c__XlaOp* rhs) {
        auto lhs_ = reinterpret_cast<XlaOp*>(lhs);
        auto rhs_ = reinterpret_cast<XlaOp*>(rhs);
        auto res = new XlaOp();
        *res = Eq(*lhs_, *rhs_);
        return reinterpret_cast<c__XlaOp*>(res);
    }

    /*
     *
     *
     * Non-standard temporary utility functions
     * 
     * 
     */

    void eval (c__XlaBuilder* builder, c__XlaOp* op) {
        auto op_ = reinterpret_cast<XlaOp*>(op);
        auto builder_ = reinterpret_cast<XlaBuilder*>(builder);
        ValueInferenceMode value_inf_mode;
        auto value_inf = new ValueInference(builder_);
        auto opt_literal_or = value_inf->AnalyzeConstant(*op_, value_inf_mode);
        
        if (opt_literal_or.ok()) {
            // auto ints = absl::Span<const int64>(new int64[1], 1);
            std::cout << opt_literal_or->GetValue()->ToString();
        } else {}
    }
}
