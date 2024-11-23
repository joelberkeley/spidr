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
#include "xla/hlo/builder/lib/prng.h"

#include "../../../shape.h"
#include "../xla_builder.h"

xla::BitGeneratorTy BitGenerator(int bit_generator) {
    xla::BitGeneratorTy bit_generator_;

    switch (bit_generator) {
            case 0:
                bit_generator_ = xla::ThreeFryBitGenerator;
                break;
            case 1:
                bit_generator_ = xla::PhiloxBitGenerator;
                break;
            default:
                break;
        };

    return bit_generator_;
}

extern "C" {
    struct RngOutput {
        XlaOp* value;
        XlaOp* state;
    };

    void delete_RngOutput(RngOutput* rngOutput) {
        free(rngOutput);
    }

    RngOutput* UniformFloatingPointDistribution(
        XlaOp& key,
        XlaOp& initial_state,
        int bit_generator,
        XlaOp& minval,
        XlaOp& maxval,
        Shape& shape
    ) {
        auto& key_ = reinterpret_cast<xla::XlaOp&>(key);
        auto& initial_state_ = reinterpret_cast<xla::XlaOp&>(initial_state);
        xla::BitGeneratorTy bit_generator_ = BitGenerator(bit_generator);
        auto& minval_ = reinterpret_cast<xla::XlaOp&>(minval);
        auto& maxval_ = reinterpret_cast<xla::XlaOp&>(maxval);
        auto& shape_ = reinterpret_cast<xla::Shape&>(shape);

        xla::RngOutput res = xla::UniformFloatingPointDistribution(
            key_, initial_state_, bit_generator_, minval_, maxval_, shape_
        );

        return new RngOutput {
            value: reinterpret_cast<XlaOp*>(new xla::XlaOp(res.value)),
            state: reinterpret_cast<XlaOp*>(new xla::XlaOp(res.state))
        };
    }

    RngOutput* NormalFloatingPointDistribution(
        XlaOp& key,
        XlaOp& initial_state,
        int bit_generator,
        Shape& shape
    ) {
        auto& key_ = reinterpret_cast<xla::XlaOp&>(key);
        auto& initial_state_ = reinterpret_cast<xla::XlaOp&>(initial_state);
        xla::BitGeneratorTy bit_generator_ = BitGenerator(bit_generator);
        auto& shape_ = reinterpret_cast<xla::Shape&>(shape);

        xla::RngOutput res = xla::NormalFloatingPointDistribution(
            key_, initial_state_, bit_generator_, shape_
        );

        return new RngOutput {
            value: reinterpret_cast<XlaOp*>(new xla::XlaOp(res.value)),
            state: reinterpret_cast<XlaOp*>(new xla::XlaOp(res.state))
        };
    }
}
