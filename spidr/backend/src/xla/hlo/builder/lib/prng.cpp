/*
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
