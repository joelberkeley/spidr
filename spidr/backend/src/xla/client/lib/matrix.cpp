/*
Copyright (C) 2022  Joel Berkeley

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
#include "xla/client/lib/matrix.h"

#include "../xla_builder.h"

extern "C" {
    XlaOp* IdentityMatrix(XlaBuilder* builder, int type, int m, int n) {
        auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        xla::XlaOp res = xla::IdentityMatrix(
            builder_, (xla::PrimitiveType) type, (int64_t) m, (int64_t) n
        );
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* GetMatrixDiagonal(XlaOp& x) {
        auto& x_ = reinterpret_cast<xla::XlaOp&>(x);
        xla::XlaOp res = xla::GetMatrixDiagonal(x_);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* Triangle(XlaOp& x, int lower) {
        auto& x_ = reinterpret_cast<xla::XlaOp&>(x);
        xla::XlaOp res = xla::Triangle(x_, (bool) lower);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}
