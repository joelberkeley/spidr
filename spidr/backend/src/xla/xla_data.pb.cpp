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
#include "xla/xla_data.pb.h"

#include "xla_data.pb.h"

extern "C" {
    DotDimensionNumbers* DotDimensionNumbers_new() {
        return reinterpret_cast<DotDimensionNumbers*>(new xla::DotDimensionNumbers());
    }

    void DotDimensionNumbers_delete(DotDimensionNumbers* dimension_numbers) {
        delete reinterpret_cast<xla::DotDimensionNumbers*>(dimension_numbers);
    }

    void DotDimensionNumbers_add_lhs_contracting_dimensions(
        DotDimensionNumbers& dimension_numbers, int dim
    ) {
        auto& dimension_numbers_ = reinterpret_cast<xla::DotDimensionNumbers&>(dimension_numbers);
        dimension_numbers_.add_lhs_contracting_dimensions(dim);
    }

    void DotDimensionNumbers_add_rhs_contracting_dimensions(
        DotDimensionNumbers& dimension_numbers, int dim
    ) {
        auto& dimension_numbers_ = reinterpret_cast<xla::DotDimensionNumbers&>(dimension_numbers);
        dimension_numbers_.add_rhs_contracting_dimensions(dim);
    }

    void DotDimensionNumbers_add_lhs_batch_dimensions(
        DotDimensionNumbers& dimension_numbers, int dim
    ) {
        auto& dimension_numbers_ = reinterpret_cast<xla::DotDimensionNumbers&>(dimension_numbers);
        dimension_numbers_.add_lhs_batch_dimensions(dim);
    }

    void DotDimensionNumbers_add_rhs_batch_dimensions(
        DotDimensionNumbers& dimension_numbers, int dim
    ) {
        auto& dimension_numbers_ = reinterpret_cast<xla::DotDimensionNumbers&>(dimension_numbers);
        dimension_numbers_.add_rhs_batch_dimensions(dim);
    }
}
