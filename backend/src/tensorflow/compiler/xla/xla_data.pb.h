/*
Copyright 2023 Joel Berkeley

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
#include "tensorflow/compiler/xla/xla_data.pb.h"

extern "C" {
    struct DotDimensionNumbers;

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
