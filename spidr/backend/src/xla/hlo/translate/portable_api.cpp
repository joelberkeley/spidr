/*
Copyright 2024 Joel Berkeley

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
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/portable_api.h"

#include "../ir/hlo_module.h"
#include "../../../ffi.h"

extern "C" {
    string* ConvertHloToStablehlo(HloModule& hlo_module) {
        printf("ConvertHloToStablehlo ...\n");
        auto& hlo_module_ = reinterpret_cast<xla::HloModule&>(hlo_module);
        printf("0\n");
        // the implementation of this function shows how to get the actual MLIR module, which is
        // crucial for enzyme!
        auto res = xla::ConvertHloToStablehlo(hlo_module_, true);
        printf("1\n");
        return reinterpret_cast<string*>(new std::string(res.value()));
    }
}
