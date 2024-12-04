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
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"

#include "hlo_module.h"

#include "../../service/hlo.proto.h"
#include "../../service/hlo_module_config.h"

extern "C" {
    HloModule* HloModule_CreateFromProto(HloModuleProto& proto, HloModuleConfig& module_config) {
        auto& proto_ = reinterpret_cast<xla::HloModuleProto&>(proto);
        auto& module_config_ = reinterpret_cast<xla::HloModuleConfig&>(module_config);
        auto module = xla::HloModule::CreateFromProto(proto_, module_config_);
        return reinterpret_cast<HloModule*>(module.value().release());
    }

    void HloModule_delete(HloModule* s) {
        delete reinterpret_cast<xla::HloModule*>(s);
    }
}
