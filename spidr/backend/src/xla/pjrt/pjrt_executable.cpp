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
#include <string>

#include "xla/pjrt/pjrt_executable.h"

#include "../client/executable_build_options.h"
#include "../../ffi.h"

extern "C" {
  struct CompileOptions;

  CompileOptions* CompileOptions_new(ExecutableBuildOptions* executable_build_options) {
    auto executable_build_options_ = reinterpret_cast<xla::ExecutableBuildOptions*>(
      executable_build_options
    );
    auto options = new xla::CompileOptions{
      .argument_layouts = std::nullopt,
      .executable_build_options = *executable_build_options_,
      .env_option_overrides = {},
      .target_config = std::nullopt,
    };
    return reinterpret_cast<CompileOptions*>(options);
  }

  string* CompileOptions_SerializeAsString(CompileOptions* s) {
    auto s_ = reinterpret_cast<xla::CompileOptions*>(s);
    auto res = s_->ToProto()->SerializeAsString();
    return reinterpret_cast<string*>(new std::string(res));
  }
}
