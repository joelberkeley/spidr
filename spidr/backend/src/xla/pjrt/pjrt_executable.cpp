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
#include <iostream>

#include "xla/client/xla_computation.h"

#include "pjrt_executable.h"

//char* c_string_copy(std::string str) {
//    auto len = str.length();
//    auto res = (char *) malloc(len);
//    strncpy(res, str.c_str(), len);
//    return res;
//}

extern "C" {
  CompileOptions* CompileOptions_new() {
    // std::cout << "CompileOptions_new ..." << std::endl;
    auto build_options = new xla::ExecutableBuildOptions;
    build_options->set_device_ordinal(0);

    auto options = new xla::CompileOptions{
      .argument_layouts = std::nullopt,
      .executable_build_options = *build_options,
      .env_option_overrides = {},
      .target_config = std::nullopt,
    };
    return reinterpret_cast<CompileOptions*>(options);
  }

  string* CompileOptions_SerializeAsString(CompileOptions* s) {
    // std::cout << "CompileOptions_SerializeAsString ..." << std::endl;
//    // std::cout << "CompileOptions_SerializeAsString ..." << std::endl;
    auto s_ = reinterpret_cast<xla::CompileOptions*>(s);
    auto res = s_->ToProto()->SerializeAsString();
//    // std::cout << "... serialized result: " << std::endl;
//    // std::cout << res << std::endl;
    return reinterpret_cast<string*>(new std::string(res));
  }
}
