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
#include "pjrt_executable.h"

extern "C" {
  CompileOptions* CompileOptions_new() {
    auto build_options = new stream_executor::xla::ExecutableBuildOptions;
    build_options.set_device_ordinal(0);
    auto device_assignment = new xla::DeviceAssignment(1, 1);
    device_assignment(0, 0) = 0;
    build_options.set_device_assignment(device_assignment);
    auto options_str = xla::CompileOptions{
      .executable_build_options = build_options
    }

    return new xla::CompileOptions{
      .argument_layouts = std::nullopt,
      .executable_build_options = build_options,
      .env_option_overrides = {},
      .target_config = std::nullopt,
    };
  }
}