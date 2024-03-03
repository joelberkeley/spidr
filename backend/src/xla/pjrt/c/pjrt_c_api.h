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
#include "xla/pjrt/c/pjrt_c_api.h"

// remove these when we've pulled out build config
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_executable.h"
#include "xla/client/executable_build_options.h"
#include "xla/service/computation_placer.h"

#ifdef __cplusplus  // is this C or C++?
extern "C" {
#endif

void pjrt_error_destroy(PJRT_Api* api, PJRT_Error_Destroy_Args* args) {
  api->PJRT_Error_Destroy(args);
}

void pjrt_error_message(PJRT_Api* api, PJRT_Error_Message_Args* args) {
  api->PJRT_Error_Message(args);
}

// how is a C enum represented through FFI? An integer of some sort?
PJRT_Error* pjrt_error_getcode(PJRT_Api* api, PJRT_Error_GetCode_Args* args) {
  api->PJRT_Error_GetCode(args);
}

PJRT_Error* pjrt_client_create(PJRT_Api* api, PJRT_Client_Create_Args* args) {
  return api->PJRT_Client_Create(args);
}

PJRT_Program* PJRT_Program_new(char* code) {
  auto format = pjrt::kHloFormat;
  auto program = new PJRT_Program{
    .struct_size = PJRT_Program_STRUCT_SIZE,
    .extension_start = nullptr,
    .code = code,
    .code_size = strlen(code),
    .format = format,
    .format_size = strlen(format),
  };
}

PJRT_Client_Compile_Args* PJRT_Client_Compile_Args_new(
  PJRT_Client* client, CompileOptions* options, PJRT_Program* program
) {
  // move ->ToProto()->SerializeAsString() out and accept char*
  auto options_str = options->ToProto()->SerializeAsString();

  return new PJRT_Client_Compile_Args{
    .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .client = client,
    .compile_options = options,
    .compile_options_size = strlen(options),
    .program = &program,
  };
}

PJRT_LoadedExecutable* PJRT_Client_Compile_Args_executable(PJRT_Client_Compile_Args* args) {
  return args->executable;
}

PJRT_Error* pjrt_client_compile(PJRT_Api* api, PJRT_Client_Compile_Args* args) {
  return api->PJRT_Client_Compile(args);
}

PJRT_ExecuteOptions* PJRT_ExecuteOptions_new() {
  return new PJRT_ExecuteOptions{
    .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
    .extension_start = nullptr,
    .send_callbacks = nullptr,
    .recv_callbacks = nullptr,
    .num_send_ops = 0,
    .num_recv_ops = 0,
    .launch_id = 0,
    .non_donatable_input_indices = nullptr,
    .num_non_donatable_input_indices = 0,
  };
}

PJRT_LoadedExecutable_Execute_Args* PJRT_LoadedExecutable_Execute_Args_new(
  PJRT_LoadedExecutable* executable, PJRT_ExecuteOptions* options
) {
  return PJRT_LoadedExecutable_Execute_Args{
    .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .executable = executable,
    .options = options,
    .argument_lists = {nullptr},
    .num_devices = 1,
    .num_args = 0,
    .output_lists = {{nullptr}},
    .device_complete_events = nullptr,
    .execute_device = nullptr,
  };
}

PJRT_Buffer*** PJRT_LoadedExecutable_Execute_Args_output_lists(
  PJRT_LoadedExecutable_Execute_Args* args
) {
  return args->output_lists;
}

// hacky shortcut
Literal* LiteralSingleton(PJRT_Buffer*** output_lists) {
  return output_lists[0][0]->ToLiteralSync();
}

PJRT_Error* pjrt_loadedexecutable_execute(
  PJRT_Api* api, PJRT_LoadedExecutable_Execute_Args* args
) {
  return api->PJRT_LoadedExecutable_Execute(args);
}

// does this test show us how to copy a PJRT_Buffer to xla::Literal?
// TEST_F(PjrtCApiBufferTest, ToHostBufferNoHostLayout) {

#ifdef __cplusplus
}
#endif
