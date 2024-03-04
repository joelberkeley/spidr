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
#include <cstddef>

// remove these when we've pulled out build config
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/client/executable_build_options.h"
#include "xla/service/computation_placer.h"

#include "../../literal.h"
#include "../pjrt_executable.h"
#include "pjrt_c_api.h"

#ifdef __cplusplus  // is this C or C++?
extern "C" {
#endif

PJRT_Error_Destroy_Args* PJRT_Error_Destroy_Args_new(PJRT_Error* error) {
  return new PJRT_Error_Destroy_Args{
    .struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .error = error,
  };
}

void pjrt_error_destroy(PJRT_Api* api, PJRT_Error_Destroy_Args* args) {
  api->PJRT_Error_Destroy(args);
}

PJRT_Error_Message_Args* PJRT_Error_Message_Args_new(PJRT_Error* error) {
  return new PJRT_Error_Message_Args{
    .struct_size = PJRT_Error_Message_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .error = error,
  };
}

char* PJRT_Error_Message_Args_message(PJRT_Error_Message_Args* args) {
  auto len = args->message_size;
  auto res = (char*) malloc(len + 1);
  strncpy(res, args->message, len);
  res[len] = '\0';
  return res;
}

void pjrt_error_message(PJRT_Api* api, PJRT_Error_Message_Args* args) {
  api->PJRT_Error_Message(args);
}

PJRT_Error_GetCode_Args* PJRT_Error_GetCode_Args_new(PJRT_Error* error) {
  return new PJRT_Error_GetCode_Args{
    .struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .error = error,
  };
}

// how is a C enum represented through FFI? An integer of some sort?
PJRT_Error* pjrt_error_getcode(PJRT_Api* api, PJRT_Error_GetCode_Args* args) {
  return api->PJRT_Error_GetCode(args);
}

PJRT_Client_Create_Args* PJRT_Client_Create_Args_new() {
  return new PJRT_Client_Create_Args{
    .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .create_options = nullptr,
    .num_options = 0,
    .kv_get_callback = nullptr,
    .kv_get_user_arg = nullptr,
    .kv_put_callback = nullptr,
    .kv_put_user_arg = nullptr,
    .client = nullptr,
  }
}

PJRT_Client_Destroy_Args* PJRT_Client_Destroy_Args_new(PJRT_Client* client) {
  return new PJRT_Client_Destroy_Args{
    ...
  }
}

PJRT_Error* pjrt_client_destroy(PJRT_Api* api, PJRT_Client_Destroy_Args* args);

PJRT_Client* PJRT_Client_Create_Args_client(PJRT_Client_Create_Args* args) {
  return args->client;
}

PJRT_Error* pjrt_client_create(PJRT_Api* api, PJRT_Client_Create_Args* args) {
  return api->PJRT_Client_Create(args);
}

PJRT_Program* PJRT_Program_new(char* code) {
  // are we allocating memory here in std::string?
  // if so, how can we not, since we're using `free` to free the struct
  auto format = std::string(pjrt::kHloFormat);
  return new PJRT_Program{
    .struct_size = PJRT_Program_STRUCT_SIZE,
    .extension_start = nullptr,
    .code = code,
    .code_size = strlen(code),
    .format = format.data(),
    .format_size = format.length(),
  };
}

PJRT_Client_Compile_Args* PJRT_Client_Compile_Args_new(
  PJRT_Client* client, CompileOptions* options, PJRT_Program* program
) {
  // move ->ToProto()->SerializeAsString() out and accept char*
  auto options_ = reinterpret_cast<xla::CompileOptions*>(options);
  auto options_str = options_->ToProto()->SerializeAsString();

  return new PJRT_Client_Compile_Args{
    .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
    .extension_start = nullptr,
    .client = client,
    .program = program,
    // FIXME these two args aren't co-operating
//    .compile_options = options_str.c_str(),
//    .compile_options_size = options_str.size(),
  };
}

//PJRT_LoadedExecutable* PJRT_Client_Compile_Args_executable(PJRT_Client_Compile_Args* args) {
//  return args->executable;
//}

PJRT_Error* pjrt_client_compile(PJRT_Api* api, PJRT_Client_Compile_Args* args) {
  return api->PJRT_Client_Compile(args);
}

//PJRT_ExecuteOptions* PJRT_ExecuteOptions_new() {
//  return new PJRT_ExecuteOptions{
//    .struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
//    .extension_start = nullptr,
//    .send_callbacks = nullptr,
//    .recv_callbacks = nullptr,
//    .num_send_ops = 0,
//    .num_recv_ops = 0,
//    .launch_id = 0,
//    .non_donatable_input_indices = nullptr,
//    .num_non_donatable_input_indices = 0,
//  };
//}

//PJRT_LoadedExecutable_Execute_Args* PJRT_LoadedExecutable_Execute_Args_new(
//  PJRT_LoadedExecutable* executable, PJRT_ExecuteOptions* options
//) {
//  return new PJRT_LoadedExecutable_Execute_Args{
//    .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
//    .extension_start = nullptr,
//    .executable = executable,
//    .options = options,
//    .argument_lists = nullptr,
//    .num_devices = 1,
//    .num_args = 0,
//    .output_lists = nullptr,
//    .device_complete_events = nullptr,
//    .execute_device = nullptr,
//  };
//}

// hacky shortcut
PJRT_Buffer* PJRT_LoadedExecutable_Execute_Args_output_singleton(
  PJRT_LoadedExecutable_Execute_Args* args
) {
  return args->output_lists[0][0];
}

PJRT_Error* pjrt_loadedexecutable_execute(
  PJRT_Api* api, PJRT_LoadedExecutable_Execute_Args* args
) {
  return api->PJRT_LoadedExecutable_Execute(args);
}

//PJRT_Buffer_ToHostBuffer_Args* PJRT_Buffer_ToHostBuffer_Args_new(
//  PJRT_Buffer* src, void* dst
//) {
//  return new PJRT_Buffer_ToHostBuffer_Args{
//    .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
//    .extension_start = nullptr,
//    .src = src,
//    .host_layout = nullptr,
//    .dst = dst,
//    .dst_size = 0,  // overriden, not sure if I need to specify this
//    .event = nullptr,
//  };
//}

PJRT_Error* pjrt_buffer_tohostbuffer(PJRT_Api* api, PJRT_Buffer_ToHostBuffer_Args* args) {
  return api->PJRT_Buffer_ToHostBuffer(args);
}

// does this test show us how to copy a PJRT_Buffer to xla::Literal?
// TEST_F(PjrtCApiBufferTest, ToHostBufferNoHostLayout) {

#ifdef __cplusplus
}
#endif
