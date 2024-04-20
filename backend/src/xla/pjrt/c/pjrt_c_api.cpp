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
#include <iostream>

// remove these when we've pulled out build config
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/client/executable_build_options.h"
#include "xla/service/computation_placer.h"

#include "../pjrt_executable.h"
#include "pjrt_c_api.h"

extern "C" {
  // ---------------------------------- Errors -----------------------------------

  PJRT_Error_Destroy_Args* PJRT_Error_Destroy_Args_new(PJRT_Error* error) {
    // std::cout << "PJRT_Error_Destroy_Args_new ..." << std::endl;
    return new PJRT_Error_Destroy_Args{
      .struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .error = error,
    };
  }

  void pjrt_error_destroy(PJRT_Api* api, PJRT_Error_Destroy_Args* args) {
    // std::cout << "pjrt_error_destroy ..." << std::endl;
    api->PJRT_Error_Destroy(args);
  }

  PJRT_Error_Message_Args* PJRT_Error_Message_Args_new(PJRT_Error* error) {
    // std::cout << "PJRT_Error_Message_Args_new ..." << std::endl;
    return new PJRT_Error_Message_Args{
      .struct_size = PJRT_Error_Message_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .error = error,
    };
  }

  char* PJRT_Error_Message_Args_message(PJRT_Error_Message_Args* args) {
    // std::cout << "PJRT_Error_Message_Args_message ..." << std::endl;
    auto len = args->message_size;
    auto res = (char*) malloc(len + 1);
    strncpy(res, args->message, len);
    res[len] = '\0';
    return res;
  }

  void pjrt_error_message(PJRT_Api* api, PJRT_Error_Message_Args* args) {
    // std::cout << "pjrt_error_message ..." << std::endl;
    api->PJRT_Error_Message(args);
  }

  PJRT_Error_GetCode_Args* PJRT_Error_GetCode_Args_new(PJRT_Error* error) {
    // std::cout << "PJRT_Error_GetCode_Args_new ..." << std::endl;
    return new PJRT_Error_GetCode_Args{
      .struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .error = error,
    };
  }

  int PJRT_Error_GetCode_Args_code(PJRT_Error_GetCode_Args* args) {
    // std::cout << "PJRT_Error_GetCode_Args_code ..." << std::endl;
    return (int) args->code;
  }

  PJRT_Error* pjrt_error_getcode(PJRT_Api* api, PJRT_Error_GetCode_Args* args) {
    // std::cout << "pjrt_error_getcode ..." << std::endl;
    return api->PJRT_Error_GetCode(args);
  }

  // ---------------------------------- Events -----------------------------------

  PJRT_Event_Destroy_Args* PJRT_Event_Destroy_Args_new(PJRT_Event* event) {
    // std::cout << "PJRT_Event_Destroy_Args_new ..." << std::endl;
    return new PJRT_Event_Destroy_Args{
      .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .event = event,
    };
  }

  void pjrt_event_destroy(PJRT_Api* api, PJRT_Event_Destroy_Args* args) {
    // std::cout << "pjrt_event_destroy ..." << std::endl;
    api->PJRT_Event_Destroy(args);
  }

  PJRT_Event_Await_Args* PJRT_Event_Await_Args_new(PJRT_Event* event) {
    // std::cout << "PJRT_Event_Await_Args_new ..." << std::endl;
    return new PJRT_Event_Await_Args{
      .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .event = event,
    };
  }

  void pjrt_event_await(PJRT_Api* api, PJRT_Event_Await_Args* args) {
    // std::cout << "pjrt_event_await ..." << std::endl;
    api->PJRT_Event_Await(args);
  }

  // ---------------------------------- Client -----------------------------------

  PJRT_Client_Create_Args* PJRT_Client_Create_Args_new() {
    // std::cout << "PJRT_Client_Create_Args_new ..." << std::endl;
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
    };
  }

  PJRT_Client_Destroy_Args* PJRT_Client_Destroy_Args_new(PJRT_Client* client) {
    // std::cout << "PJRT_Client_Destroy_Args_new ..." << std::endl;
    return new PJRT_Client_Destroy_Args{
      .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client,
    };
  }

  PJRT_Error* pjrt_client_destroy(PJRT_Api* api, PJRT_Client_Destroy_Args* args) {
    // std::cout << "pjrt_client_destroy ..." << std::endl;
    return api->PJRT_Client_Destroy(args);
  }

  PJRT_Client* PJRT_Client_Create_Args_client(PJRT_Client_Create_Args* args) {
    // std::cout << "PJRT_Client_Create_Args_client ..." << std::endl;
    return args->client;
  }

  PJRT_Error* pjrt_client_create(PJRT_Api* api, PJRT_Client_Create_Args* args) {
    // std::cout << "pjrt_client_create ..." << std::endl;
    return api->PJRT_Client_Create(args);
  }

  PJRT_Program* PJRT_Program_new(char* code, size_t code_size) {
    // std::cout << "PJRT_Program_new ..." << std::endl;
    auto format = pjrt::kHloFormat;
    return new PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .extension_start = nullptr,
      .code = code,
      .code_size = code_size,
      .format = format.data(),
      .format_size = format.size(),
    };
  }

  PJRT_Client_Compile_Args* PJRT_Client_Compile_Args_new(
    PJRT_Client* client, PJRT_Program* program, char* compile_options, size_t compile_options_size
  ) {
    // std::cout << "PJRT_Client_Compile_Args_new ..." << std::endl;
    return new PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client,
      .program = program,
      .compile_options = compile_options,
      .compile_options_size = compile_options_size,
    };
  }

  PJRT_LoadedExecutable* PJRT_Client_Compile_Args_executable(PJRT_Client_Compile_Args* args) {
    // std::cout << "PJRT_Client_Compile_Args_executable ..." << std::endl;
    return args->executable;
  }

  PJRT_Error* pjrt_client_compile(PJRT_Api* api, PJRT_Client_Compile_Args* args) {
    // std::cout << "pjrt_client_compile ..." << std::endl;
    return api->PJRT_Client_Compile(args);
  }

  PJRT_LoadedExecutable_Destroy_Args* PJRT_LoadedExecutable_Destroy_Args_new(
    PJRT_LoadedExecutable* executable
  ) {
    // std::cout << "PJRT_LoadedExecutable_Destroy_Args_new ..." << std::endl;
    return new PJRT_LoadedExecutable_Destroy_Args{
      .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .executable = executable,
    };
  }

  PJRT_Error* pjrt_loadedexecutable_destroy(
    PJRT_Api* api, PJRT_LoadedExecutable_Destroy_Args* args
  ) {
    // std::cout << "pjrt_loadedexecutable_destroy ..." << std::endl;
    return api->PJRT_LoadedExecutable_Destroy(args);
  }

  PJRT_ExecuteOptions* PJRT_ExecuteOptions_new() {
    // std::cout << "PJRT_ExecuteOptions_new ..." << std::endl;
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
    PJRT_LoadedExecutable* executable,
    PJRT_ExecuteOptions* options,
    PJRT_Buffer*** output_lists
  ) {
    // std::cout << "PJRT_LoadedExecutable_Execute_Args_new ..." << std::endl;
    return new PJRT_LoadedExecutable_Execute_Args{
      .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .executable = executable,
      .options = options,
      .argument_lists = nullptr,
      .num_devices = 1,
      .num_args = 0,
      .output_lists = output_lists,
      .device_complete_events = nullptr,
      .execute_device = nullptr,
    };
  }

  PJRT_Buffer** const* PJRT_LoadedExecutable_Execute_Args_output_lists(
    PJRT_LoadedExecutable_Execute_Args* args
  ) {
    // std::cout << "PJRT_LoadedExecutable_Execute_Args_output_lists ..." << std::endl;
    return args->output_lists;
  }

  PJRT_Error* pjrt_loadedexecutable_execute(
    PJRT_Api* api, PJRT_LoadedExecutable_Execute_Args* args
  ) {
    // std::cout << "pjrt_loadedexecutable_execute ..." << std::endl;
    return api->PJRT_LoadedExecutable_Execute(args);
  }

  PJRT_Buffer_Destroy_Args* PJRT_Buffer_Destroy_Args_new(PJRT_Buffer* buffer) {
    // std::cout << "PJRT_Buffer_Destroy_Args_new ..." << std::endl;
    return new PJRT_Buffer_Destroy_Args{
      .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .buffer = buffer,
    };
  }

  PJRT_Error* pjrt_buffer_destroy(PJRT_Api* api, PJRT_Buffer_Destroy_Args* args) {
    // std::cout << "pjrt_buffer_destroy ..." << std::endl;
    return api->PJRT_Buffer_Destroy(args);
  }

  PJRT_Buffer_ToHostBuffer_Args* PJRT_Buffer_ToHostBuffer_Args_new(
    PJRT_Buffer* src, void* dst, size_t dst_size
  ) {
    // std::cout << "PJRT_Buffer_ToHostBuffer_Args_new ..." << std::endl;
    return new PJRT_Buffer_ToHostBuffer_Args{
      .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .src = src,
      .host_layout = nullptr,
      .dst = dst,
      .dst_size = dst_size,
      .event = nullptr,
    };
  }

  PJRT_Event* PJRT_Buffer_ToHostBuffer_Args_event(PJRT_Buffer_ToHostBuffer_Args* args) {
    // std::cout << "PJRT_Buffer_ToHostBuffer_Args_event ..." << std::endl;
    return args->event;
  }

  PJRT_Error* pjrt_buffer_tohostbuffer(PJRT_Api* api, PJRT_Buffer_ToHostBuffer_Args* args) {
    // std::cout << "pjrt_buffer_tohostbuffer ..." << std::endl;
    return api->PJRT_Buffer_ToHostBuffer(args);
  }
}
