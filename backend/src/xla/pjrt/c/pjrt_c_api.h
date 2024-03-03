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

void pjrt_error_destroy(PJRT_Api* api, PJRT_Error_Destroy_Args* args);
}

void pjrt_error_message(PJRT_Api* api, PJRT_Error_Message_Args* args);

// how is a C enum represented through FFI? An integer of some sort?
PJRT_Error* pjrt_error_getcode(PJRT_Api* api, PJRT_Error_GetCode_Args* args);

PJRT_Error* pjrt_client_create(PJRT_Api* api, PJRT_Client_Create_Args* args);

PJRT_Program* PJRT_Program_new(char* code);

PJRT_Client_Compile_Args* PJRT_Client_Compile_Args_new(
  PJRT_Client* client, CompileOptions* options, PJRT_Program* program
);

PJRT_LoadedExecutable* PJRT_Client_Compile_Args_executable(PJRT_Client_Compile_Args* args);

PJRT_Error* pjrt_client_compile(PJRT_Api* api, PJRT_Client_Compile_Args* args);

PJRT_ExecuteOptions* PJRT_ExecuteOptions_new();

PJRT_LoadedExecutable_Execute_Args* PJRT_LoadedExecutable_Execute_Args_new(
  PJRT_LoadedExecutable* executable, PJRT_ExecuteOptions* options
);

PJRT_Buffer*** PJRT_LoadedExecutable_Execute_Args_output_lists(
  PJRT_LoadedExecutable_Execute_Args* args
);

// hacky shortcut
Literal* LiteralSingleton(PJRT_Buffer*** output_lists);

PJRT_Error* pjrt_loadedexecutable_execute(
  PJRT_Api* api, PJRT_LoadedExecutable_Execute_Args* args
);

// does this test show us how to copy a PJRT_Buffer to xla::Literal?
// TEST_F(PjrtCApiBufferTest, ToHostBufferNoHostLayout) {

#ifdef __cplusplus
}
#endif
