{--
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
--}
module Compiler.Xla.PJRT.C.PJRT_C_API

import Control.Monad.Either

import Language.Reflection
import Derive.Prelude

import Compiler.FFI
import Compiler.Xla.Literal

%language ElabReflection

-- keep the API as close to the PJRT api as possible except:
-- * don't expose _Args, so we don't need to handle null ptrs. I really doubt we'd
--   ever need to expose them
-- use pointers for two reasons:
-- we can use onCollectAny to GC our data!
-- we can hide GCAnyPtr/AnyPtr in a more type-safe Idris API

public export
data PjrtApi = MkPjrtApi AnyPtr

public export
data PjrtErrorCode =
    PJRT_Error_Code_CANCELLED
  | PJRT_Error_Code_UNKNOWN
  | PJRT_Error_Code_INVALID_ARGUMENT
  | PJRT_Error_Code_DEADLINE_EXCEEDED
  | PJRT_Error_Code_NOT_FOUND
  | PJRT_Error_Code_ALREADY_EXISTS
  | PJRT_Error_Code_PERMISSION_DENIED
  | PJRT_Error_Code_RESOURCE_EXHAUSTED
  | PJRT_Error_Code_FAILED_PRECONDITION
  | PJRT_Error_Code_ABORTED
  | PJRT_Error_Code_OUT_OF_RANGE
  | PJRT_Error_Code_UNIMPLEMENTED
  | PJRT_Error_Code_INTERNAL
  | PJRT_Error_Code_UNAVAILABLE
  | PJRT_Error_Code_DATA_LOSS
  | PJRT_Error_Code_UNAUTHENTICATED

%runElab derive "PjrtErrorCode" [Show]

public export
record PjrtError where
  constructor MkPjrtError
  message : String
  code : Maybe PjrtErrorCode

%runElab derive "PjrtError" [Show]

public export 0
ErrIO : Type -> Type -> Type
ErrIO e a = EitherT e IO a

%foreign (libxla "PJRT_Error_Destroy_Args_new")
prim__mkPjrtErrorDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_error_destroy")
prim__pjrtErrorDestroy : AnyPtr -> AnyPtr -> PrimIO ()

destroyPjrtError : HasIO io => AnyPtr -> AnyPtr -> io ()
destroyPjrtError api err = do
  destroyArgs <- primIO $ prim__mkPjrtErrorDestroyArgs err
  primIO $ prim__pjrtErrorDestroy api destroyArgs
  free destroyArgs

%foreign (libxla "PJRT_Error_Message_Args_new")
prim__mkPjrtErrorMessageArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Error_Message_Args_message")
prim__pjrtErrorMessageArgsMessage : AnyPtr -> PrimIO String

%foreign (libxla "pjrt_error_message")
prim__pjrtErrorMessage : AnyPtr -> AnyPtr -> PrimIO ()

pjrtErrorMessage : HasIO io => AnyPtr -> AnyPtr -> io String
pjrtErrorMessage api err = do
  args <- primIO $ prim__mkPjrtErrorMessageArgs err
  primIO $ prim__pjrtErrorMessage api args
  msg <- primIO $ prim__pjrtErrorMessageArgsMessage args
  free args
  pure msg

%foreign (libxla "PJRT_Error_GetCode_Args_new")
prim__mkPjrtErrorGetCodeArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Error_GetCode_Args_code")
prim__mkPjrtErrorGetCodeArgsCode : AnyPtr -> Int

%foreign (libxla "pjrt_error_getcode")
prim__pjrtErrorGetCode : AnyPtr -> AnyPtr -> PrimIO AnyPtr

pjrtErrorCodeFromCInt : Int -> PjrtErrorCode
pjrtErrorCodeFromCInt = \case
  1  => PJRT_Error_Code_CANCELLED
  2  => PJRT_Error_Code_UNKNOWN
  3  => PJRT_Error_Code_INVALID_ARGUMENT
  4  => PJRT_Error_Code_DEADLINE_EXCEEDED
  5  => PJRT_Error_Code_NOT_FOUND
  6  => PJRT_Error_Code_ALREADY_EXISTS
  7  => PJRT_Error_Code_PERMISSION_DENIED
  8  => PJRT_Error_Code_RESOURCE_EXHAUSTED
  9  => PJRT_Error_Code_FAILED_PRECONDITION
  10 => PJRT_Error_Code_ABORTED
  11 => PJRT_Error_Code_OUT_OF_RANGE
  12 => PJRT_Error_Code_UNIMPLEMENTED
  13 => PJRT_Error_Code_INTERNAL
  14 => PJRT_Error_Code_UNAVAILABLE
  15 => PJRT_Error_Code_DATA_LOSS
  16 => PJRT_Error_Code_UNAUTHENTICATED
  n  => assert_total $ idris_crash
    "Unexpected PJRT_Error_Code value received through FFI from XLA: \{show n}"

export
data PjrtClient = MkPjrtClient GCAnyPtr

%foreign (libxla "PJRT_Client_Create_Args_new")
prim__mkPjrtClientCreateArgs : PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Create_Args_client")
prim__pjrtClientCreateArgsClient : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_client_create")
prim__pjrtClientCreate : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Destroy_Args_new")
prim__mkPjrtClientDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Destroy")
prim__pjrtClientDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

-- warnDestroyFailure : PjrtApi -> IO GCAnyPtr -> IO ()
-- warnDestroyFailure api err =

-- add address of target pointer?
handleErrOnDestroy : HasIO io => AnyPtr -> AnyPtr -> String -> io ()
handleErrOnDestroy api err target = unless (isNullPtr err) $ do
  msg <- pjrtErrorMessage api err
  args <- primIO $ prim__mkPjrtErrorGetCodeArgs err
  getCodeErr <- primIO $ prim__pjrtErrorGetCode api args
  if (isNullPtr getCodeErr) then do
      let code = pjrtErrorCodeFromCInt $ prim__mkPjrtErrorGetCodeArgsCode args
      printLn "WARN: Failed to destroy \{target} with error code \{show code}; message: \{msg}"
    else do
      printLn "WARN: Failed to fetch error code"
      printLn "WARN: Failed to destroy \{target} with unknown error code; message: \{msg}"
      destroyPjrtError api getCodeErr
  free args
  destroyPjrtError api err

try : AnyPtr -> AnyPtr -> a -> ErrIO PjrtError a
try api err onOk = if (isNullPtr err) then right onOk else do
  msg <- pjrtErrorMessage api err
  args <- primIO $ prim__mkPjrtErrorGetCodeArgs err
  getCodeErr <- primIO $ prim__pjrtErrorGetCode api args
  code <- if (isNullPtr getCodeErr) then pure Nothing else do
    let code = prim__mkPjrtErrorGetCodeArgsCode args
    destroyPjrtError api getCodeErr
    pure $ Just code
  free args
  destroyPjrtError api err
  left $ MkPjrtError msg $ map pjrtErrorCodeFromCInt code

export
pjrtClientCreate : PjrtApi -> ErrIO PjrtError PjrtClient
pjrtClientCreate (MkPjrtApi api) = do
  putStrLn "pjrtClientCreate ..."
  args <- primIO prim__mkPjrtClientCreateArgs
  err <- primIO $ prim__pjrtClientCreate api args
  try api err =<< do
    let client = prim__pjrtClientCreateArgsClient args
    free args
    client <- onCollectAny client destroyClient
    pure $ MkPjrtClient client

  where

  destroyClient : AnyPtr -> IO ()
  destroyClient client = do
    args <- primIO $ prim__mkPjrtClientDestroyArgs client
    err <- primIO $ prim__pjrtClientDestroy api args
    free args
    handleErrOnDestroy api err "PJRT_Client"

export
data PjrtProgram = MkPjrtProgram GCAnyPtr

%foreign (libxla "PJRT_Program_new")
prim__mkPjrtProgram : GCPtr Char -> Int -> PrimIO AnyPtr

export
mkPjrtProgram : HasIO io => GCPtr Char -> Int -> io PjrtProgram
mkPjrtProgram code codeSize = do
  ptr <- primIO $ prim__mkPjrtProgram code codeSize
  ptr <- onCollectAny ptr free
  pure (MkPjrtProgram ptr)

%foreign (libxla "PJRT_Client_Compile_Args_new")
prim__mkPjrtClientCompileArgs : GCAnyPtr -> GCAnyPtr -> GCPtr Char -> Int -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Compile_Args_executable")
prim__pjrtClientCompileArgsExecutable : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_client_compile")
prim__pjrtClientCompile : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_loadedexecutable_destroy")
prim__pjrtLoadedExecutableDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_LoadedExecutable_Destroy_Args_new")
prim__mkPjrtLoadedExecutableDestroyArgs : AnyPtr -> PrimIO AnyPtr

export
data PjrtLoadedExecutable = MkPjrtLoadedExecutable GCAnyPtr

export
pjrtClientCompile :
  PjrtApi ->
  PjrtClient ->
  PjrtProgram ->
  GCPtr Char ->
  Int ->
  ErrIO PjrtError PjrtLoadedExecutable
pjrtClientCompile
  (MkPjrtApi api)
  (MkPjrtClient client)
  (MkPjrtProgram program)
  compileOptions
  compileOptionsSize = do
    putStrLn "pjrtClientCompile ..."
    args <- primIO $ prim__mkPjrtClientCompileArgs client program compileOptions compileOptionsSize
    err <- primIO $ prim__pjrtClientCompile api args
    let executable = prim__pjrtClientCompileArgsExecutable args
    free args
    try api err =<< do
      executable <- onCollectAny executable destroyExecutable
      pure $ MkPjrtLoadedExecutable executable

    where

    destroyExecutable : AnyPtr -> IO ()
    destroyExecutable executable = do
      args <- primIO $ prim__mkPjrtLoadedExecutableDestroyArgs executable
      err <- primIO $ prim__pjrtLoadedExecutableDestroy api args
      free args
      handleErrOnDestroy api err "PJRT_LoadedExecutable"

%foreign (libxla "PJRT_ExecuteOptions_new")
prim__mkPjrtExecuteOptions : PrimIO AnyPtr

%foreign (libxla "PJRT_LoadedExecutable_Execute_Args_new")
prim__mkPjrtLoadedExecutableExecuteArgs : GCAnyPtr -> AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_loadedexecutable_execute")
prim__pjrtLoadedExecutableExecute : AnyPtr -> AnyPtr -> PrimIO AnyPtr

export
data PjrtBuffer = MkPjrtBuffer AnyPtr  -- will be GCAnyPtr once completed

export
pjrtLoadedExecutableExecute : PjrtApi -> PjrtLoadedExecutable -> ErrIO PjrtError PjrtBuffer
pjrtLoadedExecutableExecute (MkPjrtApi api) (MkPjrtLoadedExecutable executable) = do
  putStrLn "pjrtLoadedExecutableExecute ..."
  outputListsInner <- malloc sizeofPtr
  outputLists <- malloc sizeofPtr
  primIO $ prim__setArrayPtr outputLists 0 outputListsInner
  options <- primIO prim__mkPjrtExecuteOptions
  args <- primIO $ prim__mkPjrtLoadedExecutableExecuteArgs executable options outputLists
  err <- primIO $ prim__pjrtLoadedExecutableExecute api args
  let buffer = prim__index 0 outputListsInner
  free outputListsInner
  free outputLists
  free options
  free args
  try api err $ MkPjrtBuffer buffer  -- todo gc with PJRT_Buffer_Destroy

%foreign (libxla "PJRT_Buffer_ToHostBuffer_Args_new")
prim__mkPjrtBufferToHostBufferArgs : AnyPtr -> AnyPtr -> Int -> PrimIO AnyPtr

%foreign (libxla "pjrt_buffer_tohostbuffer")
prim__pjrtBufferToHostBuffer : AnyPtr -> AnyPtr -> PrimIO AnyPtr

export
pjrtBufferToHostBuffer : PjrtApi -> PjrtBuffer -> Literal -> ErrIO PjrtError ()
pjrtBufferToHostBuffer (MkPjrtApi api) (MkPjrtBuffer buffer) (MkLiteral literal) = do
  putStrLn "pjrtBufferToHostBuffer ..."
  let untypedData = prim__literalUntypedData literal
      sizeBytes = prim__literalSizeBytes literal
  args <- primIO $ prim__mkPjrtBufferToHostBufferArgs buffer untypedData sizeBytes
  err <- primIO $ prim__pjrtBufferToHostBuffer api args
  free args
  try api err ()
