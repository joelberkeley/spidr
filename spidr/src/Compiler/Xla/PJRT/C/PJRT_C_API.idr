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
import Data.SortedMap
import Derive.Prelude
import Language.Reflection

import Compiler.FFI
import Compiler.Xla.Literal
import Types
import Util

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

%foreign (libxla "PJRT_Error_Destroy_Args_new")
prim__mkPjrtErrorDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_error_destroy")
prim__pjrtErrorDestroy : AnyPtr -> AnyPtr -> PrimIO ()

destroyPjrtError : HasIO io => AnyPtr -> AnyPtr -> io ()
destroyPjrtError api err = do
  args <- primIO $ prim__mkPjrtErrorDestroyArgs err
  primIO $ prim__pjrtErrorDestroy api args
  free args

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
data PjrtEvent = MkPjrtEvent AnyPtr

%foreign (libxla "PJRT_Event_Destroy_Args_new")
prim__mkPjrtEventDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_event_destroy")
prim__pjrtEventDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Event_Await_Args_new")
prim__mkPjrtEventAwaitArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_event_await")
prim__pjrtEventAwait : AnyPtr -> AnyPtr -> PrimIO AnyPtr

export
pjrtEventAwait : PjrtApi -> PjrtEvent -> ErrIO PjrtError ()
pjrtEventAwait (MkPjrtApi api) (MkPjrtEvent event) = do
  args <- primIO $ prim__mkPjrtEventAwaitArgs event
  err <- primIO $ prim__pjrtEventAwait api args
  free args
  try api err ()

export
data PjrtClient = MkPjrtClient GCAnyPtr

%foreign (libxla "PJRT_Client_Create_Args_new")
prim__mkPjrtClientCreateArgs : AnyPtr -> Bits64 -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Create_Args_client")
prim__pjrtClientCreateArgsClient : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_client_create")
prim__pjrtClientCreate : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Destroy_Args_new")
prim__mkPjrtClientDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_client_destroy")
prim__pjrtClientDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

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

public export
data PjrtValue =
    PjrtString String
  | PjrtInt64 Int64
  | PjrtInt64Array (List Int64)
  | PjrtFloat Double
  | PjrtBool Bool

%foreign (libxla "PJRT_NamedValue_string")
prim__pjrtNamedValue_String : AnyPtr -> String -> Bits64 -> String -> Bits64 -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_int64")
prim__pjrtNamedValue_Int64 : AnyPtr -> String -> Bits64 -> Int64 -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_int64_array")
prim__pjrtNamedValue_Int64Array : AnyPtr -> String -> Bits64 -> Ptr Int64 -> Bits64 -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_float")
prim__pjrtNamedValue_Float : AnyPtr -> String -> Bits64 -> Double -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_bool")
prim__pjrtNamedValue_Bool : AnyPtr -> String -> Bits64 -> Int -> PrimIO ()

%foreign (libxla "set_array_int64_t")
prim__setInt64 : Ptr Int64 -> Bits64 -> Int64 -> PrimIO ()

pjrtNamedValue : HasIO io => AnyPtr -> String -> PjrtValue -> io ()
pjrtNamedValue addr name (PjrtString string) = primIO $ prim__pjrtNamedValue_String addr name (cast $ length name) string (cast $ length string)
pjrtNamedValue addr name (PjrtInt64 int64) = primIO $ prim__pjrtNamedValue_Int64 addr name (cast $ length name) int64
pjrtNamedValue addr name (PjrtInt64Array int64s) = do
  let sizeof_int64_t = 8
  -- arr needs freeing
  arr <- prim__castPtr <$> malloc (cast (length int64s) * sizeof_int64_t)
  traverse_ (\(idx, x) => primIO $ prim__setInt64 arr (cast idx) x) (enumerate int64s)
  primIO $ prim__pjrtNamedValue_Int64Array addr name (cast $ length name) arr (cast $ length int64s)
pjrtNamedValue addr name (PjrtFloat double) = primIO $ prim__pjrtNamedValue_Float addr name (cast $ length name) double
pjrtNamedValue addr name (PjrtBool bool) = primIO $ prim__pjrtNamedValue_Bool addr name (cast $ length name) (boolToCInt bool)

%foreign (libxla "sizeof_PJRT_NamedValue")
prim__sizeofPjrtNamedValue : Bits64

%foreign (libxla "shift_by_PJRT_NamedValue")
prim__shift_by_PJRT_NamedValue : AnyPtr -> Bits64 -> AnyPtr

export
pjrtClientCreate : PjrtApi -> SortedMap String PjrtValue -> ErrIO PjrtError PjrtClient
pjrtClientCreate (MkPjrtApi api) createOptions = do
  let createOptions = toList createOptions
      numOptions = List.length createOptions

  createOptionsArr <- malloc (cast numOptions * cast prim__sizeofPjrtNamedValue)

  let setOpt : (Nat, String, PjrtValue) -> IO ()
      setOpt (idx, name, value) =
        let addr = prim__shift_by_PJRT_NamedValue createOptionsArr (cast idx)
         in pjrtNamedValue addr name value

  traverse_ (liftIO . setOpt) (enumerate createOptions)
  args <- primIO $ prim__mkPjrtClientCreateArgs createOptionsArr (cast numOptions)
  err <- primIO $ prim__pjrtClientCreate api args
  -- free int64 array
  free createOptionsArr
  let client = prim__pjrtClientCreateArgsClient args
  free args
  try api err =<< do
    client <- onCollectAny client destroy
    pure $ MkPjrtClient client

    where

    destroy : AnyPtr -> IO ()
    destroy client = do
      args <- primIO $ prim__mkPjrtClientDestroyArgs client
      err <- primIO $ prim__pjrtClientDestroy api args
      free args
      handleErrOnDestroy api err "PJRT_Client"

export
data PjrtProgram = MkPjrtProgram GCAnyPtr

%foreign (libxla "PJRT_Program_new")
prim__mkPjrtProgram : Ptr Char -> Bits64 -> PrimIO AnyPtr

||| The CharArray must live as long as the PjrtProgram.
export
mkPjrtProgram : HasIO io => CharArray -> io PjrtProgram
mkPjrtProgram (MkCharArray code codeSize) = do
  ptr <- primIO $ prim__mkPjrtProgram code codeSize
  ptr <- onCollectAny ptr free
  pure (MkPjrtProgram ptr)

%foreign (libxla "PJRT_Client_Compile_Args_new")
prim__mkPjrtClientCompileArgs : GCAnyPtr -> GCAnyPtr -> Ptr Char -> Bits64 -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Compile_Args_executable")
prim__pjrtClientCompileArgsExecutable : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_client_compile")
prim__pjrtClientCompile : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_LoadedExecutable_Destroy_Args_new")
prim__mkPjrtLoadedExecutableDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_loadedexecutable_destroy")
prim__pjrtLoadedExecutableDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

export
data PjrtLoadedExecutable = MkPjrtLoadedExecutable AnyPtr

export
pjrtLoadedExecutableDestroy : HasIO io => PjrtApi -> PjrtLoadedExecutable -> io () -- note this could now be ErrIO PjrtError ()
pjrtLoadedExecutableDestroy (MkPjrtApi api) (MkPjrtLoadedExecutable executable) = do
  args <- primIO $ prim__mkPjrtLoadedExecutableDestroyArgs executable
  err <- primIO $ prim__pjrtLoadedExecutableDestroy api args
  free args
  handleErrOnDestroy api err "PJRT_LoadedExecutable"

||| It is up to the caller to free the PjrtLoadedExecutable.
export
pjrtClientCompile :
  PjrtApi ->
  PjrtClient ->
  PjrtProgram ->
  CharArray ->
  ErrIO PjrtError PjrtLoadedExecutable
pjrtClientCompile
  (MkPjrtApi api)
  (MkPjrtClient client)
  (MkPjrtProgram program)
  (MkCharArray options optionsSize) = do
    args <- primIO $ prim__mkPjrtClientCompileArgs client program options optionsSize
    err <- primIO $ prim__pjrtClientCompile api args
    let executable = prim__pjrtClientCompileArgsExecutable args
    free args
    try api err $ MkPjrtLoadedExecutable executable

%foreign (libxla "PJRT_ExecuteOptions_new")
prim__mkPjrtExecuteOptions : PrimIO AnyPtr

%foreign (libxla "PJRT_LoadedExecutable_Execute_Args_new")
prim__mkPjrtLoadedExecutableExecuteArgs : AnyPtr -> AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_loadedexecutable_execute")
prim__pjrtLoadedExecutableExecute : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Buffer_Destroy_Args_new")
prim__mkPjrtBufferDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_buffer_destroy")
prim__pjrtBufferDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

export
data PjrtBuffer = MkPjrtBuffer AnyPtr

export
pjrtBufferDestroy : HasIO io => PjrtApi -> PjrtBuffer -> io () -- note this could now be ErrIO PjrtError ()
pjrtBufferDestroy (MkPjrtApi api) (MkPjrtBuffer buffer) = do
  args <- primIO $ prim__mkPjrtBufferDestroyArgs buffer
  err <- primIO $ prim__pjrtBufferDestroy api args
  free args
  handleErrOnDestroy api err "PJRT_Buffer"

||| It is up to the caller to free the `PjrtBuffer`.
export
pjrtLoadedExecutableExecute : PjrtApi -> PjrtLoadedExecutable -> ErrIO PjrtError PjrtBuffer
pjrtLoadedExecutableExecute (MkPjrtApi api) (MkPjrtLoadedExecutable executable) = do
  outputListsInner <- malloc sizeofPtr
  outputLists <- malloc sizeofPtr
  primIO $ prim__setArrayPtr outputLists 0 outputListsInner
  options <- primIO prim__mkPjrtExecuteOptions
  args <- primIO $ prim__mkPjrtLoadedExecutableExecuteArgs executable options outputLists
  err <- primIO $ prim__pjrtLoadedExecutableExecute api args
  let buffer = prim__index 0 outputListsInner
  free args
  free outputLists
  free outputListsInner
  free options
  try api err $ MkPjrtBuffer buffer

%foreign (libxla "PJRT_Buffer_ToHostBuffer_Args_new")
prim__mkPjrtBufferToHostBufferArgs : AnyPtr -> AnyPtr -> Int -> PrimIO AnyPtr

%foreign (libxla "PJRT_Buffer_ToHostBuffer_Args_event")
prim__pjrtBufferToHostBufferArgsEvent : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_buffer_tohostbuffer")
prim__pjrtBufferToHostBuffer : AnyPtr -> AnyPtr -> PrimIO AnyPtr

export
pjrtEventDestroy : HasIO io => PjrtApi -> PjrtEvent -> io ()
pjrtEventDestroy (MkPjrtApi api) (MkPjrtEvent event) = do
  args <- primIO $ prim__mkPjrtEventDestroyArgs event
  err <- primIO $ prim__pjrtEventDestroy api args
  free args
  handleErrOnDestroy api err "PJRT_Event"

||| It is up to the caller to free the `PjrtEvent`.
export
pjrtBufferToHostBuffer : PjrtApi -> PjrtBuffer -> Literal -> ErrIO PjrtError PjrtEvent
pjrtBufferToHostBuffer (MkPjrtApi api) (MkPjrtBuffer buffer) (MkLiteral literal) = do
  let untypedData = prim__literalUntypedData literal
      sizeBytes = prim__literalSizeBytes literal
  args <- primIO $ prim__mkPjrtBufferToHostBufferArgs buffer untypedData sizeBytes
  err <- primIO $ prim__pjrtBufferToHostBuffer api args
  let event = prim__pjrtBufferToHostBufferArgsEvent args
  free args
  try api err $ MkPjrtEvent event
