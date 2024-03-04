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
module Compiler.Xla.Xla.PJRT.C.PJRT_C_API

import System.FFI

import Compiler.Xla.Prim.Util

-- keep the API as close to the PJRT api as possible except:
-- * don't expose _Args, so we don't need to handle null ptrs. I really doubt we'd
--   ever need to expose them
-- use pointers for two reasons:
-- we can use onCollectAny to GC our data!
-- we can hide GCAnyPtr/AnyPtr in a more type-safe Idris API

export
data PjrtApi = MkPjrtApi AnyPtr

public export 0
ErrIO : Type -> Type -> Type
ErrIO e a = EitherT e IO a

export
data PjrtError = MkPjrtError GCAnyPtr

public export 0
PjrtErrIO : Type -> Type
PjrtErrIO = ErrIO PjrtError

%foreign (libxla "PJRT_Error_Destroy_Args_new")
prim__mkPjrtErrorDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_error_destroy")
prim__pjrtErrorDestroy : AnyPtr -> AnyPtr -> PrimIO ()

destroyPjrtError : AnyPtr -> IO ()
destroyPjrtError err = do
  destroyArgs <- primIO $ prim__mkPjrtErrorDestroyArgs err
  primIO $ prim__pjrtErrorDestroy api destroyArgs
  free destroyArgs

%foreign (libxla "PJRT_Error_Message_Args_new")
prim__mkPjrtErrorMessageArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Error_Message_Args_message")
prim__pjrtErrorMessageArgsMessage : AnyPtr -> PrimIO String

%foreign (libxla "pjrt_error_message")
prim__pjrtErrorMessage : AnyPtr -> AnyPtr -> PrimIO ()

export
pjrtErrorMessage : HasIO io => PjrtError -> io ()

%foreign (libxla "pjrt_error_getcode")
prim__pjrtErrorGetcode : AnyPtr -> GCAnyPtr -> PrimIO AnyPtr

-- i'm going to keep the Idris API as close as possible to the C API, and convert the
-- errors further up the stack
{-
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

fromCInt : Int64 -> PjrtErrorCode
fromCInt = \case
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

public export
data PjrtError = MkPjrtError String PjrtErrorCode
-}

export
data PjrtClient = MkPjrtClient GCAnyPtr

%foreign (libxla "PJRT_Client_Create_Args_new")
prim__mkPjrtClientCreateArgs : PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Create_Args_client")
prim__pjrtClientCreateArgsClient : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_client_create")
prim__pjrtClientCreate : AnyPtr -> GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Destroy_Args_new")
prim__mkPjrtClientDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Destroy")
prim__pjrtClientDestroy : AnyPtr -> PrimIO ()

export
pjrtClientCreate : PjrtApi -> ErrIO PjrtError PjrtClient
pjrtClientCreate (MkPjrtApi api) = do
  args <- primIO prim__mkPjrtClientCreateArgs
  err <- primIO $ prim__pjrtClientCreate api args
  if isNullPtr err
  then do let client = prim__pjrtClientCreateArgsClient args
          free args
          client <- onCollectAny client destroyClient
          right $ PjrtClient client
  else do err <- onCollectAny err destroyPjrtError
          left $ MkPjrtError err

  where

  destroyClient : AnyPtr -> IO ()
  destroyClient client = do
    args <- primIO $ prim__mkPjrtClientDestroyArgs client
    err <- primIO $ prim__pjrtClientDestroy api args
    free args
    unless (isNullPtr err) $ do
      args <- primIO $ prim__mkPjrtErrorMessageArgs err
      prim__pjrtErrorMessage api args
      msg <- primIO $ prim__pjrtErrorMessageArgsMessage args
      -- mention the address of the client?
      printLn "Failed to destroy PJRT_Client, continuing operation."
      free msgArgs
      destroyPjrtError err

export
data PjrtProgram = MkPjrtProgram GCAnyPtr

%foreign (libxla "PJRT_Program_new")
prim__mkPjrtProgram : String -> PrimIO AnyPtr

export
mkPjrtProgram : HasIO io => String -> io PjrtProgram
mkPjrtProgram code = do
  ptr <- primIO $ prim__mkPjrtProgram code
  ptr <- onCollectAny ptr free
  pure (MkPjrtProgram ptr)
