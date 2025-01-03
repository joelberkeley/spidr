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
||| For internal spidr use, and use by plugin developers.
|||
||| The Idris API for PJRT.
module Compiler.Xla.PJRT.C.PjrtCApi

import Data.SortedMap
import public Control.Monad.Either
import Derive.Prelude
import Language.Reflection

import Compiler.FFI
import Compiler.Xla.Literal
import Types
import Util

%language ElabReflection

%foreign (libxla "sizeof_PJRT_NamedValue")
sizeofPjrtNamedValue : Bits64

public export
data PjrtValue
  = PjrtValueString String
  | PjrtValueInt64 Int64
  | PjrtValueInt64List (List Int64)
  | PjrtValueFloat Double
  | PjrtValueBool Bool

%foreign (libxla "PJRT_NamedValue_array_set_string")
prim__pjrtNamedValueArraySetString :
  AnyPtr -> Bits64 -> String -> Bits64 -> String -> Bits64 -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_array_set_int64")
prim__pjrtNamedValueArraySetInt64 : AnyPtr -> Bits64 -> String -> Bits64 -> Int64 -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_array_set_int64list")
prim__pjrtNamedValueArraySetInt64List :
  AnyPtr -> Bits64 -> String -> Bits64 -> AnyPtr -> Bits64 -> PrimIO ()

-- this really only accepts float ... how to handle?
%foreign (libxla "PJRT_NamedValue_array_set_float")
prim__pjrtNamedValueArraySetFloat : AnyPtr -> Bits64 -> String -> Bits64 -> Double -> PrimIO ()

%foreign (libxla "PJRT_NamedValue_array_set_bool")
prim__pjrtNamedValueArraySetBool : AnyPtr -> Bits64 -> String -> Bits64 -> Int -> PrimIO ()

mkPJRTNamedValueArray : SortedMap String PjrtValue -> IO (AnyPtr, IO ())
mkPJRTNamedValueArray xs = do
  -- putStrLn "mkPJRTNamedValueArray"
  let xs = toList xs
  arr <- malloc (cast (length xs) * cast sizeofPjrtNamedValue)
  finalizers <- traverse (\(idx, nv) => uncurry (set arr (cast idx)) nv) (enumerate xs)
  pure (arr, sequence_ finalizers)

  where

  set : AnyPtr -> Bits64 -> String -> PjrtValue -> IO (IO ())
  set arr idx name =
    let idx = cast idx
        nameLen = cast $ length name
     in \case
      PjrtValueString x => do
        primIO $ prim__pjrtNamedValueArraySetString arr idx name nameLen x (cast $ length x)
        pure (pure ())
      PjrtValueInt64 x => do
        primIO $ prim__pjrtNamedValueArraySetInt64 arr idx name nameLen x
        pure (pure ())
      PjrtValueInt64List xs => do
        int64s <- malloc (cast (length xs) * cast sizeofInt64)
        traverse_ (\(idx, x) => primIO $ prim__setArrayInt64 int64s (cast idx) (cast x)) (enumerate xs)
        primIO $ prim__pjrtNamedValueArraySetInt64List
          arr idx name nameLen int64s (cast $ length xs)
        pure (free int64s)
      PjrtValueFloat x => do
        primIO $ prim__pjrtNamedValueArraySetFloat arr idx name nameLen x
        pure (pure ())
      PjrtValueBool x => do
        primIO $ prim__pjrtNamedValueArraySetBool arr idx name nameLen (boolToCInt x)
        pure (pure ())

||| For use by plugin developers.
|||
||| A minimal wrapper round a C `PJRT_Api` struct pointer. The memory should be owned by the
||| code producing the pointer.
public export
data PjrtApi = MkPjrtApi AnyPtr

||| The cause of a `PjrtError`.
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

||| Indicates an error in the PJRT C layer, either due to internal errors or user error.
public export
record PjrtError where
  constructor MkPjrtError

  ||| The error message.
  message : String

  ||| The error cause code, if one exists.
  code : Maybe PjrtErrorCode

export
Show PjrtError where
  show e =
    let code = case e.code of
          Nothing => "unknown"
          Just c => show c
     in "PjrtError (error code \{code})\n\{e.message}"

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

||| A `Pjrt a` produces an `a` or an error from the PJRT layer.
public export 0
Pjrt : Type -> Type
Pjrt = EitherT PjrtError IO

try : AnyPtr -> AnyPtr -> a -> Pjrt a
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

||| For internal spidr use only.
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

||| For internal spidr use only.
export
pjrtEventAwait : PjrtApi -> PjrtEvent -> Pjrt ()
pjrtEventAwait (MkPjrtApi api) (MkPjrtEvent event) = do
  args <- primIO $ prim__mkPjrtEventAwaitArgs event
  err <- primIO $ prim__pjrtEventAwait api args
  free args
  try api err ()

||| For use by plugin developers.
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

||| For use by plugin developers.
|||
||| Create a `PjrtClient`.
|||
||| @createOptions Platform-specific options. See plugin documentation for details.
export
pjrtClientCreate : PjrtApi -> (createOptions : SortedMap String PjrtValue) -> Pjrt PjrtClient
pjrtClientCreate (MkPjrtApi api) createOptions = do
  --putStrLn "pjrtClientCreate"
  --printLn 1
  (createOptionsPtr, createOptionsFinalizers) <- liftIO $ mkPJRTNamedValueArray createOptions
  --printLn 2
  args <- primIO $ prim__mkPjrtClientCreateArgs createOptionsPtr (cast $ length $ Prelude.toList createOptions)
  --printLn 3
  err <- primIO $ prim__pjrtClientCreate api args
  --printLn 4
  let client = prim__pjrtClientCreateArgsClient args
  -- printLn 5
  free args
  liftIO createOptionsFinalizers
  --putStrLn "pjrtClientCreate return"
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

-- docstring
public export
data PjrtTopologyDescription = MkPjrtTopologyDescription GCAnyPtr

%foreign (libxla "PJRT_Client_TopologyDescription_Args_new")
prim__mkPjrtClientTopologyDescriptionArgs : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_TopologyDescription_Args_topology")
prim__pjrtClientTopologyDescriptionArgsTopology : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_client_topologydescription")
prim__pjrtClientTopologyDescription : AnyPtr -> AnyPtr -> PrimIO AnyPtr

-- docstring ... this is probably exposed to user
export
pjrtClientTopologyDescription : PjrtApi -> PjrtClient -> Pjrt PjrtTopologyDescription
pjrtClientTopologyDescription (MkPjrtApi api) (MkPjrtClient client) = do
  args <- primIO $ prim__mkPjrtClientTopologyDescriptionArgs client
  err <- primIO $ prim__pjrtClientTopologyDescription api args
  topology <- primIO $ prim__pjrtClientTopologyDescriptionArgsTopology args
  free args
  try api err =<< do
    topology <- onCollectAny topology (const $ pure ())  -- client owns the topology
    pure $ MkPjrtTopologyDescription topology

%foreign (libxla "PJRT_Client_Devices_Args_new")
prim__mkPjrtClientDevicesArgs : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Client_Devices_Args_devices")
prim__pjrtClientDevicesArgsDevices : AnyPtr -> AnyPtr

%foreign (libxla "PJRT_Client_Devices_Args_num_devices")
prim__pjrtClientDevicesArgsNumDevices : AnyPtr -> Bits64

%foreign (libxla "pjrt_client_devices")
prim__pjrtClientDevices : AnyPtr -> AnyPtr -> PrimIO AnyPtr

--docstring
public export
data PjrtDevice = MkPjrtDevice AnyPtr  -- owned by client

||| write me.
export
pjrtClientDevices : PjrtApi -> PjrtClient -> Pjrt (List PjrtDevice)
pjrtClientDevices (MkPjrtApi api) (MkPjrtClient client) = do
  args <- primIO $ prim__mkPjrtClientDevicesArgs client
  err <- primIO $ prim__pjrtClientDevices api args
  let argsDevices = prim__pjrtClientDevicesArgsDevices args
      argsNumDevices = prim__pjrtClientDevicesArgsNumDevices args
  let devices = Prelude.map (\i => MkPjrtDevice $ prim__index (cast i) argsDevices) (range $ cast argsNumDevices)
  free args
  try api err devices

||| For internal spidr use only.
export
data PjrtProgram = MkPjrtProgram GCAnyPtr

%foreign (libxla "PJRT_Program_new")
prim__mkPjrtProgram : Ptr Char -> Bits64 -> PrimIO AnyPtr

||| For internal spidr use only.
|||
||| The `CharArray` must live as long as the `PjrtProgram`.
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

%foreign (libxla "PJRT_Client_DefaultDeviceAssignment_Args_new")
prim__mkPjrtClientDefaultDeviceAssignmentArgs :
  GCAnyPtr -> Int -> Int -> Bits64 -> Ptr Int -> PrimIO AnyPtr

%foreign (libxla "pjrt_client_defaultdeviceassignment")
prim__pjrtClientDefaultDeviceAssignment : AnyPtr -> AnyPtr -> PrimIO AnyPtr

||| For internal spidr use only.
export
pjrtClientDefaultDeviceAssignment :
  PjrtApi -> PjrtClient -> (num_replicas, num_partitions : Nat) -> Pjrt (List Int)
pjrtClientDefaultDeviceAssignment
  (MkPjrtApi api) (MkPjrtClient client) num_replicas num_partitions = do
    let defaultAssignmentSize = num_replicas * num_partitions
    defaultAssignment <- prim__castPtr <$> malloc (cast defaultAssignmentSize * sizeofInt)
    args <- primIO $ prim__mkPjrtClientDefaultDeviceAssignmentArgs
      client
      (cast num_replicas)
      (cast num_partitions)
      (cast defaultAssignmentSize)
      defaultAssignment
    err <- primIO $ prim__pjrtClientDefaultDeviceAssignment api args
    free args
    let defaultAssignment' =
          [prim__indexInt (cast idx) defaultAssignment | idx <- range defaultAssignmentSize]
    free $ prim__forgetPtr defaultAssignment
    try api err $ defaultAssignment'

-------------------------- Device Descriptions ------------------------------

-- docstring
public export
data PjrtDeviceDescription = MkPjrtDeviceDescription AnyPtr  -- not GCAnyPtr as there's no PJRT function to delete it

%foreign (libxla "PJRT_DeviceDescription_DebugString_Args_new")
prim__mkPjrtDeviceDescriptionDebugStringArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Device_GetDescription_Args_debug_string")
prim__pjrtDeviceDescriptionDebugStringArgsDebugString : AnyPtr -> PrimIO String

%foreign (libxla "pjrt_devicedescription_debugstring")
prim__pjrtDeviceDescriptionDebugString : AnyPtr -> AnyPtr -> PrimIO AnyPtr

-- docstring ... this is probably exposed to user
export
pjrtDeviceDescriptionDebugString : PjrtApi -> PjrtDeviceDescription -> Pjrt String
pjrtDeviceDescriptionDebugString (MkPjrtApi api) (MkPjrtDeviceDescription descr) = do
    args <- primIO $ prim__mkPjrtDeviceDescriptionDebugStringArgs descr
    err <- primIO $ prim__pjrtDeviceDescriptionDebugString api args
    debugString <- primIO $ prim__pjrtDeviceDescriptionDebugStringArgsDebugString args
    free args
    try api err debugString

--------------------------------- Devices -----------------------------------

%foreign (libxla "PJRT_Device_GetDescription_Args_new")
prim__mkPjrtDeviceGetDescriptionArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Device_GetDescription_Args_device_description")
prim__pjrtDeviceGetDescriptionArgsDeviceDescription : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_device_getdescription")
prim__pjrtDeviceGetDescription : AnyPtr -> AnyPtr -> PrimIO AnyPtr

-- docstring ... this is probably exposed to user
export
pjrtDeviceGetDescription : PjrtApi -> PjrtDevice -> Pjrt PjrtDeviceDescription
pjrtDeviceGetDescription (MkPjrtApi api) (MkPjrtDevice device) = do
    args <- primIO $ prim__mkPjrtDeviceGetDescriptionArgs device
    err <- primIO $ prim__pjrtDeviceGetDescription api args
    descr <- primIO $ prim__pjrtDeviceGetDescriptionArgsDeviceDescription args
    free args
    try api err $ MkPjrtDeviceDescription descr

-------------------------------- Memory --------------------------------------

------------------------------- Execute Context -----------------------------

------------------------------- Executables ---------------------------------

%foreign (libxla "PJRT_LoadedExecutable_Destroy_Args_new")
prim__mkPjrtLoadedExecutableDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_loadedexecutable_destroy")
prim__pjrtLoadedExecutableDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

||| For internal spidr use only.
export
data PjrtLoadedExecutable = MkPjrtLoadedExecutable AnyPtr

||| For internal spidr use only.
export
pjrtLoadedExecutableDestroy : HasIO io => PjrtApi -> PjrtLoadedExecutable -> io ()
pjrtLoadedExecutableDestroy (MkPjrtApi api) (MkPjrtLoadedExecutable executable) = do
  args <- primIO $ prim__mkPjrtLoadedExecutableDestroyArgs executable
  err <- primIO $ prim__pjrtLoadedExecutableDestroy api args
  free args
  handleErrOnDestroy api err "PJRT_LoadedExecutable"

||| For internal spidr use only.
|||
||| It is up to the caller to free the `PjrtLoadedExecutable`.
export
pjrtClientCompile :
  PjrtApi ->
  PjrtClient ->
  PjrtProgram ->
  CharArray ->
  Pjrt PjrtLoadedExecutable
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
prim__mkPjrtLoadedExecutableExecuteArgs : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_loadedexecutable_execute")
prim__pjrtLoadedExecutableExecute : AnyPtr -> AnyPtr -> PrimIO AnyPtr

%foreign (libxla "PJRT_Buffer_Destroy_Args_new")
prim__mkPjrtBufferDestroyArgs : AnyPtr -> PrimIO AnyPtr

%foreign (libxla "pjrt_buffer_destroy")
prim__pjrtBufferDestroy : AnyPtr -> AnyPtr -> PrimIO AnyPtr

||| For internal spidr use only.
export
data PjrtBuffer = MkPjrtBuffer AnyPtr

||| For internal spidr use only.
export
pjrtBufferDestroy : HasIO io => PjrtApi -> PjrtBuffer -> io ()
pjrtBufferDestroy (MkPjrtApi api) (MkPjrtBuffer buffer) = do
  args <- primIO $ prim__mkPjrtBufferDestroyArgs buffer
  err <- primIO $ prim__pjrtBufferDestroy api args
  free args
  handleErrOnDestroy api err "PJRT_Buffer"

||| For internal spidr use only.
|||
||| It is up to the caller to free the `PjrtBuffer`s.
export
pjrtLoadedExecutableExecute :
  PjrtApi -> PjrtLoadedExecutable -> (outputs : Nat) -> PjrtDevice -> Pjrt (Vect outputs PjrtBuffer)
pjrtLoadedExecutableExecute
  (MkPjrtApi api) (MkPjrtLoadedExecutable executable) outputs (MkPjrtDevice device) = do
    outputListsInner <- malloc (cast outputs * sizeofPtr)
    outputLists <- malloc sizeofPtr
    primIO $ prim__setArrayPtr outputLists 0 outputListsInner
    options <- primIO prim__mkPjrtExecuteOptions
    args <- primIO $ prim__mkPjrtLoadedExecutableExecuteArgs executable options outputLists device
    err <- primIO $ prim__pjrtLoadedExecutableExecute api args
    free args
    free options
    let buffers = map (\o => MkPjrtBuffer $ prim__index (cast o) outputListsInner) (range outputs)
    free outputLists
    free outputListsInner
    try api err buffers

%foreign (libxla "PJRT_Buffer_ToHostBuffer_Args_new")
prim__mkPjrtBufferToHostBufferArgs : AnyPtr -> AnyPtr -> Int -> PrimIO AnyPtr

%foreign (libxla "PJRT_Buffer_ToHostBuffer_Args_event")
prim__pjrtBufferToHostBufferArgsEvent : AnyPtr -> AnyPtr

%foreign (libxla "pjrt_buffer_tohostbuffer")
prim__pjrtBufferToHostBuffer : AnyPtr -> AnyPtr -> PrimIO AnyPtr

||| For internal spidr use only.
export
pjrtEventDestroy : HasIO io => PjrtApi -> PjrtEvent -> io ()
pjrtEventDestroy (MkPjrtApi api) (MkPjrtEvent event) = do
  args <- primIO $ prim__mkPjrtEventDestroyArgs event
  err <- primIO $ prim__pjrtEventDestroy api args
  free args
  handleErrOnDestroy api err "PJRT_Event"

||| For internal spidr use only.
|||
||| It is up to the caller to free the `PjrtEvent`.
export
pjrtBufferToHostBuffer : PjrtApi -> PjrtBuffer -> Literal -> Pjrt PjrtEvent
pjrtBufferToHostBuffer (MkPjrtApi api) (MkPjrtBuffer buffer) (MkLiteral literal) = do
  let untypedData = prim__literalUntypedData literal
      sizeBytes = prim__literalSizeBytes literal
  args <- primIO $ prim__mkPjrtBufferToHostBufferArgs buffer untypedData sizeBytes
  err <- primIO $ prim__pjrtBufferToHostBuffer api args
  let event = prim__pjrtBufferToHostBufferArgsEvent args
  free args
  try api err $ MkPjrtEvent event
