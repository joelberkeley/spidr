{--
Copyright 2023 Joel Berkeley

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
module Compiler.Eval

import Data.Linear
import Data.Array
import Data.Fin
import Control.Monad.Error.Either
import Control.Monad.Trans
import Control.Linear.LIO

import Compiler.Expr
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Arithmetic
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Constants
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Math
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.Matrix
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.Lib.PRNG
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaComputation
import Compiler.Xla.TensorFlow.Compiler.Xla.Literal
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Compiler.Xla.TensorFlow.Compiler.Xla.Service.PlatformUtil
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.ClientLibrary
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.LocalClient
import Compiler.Xla.TensorFlow.Core.CommonRuntime.GPU.GPUInit
import Compiler.Xla.TensorFlow.Core.Platform.Status
import Compiler.Xla.TensorFlow.StreamExecutor.Platform

import Literal
import Primitive
import Types
import Util

export
data Err = OutOfBounds Nat Nat
         | NoValueFound Nat

Show Err where
  show (OutOfBounds idx size) = "Index \{show idx} is out of bounds for array of size \{show size}"
  show (NoValueFound idx) = "No value found at index \{show idx}"

data LEither : Type -> Type -> Type where
  Left : e -> LEither e a
  Right : (1 _ : a) -> LEither e a

0 ErrIO : Type -> Type
ErrIO a = EitherT Err IO a

0 LErrIO : Type -> Type
LErrIO a = L1 IO (LEither Err a)

0 Cache : Nat -> Type
Cache n = MArray n (Maybe XlaOp)

interpret : XlaBuilder -> List (Nat, Expr) -> {n : Nat} -> Cache n -@ LErrIO (Cache n)
interpret _          []              cache = pure1 $ Right cache
interpret xlaBuilder ((i, x) :: ixs) cache = do
  Right cache <- interpret xlaBuilder ixs cache | Left err => pure1 $ Left err
  Right (xlaOp # cache) <- interpretE x cache | Left err => ?interpret_err
  let Just i = natToFin i n | _ => ?interpret_natToFin_err
  pure1 $ Right $ set i (Just xlaOp) cache

  where

  interpretE : Expr -> Cache n -@ LErrIO (CRes XlaOp $ Cache n)
  interpretE (FromLiteral {dtype} lit) cache = do
    xlaOp <- constantLiteral xlaBuilder !(write {dtype} lit)
    pure1 $ Right $ xlaOp # cache
  interpretE (Diag x) cache = do
    let Just x' = natToFin x n | _ => pure1 $ discarding cache $ Left $ OutOfBounds x n
        Just xlaOp # cache = Core.get x' cache | _ => pure1 $ Left $ NoValueFound x
    xlaOp <- getMatrixDiagonal xlaOp
    pure1 $ Right (xlaOp # cache)
  interpretE _        _     = ?interpretE_rhs

compile : String -> Nat -> Env -> ErrIO XlaComputation
compile builderName root env = do
  xlaBuilder <- mkXlaBuilder builderName
  -- convert all Nat to Fin n here to save the headache later on
  let (max, env) = toList env
  -- consider unsafeAlloc to avoid headache of handling Maybe ... we know it's safe because it's
  -- topologically sorted
  let 1 bar0 : L1 IO (LEither Err $ !* XlaOp) = alloc max Nothing (createRoot xlaBuilder env)
      bar1 : L IO (Either Err XlaOp) = do
        root <- bar0
        pure $ unrestricted root
      bar2 : IO (Either Err XlaOp) = run bar1
      bar3 : EitherT IO Err XlaOp = do
        Right root <- bar2 | Left err => left err
        right root
  root <- bar3
  build xlaBuilder root

  where

  createRoot : XlaBuilder ->
               List (Nat, Expr) ->
               {n : Nat} ->
               Cache n -@
               L IO (!* Either Err XlaOp)
  createRoot xlaBuilder env cache = do
    Right cache <- interpret xlaBuilder env cache | Left err => pure $ MkBang $ Left err
    let Just root' = natToFin root n
          | _ => discarding cache $ pure $ MkBang $ Left $ OutOfBounds root n
        Just xlaOp # cache = Core.get root' cache
          | _ => pure $ MkBang $ Left $ NoValueFound root
    discarding cache $ pure $ MkBang $ Right xlaOp

export
execute : PrimitiveRW dtype a => Nat -> Env -> {shape : _} -> ErrIO $ Literal shape a
execute root env = do
  computation <- compile "root" root env
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
