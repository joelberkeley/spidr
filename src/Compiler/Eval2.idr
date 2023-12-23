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
module Compiler.Eval2

import Control.Monad.Error.Either

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

data Err where

0 ErrIO : Type -> Type
ErrIO a = EitherT Err IO

0 CRes : Type -> Type -> Type
CRes a b = Res a (const b)

0 Cache : Nat -> Type
Cache n = MArray n (Maybe XlaOp)

interpretE : XlaBuilder -> Expr -> Cache n -@ IO (LEither Err (CRes XlaOp (Cache n)))

interpret : XlaBuilder -> List (Nat, Expr) -> Cache n -@ IO (LEither Err $ Cache n)
interpret _          []        cache = pure (Right cache)
interpret xlaBuilder ((n, x) :: nxs) cache = do
  cacheOrErr <- interpret xlaBuilder xs cache
  case cacheOrErr of
    Left err => Left err
    Right cache => do
      cacheOrErr <- interpretE xlaBuilder x cache
      case cacheOrErr of
        Left err => Left err
        Right (xlaOp # cache) => case (natToFin n) of
          Nothing => ?interpret_natToFin_err
          Just n => pure $ Right $ set n xlaOp cache

interpretE xlaBuilder (FromLiteral {dtype} lit) cache = do
  xlaOp <- constantLiteral builder !(write {dtype} lit)
  pure $ Right $ (xlaOp # cache)
interpretE _          (Diag x) cache = do
  let (xlaOp # cache) = get x cache
  xlaOp <- getMatrixDiagonal xlaOp
  pure $ Right $ (xlaOp # cache)
interpretE _          _        _     = ?interpretE_rhs

compile : String -> Nat -> Env -> ErrIO XlaComputation
compile builderName root = do
  xlaBuilder <- mkXlaBuilder builderName
  let (max, env) = toList env  -- convert all Nat to Fin n here to save the headache later on
  root <- alloc max Nothing foo
  build xlaBuilder root

  where

  foo : Cache n -@ Ur (ErrIO XlaOp)
  foo cache = U $ do
    cacheOrErr <- lift $ interpret xlaBuilder env cache
    case cacheOrErr of
      Left err => lift $ left err
      Right cache => case (natToFin root) of
        Nothing => discarding $ cache ?compile_rhs_natToFin_err
        Just root => let (maybeXlaOp # cache) = get root cache
                      in case maybeXlaOp of
                           Nothing => ?compile_rhs_noXlaOp_err
                           Just xlaOp => lift $ right $ discarding cache xlaOp

export
execute : PrimitiveRW dtype a => Nat -> Env -> {shape : _} -> ErrIO (Literal shape a)
execute root env = do
  computation <- compile "root" root env
  gpuStatus <- validateGPUMachineManager
  platform <- if ok gpuStatus then gpuMachineManager else getPlatform "Host"
  client <- getOrCreateLocalClient platform
  lit <- executeAndTransfer client computation
  pure (read {dtype} lit)
