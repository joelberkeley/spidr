{--
Copyright 2022 Joel Berkeley

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
module Compiler.LiteralRW

import Compiler.Xla.Xla.XlaData
import public Compiler.Xla.Xla.Literal
import Compiler.Xla.Xla.ShapeUtil
import Literal
import Util

range : (n : Nat) -> Literal [n] Nat
range n = impl n []
  where
  impl : (p : Nat) -> Literal [q] Nat -> Literal [q + p] Nat
  impl Z xs = rewrite plusZeroRightNeutral q in xs
  impl (S p) xs = rewrite sym $ plusSuccRightSucc q p in impl p (Scalar p :: xs)

indexed : {shape : _} -> Literal shape (List Nat)
indexed = go shape []
  where
  concat : Literal [d] (Literal ds a) -> Literal (d :: ds) a
  concat [] = []
  concat (Scalar x :: xs) = x :: concat xs

  go : (shape : Types.Shape) -> List Nat -> Literal shape (List Nat)
  go [] idxs = Scalar idxs
  go (0 :: _) _ = []
  go (S d :: ds) idxs = concat $ map (\i => go ds (snoc idxs i)) (range (S d))

public export
interface Primitive dtype => LiteralRW dtype ty where
  set : Literal -> List Nat -> ShapeIndex -> ty -> IO ()
  get : Literal -> List Nat -> ShapeIndex -> ty

export
write : HasIO io =>
        LiteralRW dtype a =>
        {shape : _} ->
        List Nat ->
        Literal shape a ->
        io Literal
write idxs xs = liftIO $ do
  literal <- allocLiteral {dtype} shape
  shapeIndex <- allocShapeIndex
  traverse_ (pushBack shapeIndex) idxs
  sequence_ [| (\idxs => set {dtype} literal idxs shapeIndex) indexed xs |]
  pure literal

export
read : HasIO io =>
       LiteralRW dtype a =>
       {shape : _} ->
       List Nat ->
       Literal ->
       io $ Literal shape a
read idxs lit = do
  shapeIndex <- allocShapeIndex
  traverse_ (pushBack shapeIndex) idxs
  pure $ map (\mIdx => get {dtype} lit mIdx shapeIndex) (indexed {shape})

export
LiteralRW PRED Bool where
  set = set
  get = get

export
LiteralRW F64 Double where
  set = set
  get = get

export
LiteralRW S32 Int32 where
  set = set
  get = get

export
LiteralRW U32 Nat where
  set = UInt32t.set
  get = UInt32t.get

export
LiteralRW U64 Nat where
  set = UInt64t.set
  get = UInt64t.get
