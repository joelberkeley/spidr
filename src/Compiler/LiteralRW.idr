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

import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Compiler.Xla.TensorFlow.Compiler.Xla.Literal
import Compiler.Xla.TensorFlow.Compiler.Xla.ShapeUtil
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
write : (HasIO io, LiteralRW dtype a) => {shape : _} -> Literal shape a -> io Literal
write xs = liftIO $ do
  literal <- allocLiteral {dtype} shape
  shapeIndex <- allocShapeIndex
  sequence_ [| (\idxs => set {dtype} literal idxs shapeIndex) indexed xs |]
  pure literal

export
read : HasIO io => LiteralRW dtype a => Literal -> {shape : _} -> io $ Literal shape a
read lit = do
  shapeIndex <- allocShapeIndex
  pure $ map (\mIdx => get {dtype} lit mIdx shapeIndex) indexed

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

namespace LiteralRWVect
  public export
  data LiteralRWVect : Vect n (Shape #: Type #: Type) -> Type where
    Nil : LiteralRWVect []
    (::) : LiteralRW dtype ty ->
           LiteralRWVect stt ->
           LiteralRWVect ((shape ##:: dtype ##:: ty) :: stt)

namespace Tuple
  export
  read : HasIO io => {shapes : _} -> LiteralRWVect shapes => Literal -> io $ LiteralVect shapes
  read lit = impl 0
    where
    impl : {s : _} -> LiteralRWVect s => Nat -> io $ LiteralVect s
    impl {s = []} _ = pure []
    impl {s = ((shape ##:: dtype ##:: _) :: ss)} @{(p :: ps)} n = do
      shapeIndex <- allocShapeIndex
      pushFront shapeIndex n
      let head = map (\mIdx => get {dtype} lit mIdx shapeIndex) (indexed {shape})
      tail <- impl {s = ss} @{ps} (S n)
      pure (head :: tail)
