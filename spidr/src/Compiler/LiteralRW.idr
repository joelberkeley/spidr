{--
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
--}
||| For internal spidr use only.
module Compiler.LiteralRW

import Compiler.Xla.XlaData
import public Compiler.Xla.Literal
import Compiler.Xla.Shape
import Compiler.Xla.ShapeUtil
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
  shape <- mkShape {dtype} shape
  literal <- allocLiteral shape
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
