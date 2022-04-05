{--
Copyright 2021 Joel Berkeley

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
module Compiler.TensorFlow.Compiler.XLA.Literal

import System.FFI

import Compiler.FFI
import Compiler.TensorFlow.Compiler.XLA.Shape
import Compiler.TensorFlow.Compiler.XLA.ShapeUtil
import Compiler.TensorFlow.Compiler.XLA.XlaData
import Literal
import Types
import Util

export
interface Primitive dtype => LiteralPrimitiveRW dtype ty where
  set : GCAnyPtr -> GCPtr Int -> ty -> PrimIO ()
  get : GCAnyPtr -> GCPtr Int -> ty

export
%foreign (libxla "Literal_new")
prim__allocLiteral : GCAnyPtr -> PrimIO AnyPtr

%foreign (libxla "Literal_delete")
prim__Literal_delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__Literal_delete

%foreign (libxla "Literal_Set_bool")
prim__literalSetBool : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_bool")
literalGetBool : GCAnyPtr -> GCPtr Int -> Int

export
LiteralPrimitiveRW PRED Bool where
  set lit idxs x = prim__literalSetBool lit idxs (if x then 1 else 0)
  get lit idxs = case literalGetBool lit idxs of
    0 => False
    1 => True
    x => (assert_total idris_crash) (
           "Internal error: expected 0 or 1 from XLA C API for boolean conversion, got " ++ show x
         )

%foreign (libxla "Literal_Set_double")
prim__literalSetDouble : GCAnyPtr -> GCPtr Int -> Double -> PrimIO ()

%foreign (libxla "Literal_Get_double")
literalGetDouble : GCAnyPtr -> GCPtr Int -> Double

export
LiteralPrimitiveRW F64 Double where
  set = prim__literalSetDouble
  get = literalGetDouble

%foreign (libxla "Literal_Set_int")
prim__literalSetInt : GCAnyPtr -> GCPtr Int -> Int -> PrimIO ()

%foreign (libxla "Literal_Get_int")
literalGetInt : GCAnyPtr -> GCPtr Int -> Int

export
LiteralPrimitiveRW S32 Int where
  set = prim__literalSetInt
  get = literalGetInt

export
LiteralPrimitiveRW U32 Nat where
  set lit idx x = prim__literalSetInt lit idx (cast x)
  get = cast .: literalGetInt

enumerate : {d : _} -> {ds : _} -> Literal (d :: ds) dtype -> Vect d (Nat, Literal ds dtype)
enumerate xs = Vect.enumerate (toVect xs) where
  toVect : {0 d : _} -> Literal (d :: ds) dtype -> Vect d (Literal ds dtype)
  toVect {d=0} [] = []
  toVect (x :: xs) = x :: toVect xs

populateLiteral : {shape : _} -> LiteralPrimitiveRW dtype a => Literal shape a -> GCAnyPtr -> IO ()
populateLiteral {shape} lit ptr = impl shape [] lit where
  impl : (shape', idxs : Shape) -> Literal shape' a -> IO ()
  impl [] idxs (Scalar x) = primIO (set {dtype} ptr !(mkIntArray idxs) x)
  impl (0 :: _) _ _ = pure ()
  impl (S _ :: ds) idxs (x :: xs) =
    traverse_ (\(idx, ys) => impl ds (idxs ++ [idx]) ys) (enumerate (x :: xs))

export
mkLiteral : HasIO io => LiteralPrimitiveRW dtype a => {shape : _} -> Literal shape a -> io GCAnyPtr
mkLiteral xs = do
  xla_shape <- mkShape {dtype} shape
  literal <- primIO $ prim__allocLiteral xla_shape
  literal <- onCollectAny literal Literal.delete
  liftIO $ populateLiteral {dtype} xs literal
  pure literal

concat : Vect d (Literal ds a) -> Literal (d :: ds) a
concat [] = []
concat (x :: xs) = x :: concat xs

export
toLiteral : {shape : _} -> GCAnyPtr -> LiteralPrimitiveRW dtype a => Literal shape a
toLiteral lit = impl shape [] where
  impl : (shape', idxs : Shape) -> Literal shape' a
  impl [] idxs = Scalar (unsafePerformIO $ (map (get {dtype} lit) (mkIntArray idxs)))
  impl (0 :: ds) idxs = []
  impl (S d :: ds) idxs = concat $ map (\i => impl ds (snoc idxs i)) (range (S d))
