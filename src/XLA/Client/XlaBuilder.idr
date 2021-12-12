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
||| This module contains the Idris API to XLA.
module XLA.Client.XlaBuilder

import XLA
import Types
import System.FFI

libxla : String -> String
libxla fname = "C:" ++ fname ++ ",libxla"

xla_crash : Show a => a -> b
xla_crash x = (assert_total idris_crash) $ "Fatal: XLA C API produced unexpected value " ++ show x

%foreign (libxla "c__XlaBuilder_del")
prim__delete_XlaBuilder : AnyPtr -> PrimIO ()

export
data XlaBuilder : Type where
    MkXlaBuilder : IO GCAnyPtr -> XlaBuilder

export
mkXlaBuilder : String -> XlaBuilder
mkXlaBuilder computation_name = MkXlaBuilder $
    onCollectAny (prim__mkXlaBuilder computation_name) $ \x => do putStrLn "finaliser delete in mkXlaBuilder"
                                                                  primIO $ prim__delete_XlaBuilder x
        where %foreign (libxla "c__XlaBuilder_new")
              prim__mkXlaBuilder : String -> AnyPtr

%foreign (libxla "c__XlaBuilder_name")
prim__XlaBuilder_name : GCAnyPtr -> String

export
name : XlaBuilder -> IO String
name (MkXlaBuilder b) = pure $ prim__XlaBuilder_name !b

export
data XlaOp : Type where
    MkXlaOp : IO GCAnyPtr -> XlaOp

%foreign (libxla "c__XlaOp_del")
prim__delete_XlaOp : AnyPtr -> PrimIO ()

%foreign (libxla "c__XlaOp_builder")
prim__XlaOp_builder : GCAnyPtr -> AnyPtr

%foreign (libxla "c__ConstantR0")
prim__const : GCAnyPtr -> Int -> AnyPtr

export
const : XlaBuilder -> Int -> XlaOp
const (MkXlaBuilder builder_ptr) value =
    MkXlaOp $ do builder_ptr <- builder_ptr
                 onCollectAny (prim__const builder_ptr value) $ primIO . prim__delete_XlaOp

-- todo how to do we explain using AnyPtr here?
%foreign (libxla "c__XlaBuilder_OpToString")
prim__opToString : AnyPtr -> GCAnyPtr -> String

-- note this XlaOp must be registered in some sense with this XlaBuilder, else this function
-- errors. We can fix this by using the .builder() method on XlaOp.
export
opToString : XlaOp -> IO String
opToString op@(MkXlaOp op_ptr) =
    do op_ptr <- op_ptr
       pure $ prim__opToString (prim__XlaOp_builder op_ptr) op_ptr

%foreign (libxla "c__XlaOp_operator_add")
prim__XlaOp_operator_add : GCAnyPtr -> GCAnyPtr -> AnyPtr

export
(+) : XlaOp -> XlaOp -> XlaOp
(MkXlaOp l) + (MkXlaOp r) = MkXlaOp $
    do l <- l
       r <- r
       onCollectAny (prim__XlaOp_operator_add l r) $ primIO . prim__delete_XlaOp

%foreign (libxla "eval_int32")
prim__eval_int : GCAnyPtr -> PrimIO Int

export
eval_int : XlaOp -> IO (Array [] {dtype=Int})
eval_int (MkXlaOp op_ptr) = primIO $ prim__eval_int !op_ptr
