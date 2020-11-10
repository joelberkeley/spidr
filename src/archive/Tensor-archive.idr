import Data.Vect

||| [1, 1, 1] shape [3]
||| [1, 1] shape [2]
||| [1] shape [1]
||| [] shape [0] pr Z :: Nil
|||
||| [[1], [1], [1]] shape [3, 1]
||| [[1], [1]] shape [2, 1]
||| [[1]] shape [1, 1]
||| [] shape [0, 1] or Z :: [1]
|||
||| a scalar doesn't appear to be a tensor. If there's no way of representing
||| tf.constant(0) *and* an empty tensor, and an empty tensor is the logical
||| result of removing dimensions one by one, perhaps we shouldn't try to
||| represent scalars as tensors. However,
|||
||| [[1]] shape [1, 1]
||| [1] shape [1]
||| 1 shape [] (???)
|||
||| Q: what is a rank 0 tensor? if the length of the shape is the rank, it's
||| a tensor with shape []
|||
||| What is a Tensor [1 0] Int?
||| [[1, 1]] shape [1, 2]
||| [[1]] shape [1, 1]
||| [[]] shape [1, 0]
|||
||| implies any 0 in the shape corresponds to an empty tensor of different
||| forms. Any numbers after the first zero don't show in the bracketed
||| representation, so `Nil` is fine as `Tensor (Z :: s) dtype`? cos [[]] is
||| [] :: []?
|||
||| If a Tensor is defined in terms of tensors, we have to have some way of
||| promoting a dtype to a Tensor
|||
||| Note this is difficult becuase I want to use the [[1, 2]] syntax natively.
||| Q: is there any point? or should I just use Vects of Tensors?
|||
|||
||| You can think of a Tensor as a higher-dimensional list only for rank greater
||| than one. That is, scalars can't be expressed as Vects or Vects of Vects.
||| They would have to 'Vect-less', i.e. the raw data type, but that has to be
|||
-- public export
-- data Tensor : Vect rank Nat -> Type -> Type where
--   Scalar : dtype -> Tensor Nil dtype
--   Nil  : Tensor (Z :: s) dtype  -- either
--   -- Nil  : (s: Vect r Nat) -> Tensor (Z :: s) dtype  -- or specify the shape for each Nil, a bit like tf.empty([2, 4, 3])
--   (::) : Tensor s dtype -> Tensor (h :: s) dtype -> Tensor (S h :: s) dtype

-------------------------------------------------------------
-------------------------------------------------------------

----------------------------------------------------------------------
----------------------------------------------------------------------

-- having shape and dtype act on the type not the value (i.e. no leading ty ->)
-- means that TensorLikes can only be defined for dtyps that encode all the
-- type information for Tensor in their types. e.g. we may be forgoing the
-- option to define this for List Nat. We'll go with that for now cos it's
-- quite nice having that type safety
interface TensorLike ty where
  total rank : Nat
  total shape : Vect rank Nat
  total dtype : Type

{-
  Tensor design decisions:

  We observe that in relativity, a tensor transforms under coordinate changes
  along all space-time indices. One cannot simply choose a particular
  reference frame and from there a particular coordinate, and only transform
  remaining coordinates. This means that a tensor T_{ij} is, in general, not
  decomposable by row or column into other tensors T_{2j} or T_{i0}. For this
  reason, we decided not to define `Tensor`recursively. Instead, it is defined
  in terms of an atomic scalar or array.

  We chose `Vect` to be the basis for the array so that its type captures the
  shape information (albeit in a different form than the `Tensor`).

  We also want `Tensor` to have its own API, one that is clear and independent
  from any APIs present on integers, floats or vectors. We achieve this by
  making `Tensor` be a thin wrapper around the underlying data.

  We also want tensors to work just as they would in mathematical notation.
  For example, multiplication must be with a scalar of an appropriate type,
  while addition `+` can only be with a Tensor of the same shape. We may
  choose to support broadcasting at some point, but if we do, this will be via
  a separate mechanism where the broadcasting is *explicit*.
-}
data Tensor : Vect rank Nat -> Type -> Type where
  MkTensor : TensorLike ty => ty -> Tensor (shape {ty}) (dtype {ty})

TensorLike elem => TensorLike (Vect len elem) where
  rank = 1 + rank {ty = elem}
  shape {len} {elem} = len :: (shape {ty = elem})
  dtype {elem} = dtype {ty = elem}

{-
  todo there's a clear abstraction here for scalars. I just don't know how to
  write it yet
-}

TensorLike Int where
  rank = Z
  shape = []
  dtype = Int

TensorLike Nat where
  rank = Z
  shape = []
  dtype = Nat

TensorLike Double where
  rank = Z
  shape = []
  dtype = Double

{-
  todo make a Tensor a TensorLike (so that you can have tensors with different
  indices transforming under different groups)
-}

-- not compiling because it doesn't know t1 and t2 are of the same type
-- i need some more general way of zipping TensorLikes of the same shape and
-- dtype
-- total (+) : Tensor s d -> Tensor s d -> Tensor s d
-- (+) (MkTensor t1) (MkTensor t2) = MkTensor ?hole

two_by_four : Tensor [2, 4] Double
-- two_by_four = MkTensor [[-1.0, -2, -3, -4], [1, 2, 3, 4]]

-- main : IO ()
-- main = printLn
