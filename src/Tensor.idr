module Tensor

import Data.Vect

%default total

----------------------------- core definitions ----------------------------

public export
Shape : {rank: Nat} -> Type
Shape {rank} = Vect rank Nat

-- todo this type gives users a lot of flexibility, and i might end up paying for
--   it in code complexity
public export
Array : Shape -> Type -> Type
Array [] dtype = dtype
Array (d :: ds) dtype = Vect d (Array ds dtype)

public export
data Tensor : (shape: Shape) -> (dtype: Type) -> Type where
  MkTensor : Array shape dtype -> Tensor shape dtype

-- this would enable interchanging scalars and numbers, but we can add that
-- later when we know if we want it (we probably don't _need_ it)
-- Num dtype => Num (Tensor [] dtype) where

export
Show (Array shape dtype) => Show (Tensor shape dtype) where
  show (MkTensor x) = "Tensor " ++ show x

----------------------------- structural operations ----------------------------

export
index : (idx: Fin d) -> Tensor (d :: ds) dtype -> Tensor ds dtype
index idx (MkTensor x) = MkTensor $ index idx x

zipWith : {shape : Shape {rank}} -> (a -> b -> c) -> Tensor shape a -> Tensor shape b -> Tensor shape c
zipWith f (MkTensor x) (MkTensor y) = MkTensor (zipWithArray f x y) where
  zipWithArray : {shape': Shape {rank=rank'}} -> (a -> b -> c) -> Array shape' a -> Array shape' b -> Array shape' c
  zipWithArray {rank'=Z} {shape'=[]} f x y = f x y
  zipWithArray {rank'=(S k)} {shape'=(d :: ds)} f x y = zipWith (zipWithArray f) x y

export
transpose : Tensor [m, n] dtype -> Tensor [n, m] dtype
transpose (MkTensor x) = MkTensor $ transpose x

export
cross : Tensor

-- why does this work with a and b but not other names?
-- see http://docs.idris-lang.org/en/latest/tutorial/interfaces.html#functors-and-applicatives
public export
Functor (Tensor shape) where
  map f (MkTensor x) = MkTensor (g x) where
    g : Array s a -> Array s b
    g {s = []} y = f y
    g {s = (_ :: _)} ys = map g ys

export
replicate : (over : Shape) -> Tensor shape dtype -> Tensor (over ++ shape) dtype
replicate over (MkTensor x) = MkTensor (f over x) where
  f : (over: Shape) -> Array s dtype -> Array (over ++ s) dtype
  f [] x' = x'
  f (d :: ds) x' = replicate d (f ds x')

export
cast_dtype : Cast dtype dtype' => Tensor shape dtype -> Tensor shape dtype'
cast_dtype tensor = map cast tensor

export
foldr1 : (dtype -> dtype -> dtype) -> {leading : Shape {rank = axis}}
  -> (axis : Nat) -> Tensor (leading ++ _ :: tail) dtype -> Tensor (leading ++ tail) dtype

----------------------------- numeric operations ----------------------------

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

export partial
(@@) : Num dtype => Tensor [m, S n] dtype -> Tensor (S n :: ps) dtype -> Tensor (m :: ps) dtype
(@@) (MkTensor x) (MkTensor y) = MkTensor $ matmul' x y where
  partial
  matmul' : Num ty => Array [m', S n] ty -> Array (S n :: ps) ty -> Array (m' :: ps) ty
  matmul' [] _ = []
  matmul' (x :: xs) y = (dot x y) :: (matmul' xs y) where
    partial
    dot : Num ty => Vect (S n) ty -> Array (S n :: ps) ty -> Array ps ty
    dot {ps = []} x y = sum $ zipWith (*) x y
    dot {ps = (_ :: [])} x y = foldr1 (zipWith (+)) $ zipWith (\x_elem, y_row => map (* x_elem) y_row) x y
    dot {ps = (_ :: _ :: [])} x y = ?r

public export implicit doubleToScalar : Double -> Tensor [] Double
doubleToScalar = MkTensor

export
(+) : Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) t1 t2 = zipWith (+) t1 t2

export
(-) : Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) t1 t2 = zipWith (-) t1 t2

export
(*) : Num dtype => Tensor [] dtype -> Tensor shape dtype -> Tensor shape dtype
(*) (MkTensor x) t = map (* x) t

||| floating point division. we don't support integer division
export
(/) : Fractional dtype => Tensor shape dtype -> Tensor [] dtype -> Tensor shape dtype
(/) t (MkTensor x) = map (/ x) t

export
exp_ew : Tensor shape Double -> Tensor shape Double
exp_ew x = map exp x

-- remember, we're only writing functions that exist on tensors. For example,
--  exp and pow are wrt to tensor multiplication, not elementwise

export partial
det : Neg dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
det {n = Z} (MkTensor [[a]]) = MkTensor a
det {n = S Z} (MkTensor [[a, b], [c, d]]) = MkTensor $ a * d - b * c
det {n = S (S Z)} (MkTensor [[a, b, c], [d, e, f], [g, h, i]]) = MkTensor $ a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

partial
adjugate : Neg dtype => Tensor [S n, S n] dtype -> Tensor [S n, S n] dtype
adjugate {n = Z} (MkTensor [[a]]) = ?r
adjugate {n = (S Z)} (MkTensor [[a, b], [c, d]]) = MkTensor [[d, -c], [-b, a]]
adjugate {n = (S (S Z))} (MkTensor [[a, b, c], [d, e, f], [g, h, i]]) = MkTensor [
    [e * i - f * h, - (d * i - f * g), d * h - e * g],
    [- (b * i - c * h), a * i - c * g, - (a * h - b * g)],
    [b * f - c * e, - (a * f - c * d), a * e - b * d]
  ]

export partial
inverse : Tensor [S n, S n] Double -> Maybe (Tensor [S n, S n] Double)
inverse x = let det_@(MkTensor d) = det x in if (d == 0.0) then Nothing else Just $ (adjugate x) / det_

-- todo should taking the diag or det return a dtype or a Tensor [] dtype?
--  does the signature of index tell us it must be Tensor [] dtype?
--  i guess it would be inconvenient to have to re-wrap it into a Tensor for
--  further calculations
export
diag : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
diag (MkTensor x) = MkTensor $ product $ diag x

---------------------------- broadcasting ---------------------------------

-- see https://www.tensorflow.org/xla/broadcasting
|||
||| can add arbitrary dimensions to the head of the shape, but not to the tail
||| [3] -> [3, 3]
||| [] -> [2, 4, 7]
|||
||| can multiply any 1
||| [1, 3] -> [3, 3]
||| [3, 1] -> [3, 3]
|||
-- todo it may be miles simpler to say that you can only broadcast by the second
--  rule, then provide a way to add 1s to the head of the from shape, which
--  would then enable the second route, so rather than broadcast t, it would be
--  broadcast (expand t)
public export
data DimBroadcastable : Nat -> Nat -> Type where
  Eq : l = r -> DimBroadcastable l r
  One : l = 1 -> (r : Nat) -> DimBroadcastable l r

public export
data Broadcastable : (from : Shape {rank}) -> (to : Shape {rank}) -> Type where
  Empty : Broadcastable [] []  -- todo can this be Broadcastable x x?
  Extend : DimBroadcastable fd td -> Broadcastable f t -> Broadcastable (fd :: f) (td :: t)

cannot_broadcast_for_incompatible_heads : ((d = 1) -> Void) -> ((d = d') -> Void) -> Broadcastable (d :: ds) (d' :: ds') -> Void
cannot_broadcast_for_incompatible_heads d_neq_1 d_neq_d' (Extend (Eq prf) _) = d_neq_d' prf
cannot_broadcast_for_incompatible_heads d_neq_1 d_neq_d' (Extend (One prf _) _) = d_neq_1 prf

cannot_broadcast_for_incompatible_tails : (Broadcastable ds ds' -> Void) -> Broadcastable (d :: ds) (d' :: ds') -> Void
cannot_broadcast_for_incompatible_tails tails_are_not_broadcastable (Extend _ tails_are_broadcastable) = tails_are_not_broadcastable tails_are_broadcastable

export
is_broadcastable : Tensor from dtype -> Tensor to dtype -> Dec (Broadcastable from to)
is_broadcastable {from} {to} _ _ = is_broadcastable' from to where
  is_broadcastable' : (from : Shape) -> (to : Shape) -> Dec (Broadcastable from to)
  is_broadcastable' [] [] = Yes Empty
  is_broadcastable' (d :: ds) (d' :: ds') with (is_broadcastable' ds ds')
    | Yes tails_are_broadcastable with (decEq d d')
      | Yes d_eq_d' = Yes $ Extend (Eq d_eq_d') tails_are_broadcastable
      | No d_neq_d' with (decEq d 1)
        | Yes d_eq_1 = Yes $ Extend (One d_eq_1 d') tails_are_broadcastable
        | No d_neq_1 = No $ cannot_broadcast_for_incompatible_heads d_neq_1 d_neq_d'
    | No tails_are_not_broadcastable = No (cannot_broadcast_for_incompatible_tails tails_are_not_broadcastable)

export partial
broadcast : Tensor from dtype -> {auto prf : Broadcastable from to} -> Tensor to dtype
broadcast {from} {to} {dtype} {prf} (MkTensor x) = MkTensor (broadcast' from to x) where
  partial
  broadcast' : (from : Shape) -> (to : Shape) -> {auto prf_ : Broadcastable from to} -> Array from dtype -> Array to dtype
  broadcast' {prf_ = Empty} [] [] x = x
  broadcast' {prf_ = (Extend (Eq Refl) _)} (d :: ds) (d :: dds) x = map (broadcast' ds dds) x
  broadcast' {prf_ = (Extend (One Refl dd) _)} ((S Z) :: ds) (dd :: dds) x = replicate dd (head (map (broadcast' ds dds) x))

public export
data Nestable : (from : Shape) -> (to : Shape) -> Type where
  Same : Nestable f f  -- todo why does this break if I call it Eq?
  AddOne : Nestable f t -> Nestable f (1 :: t)

export
is_nestable : Tensor from dtype -> Tensor to dtype -> Dec (Nestable from to)
is_nestable {from} {to} _ _ = is_nestable' from to where
  is_nestable' : (from : Shape) -> (to : Shape) -> Dec (Nestable from to)

export partial
pad : {auto prf : Nestable from to} -> Tensor from dtype -> Tensor to dtype
pad (MkTensor x) = MkTensor (pad' x) where
  pad' : {auto prf' : Nestable from to} -> Array from dtype -> Array to dtype
  pad' {prf' = Same} x = x
  pad' {prf' = (AddOne y)} x = [pad' x]
