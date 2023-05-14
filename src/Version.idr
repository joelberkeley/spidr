module Version

export
data Version = MkVersion Nat Nat Nat

export
Show Version where
  MkVersion major minor patch = "\{show major}.\{show minor}.\{show patch}"

export
C_XLA_EXT : Version
C_XLA_EXT = MkVersion 0 0 1

export
XLA_EXT : Version
XLA_EXT = MkVersion 0 3 0
