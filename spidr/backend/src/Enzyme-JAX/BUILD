cc_library(
    name = "Enzyme-JAX",
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    srcs = glob(["**/*.cpp"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@enzyme//:EnzymeMLIR",
        "@enzyme-jax//src/enzyme_ad/jax:XLADerivatives",
        "@llvm-project//mlir:AllPassesAndDialects",
        "//src/mlir",
    ],
)
