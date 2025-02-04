cc_library(
    name = "Passes",
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@enzyme//:EnzymeMLIR",
        "@enzyme-jax//src/enzyme_ad/jax:XLADerivatives",
        "@llvm-project//mlir:AllPassesAndDialects",
        "//src/mlir/IR",
        "//src/mlir/Pass",
    ],
)
