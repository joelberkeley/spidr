cc_library(
    name = "Enzyme",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["**/*.cpp"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//mlir:IR",
        "//src/mlir",
    ],
    visibility = ["//visibility:public"],
)
