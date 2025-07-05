cc_library(
    name = "mlir",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["**/*.cpp"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "//src",
        "//src/llvm",
    ],
    visibility = ["//visibility:public"],
)
