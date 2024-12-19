cc_library(
    name = "Pass",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@llvm-project//mlir:Pass",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
