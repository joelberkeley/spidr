cc_library(
    name = "IR",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@llvm-project//mlir:IR",
        "//src",
    ],
    visibility = ["//visibility:public"],
)
