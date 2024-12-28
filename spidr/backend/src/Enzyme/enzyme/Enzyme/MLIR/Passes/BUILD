cc_library(
    name = "Passes",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@enzyme//:EnzymeMLIR",
        "//src/mlir/Pass",
    ],
    visibility = ["//visibility:public"],
)
