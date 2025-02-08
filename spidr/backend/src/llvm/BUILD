cc_library(
    name = "llvm",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["**/*.cpp"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@llvm-project//llvm:Support",
        "//src",
    ],
    visibility = ["//visibility:public"],
)
