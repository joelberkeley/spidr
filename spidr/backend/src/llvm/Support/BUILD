cc_library(
    name = "Support",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@llvm-project//llvm:Support",
        "//src",
    ],
    visibility = ["//visibility:public"],
)
