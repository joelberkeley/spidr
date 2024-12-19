cc_library(
    name = "ir",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/hlo/ir:hlo",
        "//src/xla/service",
    ],
    visibility = ["//visibility:public"],
)
