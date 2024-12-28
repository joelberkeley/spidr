cc_library(
    name = "service",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/service",
        "//src/xla",
        "//src",
    ],
    visibility = ["//visibility:public"],
)
