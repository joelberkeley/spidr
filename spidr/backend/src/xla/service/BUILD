cc_library(
    name = "service",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/service:platform_util",
        "//src:src",
    ],
    visibility = ["//visibility:public"],
)
