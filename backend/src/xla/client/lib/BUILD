cc_library(
    name = "lib",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/client:client",
        "//src/xla:xla",
        "//src/xla/client:client",
    ],
    visibility = ["//visibility:public"],
)
