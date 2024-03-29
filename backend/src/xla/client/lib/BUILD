cc_library(
    name = "lib",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + [
        "@xla//xla/client/lib:math",
        "@xla//xla/client/lib:matrix",
        "@xla//xla/client/lib:prng",
    ],
    hdrs = glob(["*.h"]),
    deps = [
        "//src/xla:xla",
        "//src/xla/client:client",
    ],
    visibility = ["//visibility:public"],
)
