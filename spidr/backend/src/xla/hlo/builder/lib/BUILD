cc_library(
    name = "lib",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/hlo/builder/lib:math",
        "@xla//xla/hlo/builder/lib:matrix",
        "@xla//xla/hlo/builder/lib:prng",
        "//src/xla",
        "//src/xla/hlo/builder",
    ],
    visibility = ["//visibility:public"],
)
