cc_library(
    name = "xla",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + [
        "@xla//xla:literal",
    ],
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)
