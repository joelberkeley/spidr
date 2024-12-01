cc_library(
    name = "translate",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/hlo/translate:portable_api",
        "//src/xla/hlo/ir",
        "//src",
    ],
    visibility = ["//visibility:public"],
)
