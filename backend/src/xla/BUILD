cc_library(
    name = "xla",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal",
    ],
    visibility = ["//visibility:public"],
)
