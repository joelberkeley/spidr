cc_library(
    name = "xla",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + ["@com_google_absl//absl/types:span"],
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla:literal",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/base:log_severity",
    ],
    visibility = ["//visibility:public"],
)
