# while it seems counterintuitive to build a shared library with a cc_binary rule when
# there's a cc_library rule, the docs explain how cc_binary with linkshared = True
# produce a shared library:
#
# > it is assumed that shared libraries built with a cc_binary rule are only
# > loaded manually by other programs
#
# see also this SO post https://stackoverflow.com/a/51689102/5986907

cc_binary(
    name = "c_xla",
    linkshared = True,
    linkstatic = True,
    # copts=["-dynamiclib"],
    srcs = [
        "//src/xla",
        "//src/xla/client",
        "//src/xla/client/lib",
        "//src/xla/pjrt",
        "//src/xla/pjrt/c",
        "//src",
    ],
    deps = [
        "//src/xla",
        "//src/xla/client",
        "//src/xla/client/lib",
        "//src/xla/pjrt",
        "//src/xla/pjrt/c",
        "//src",
    ],
)
