from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            "fse_ops._C",
            sources=["src/fse_ops/ops.cpp"],
            extra_compile_args=["-O2", "-g0"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
