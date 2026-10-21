#include <torch/extension.h>

torch::Tensor leaky_relu_cpu(const torch::Tensor& x) {
    TORCH_CHECK(x.device().is_cpu(), "leaky_relu expects a CPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "leaky_relu expects a float32 tensor");
    TORCH_CHECK(x.layout() == torch::kStrided,
                "leaky_relu expects a strided tensor");

    auto flat = x.resolve_neg().reshape({-1});
    auto result = torch::empty_like(flat);

    auto input = flat.accessor<float, 1>();
    auto output = result.accessor<float, 1>();

    for (int64_t i = 0; i < flat.numel(); ++i) {
        output[i] = input[i] > 0 ? input[i] : 0.1f * input[i];
    }

    return result.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_cpu);
}
