#include <zoo/core/json.hpp>
#include <zoo/zoo.hpp>

int main() {
    nlohmann::json json = {{"model_path", "model.gguf"}};
    zoo::ModelConfig config = json.get<zoo::ModelConfig>();

    using ModelLoadFn = zoo::Expected<std::unique_ptr<zoo::Model>> (*)(
        const zoo::ModelConfig&, const zoo::GenerationOptions&);

    ModelLoadFn load_fn = &zoo::Model::load;
    return (load_fn && !config.model_path.empty()) ? 0 : 1;
}
