#include <zoo/core/json.hpp>
#include <zoo/zoo.hpp>

int main() {
    nlohmann::json json = {{"model_path", "model.gguf"}};
    zoo::ModelConfig config = json.get<zoo::ModelConfig>();

    using AgentCreateFn = zoo::Expected<std::unique_ptr<zoo::Agent>> (*)(
        const zoo::ModelConfig&, const zoo::AgentConfig&, const zoo::GenerationOptions&);
    using ModelLoadFn = zoo::Expected<std::unique_ptr<zoo::core::Model>> (*)(
        const zoo::ModelConfig&, const zoo::GenerationOptions&);

    AgentCreateFn create_fn = &zoo::Agent::create;
    ModelLoadFn load_fn = &zoo::core::Model::load;
    return (create_fn && load_fn && !config.model_path.empty()) ? 0 : 1;
}
