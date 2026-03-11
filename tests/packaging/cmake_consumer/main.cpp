#include <zoo/core/json.hpp>
#include <zoo/zoo.hpp>

int main() {
    nlohmann::json json = {{"model_path", "model.gguf"}};
    zoo::Config config = json.get<zoo::Config>();

    auto create_fn = &zoo::Agent::create;
    auto load_fn = &zoo::core::Model::load;
    return (create_fn && load_fn && !config.model_path.empty()) ? 0 : 1;
}
