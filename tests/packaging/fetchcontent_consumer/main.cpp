#include <zoo/zoo.hpp>

int main() {
    zoo::ModelConfig config{};
    config.model_path = "dummy.gguf";
    return config.model_path.empty() ? 1 : 0;
}
