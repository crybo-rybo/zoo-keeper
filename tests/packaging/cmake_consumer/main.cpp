#include <zoo/zoo.hpp>

int main() {
    auto create_fn = &zoo::Agent::create;
    auto load_fn = &zoo::core::Model::load;
    return (create_fn && load_fn) ? 0 : 1;
}
