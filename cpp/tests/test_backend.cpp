// Backend selection and the graph node capacity: both fail silently, or abort,
// if they regress.

#include "reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

using namespace msl;
using namespace msl::test;

TEST_CASE("the backend is the one that was asked for", "[backend]")
{
    const std::string_view name = shared_model().backendName();
    INFO("backendName() reports: " << name);

    if (!gpu_enabled()) {
        // An explicit opt-out has one legal answer, on every build and machine.
        CHECK(name == "CPU");
        return;
    }

    // With a GPU backend built, --device gpu must not fall back to the CPU, or
    // the GPU ladder would pass without testing the GPU.
#if defined(MUSCRIPTOR_HAS_METAL)
    CHECK(name == "Metal");
#elif defined(MUSCRIPTOR_HAS_VULKAN)
    CHECK(name == "Vulkan");
#else
    // No GPU backend in the build. Asking for one must degrade, not fail.
    CHECK(name == "CPU");
#endif
}

TEST_CASE("the graph fits its capacity with room to spare", "[backend]")
{
    Model& model = shared_model();
    const Hparams& hp = model.hparams();

    // Node count is a function of the layer count alone -- neither the number
    // of conditioning frames nor the number of tokens adds nodes -- so a
    // one-frame prefill traces exactly the same graph shape as a real one and
    // costs milliseconds instead of seconds.
    model.reset();
    const std::vector<float> cond(static_cast<std::size_t>(hp.dim), 0.0f);
    const std::array<std::int32_t, 1> prompt {hp.initial_token_id};
    model.prefill(cond, 1, prompt);

    const int prefill_nodes = model.graphNodeCount();
    const int capacity = model.graphNodeCapacity();
    INFO("prefill graph: " << prefill_nodes << " nodes, capacity " << capacity);
    CHECK(prefill_nodes > 0);
    CHECK(prefill_nodes <= capacity);

    model.decode(hp.initial_token_id);
    const int decode_nodes = model.graphNodeCount();
    INFO("decode graph: " << decode_nodes << " nodes, capacity " << capacity);
    CHECK(decode_nodes > 0);
    CHECK(decode_nodes <= capacity);

    // Overrunning the capacity is a GGML_ASSERT abort, so fail while there is
    // still 2x headroom.
    CHECK(std::max(prefill_nodes, decode_nodes) * 2 <= capacity);

    model.reset();
}

TEST_CASE("the default thread count resolves to something usable", "[backend]")
{
    // The default is the performance-core count, which is not discoverable
    // portably, so this pins the property that matters rather than the number:
    // it is always set, always positive, and never more than the machine has.
    Model& model = shared_model();
    const int logical = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));

    model.setNumThreads(0);
    INFO("default resolved to " << model.numThreads() << " of " << logical << " logical cores");
    CHECK(model.numThreads() >= 1);
    CHECK(model.numThreads() <= logical);

    model.setNumThreads(3);
    CHECK(model.numThreads() == 3);

    // Restore, so this test cannot change what the timing-sensitive parts of
    // the ladder run with.
    model.setNumThreads(0);
}
