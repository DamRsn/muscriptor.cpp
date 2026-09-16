#include "reference.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace msl::test
{

namespace
{

    nlohmann::json read_manifest(const std::filesystem::path& path, const RunConfig& config)
    {
        std::ifstream stream(path);

        if (!stream) {
            throw std::runtime_error(msl::format(
                "reference manifest not found at {}.\nGenerate it with: {}", path.string(), config.refsHint()));
        }

        return nlohmann::json::parse(stream);
    }

} // namespace

Comparison compare(const std::vector<float>& got, const std::vector<float>& want)
{
    Comparison c;

    if (got.size() != want.size()) {
        throw std::runtime_error(
            msl::format("size mismatch: got {} values, reference has {}", got.size(), want.size()));
    }

    double dot = 0.0, norm_got = 0.0, norm_want = 0.0, sum_sq = 0.0, scale = 0.0;

    for (std::size_t i = 0; i < want.size(); ++i) {
        const double a = want[i];
        const double b = got[i];

        // Reserved logits are legitimately -inf in both; they must agree
        // exactly and are excluded from the numeric statistics.
        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (!(std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))) {
                c.nonfinite_mismatch = true;
            }

            continue;
        }

        const double diff = std::abs(a - b);

        if (diff > c.max_abs) {
            c.max_abs = diff;
            c.worst_index = i;
            c.worst_got = b;
            c.worst_want = a;
        }

        scale = std::max(scale, std::abs(a));
        sum_sq += (a - b) * (a - b);
        dot += a * b;
        norm_got += b * b;
        norm_want += a * a;
        ++c.count;
    }

    if (c.count > 0) {
        c.rms = std::sqrt(sum_sq / static_cast<double>(c.count));
        c.max_rel = c.max_abs / std::max(scale, 1e-12);
        const double denom = std::sqrt(norm_got) * std::sqrt(norm_want);
        c.cosine = denom > 0.0 ? dot / denom : 1.0;
    }

    return c;
}

std::string describe(const std::string& name, const Comparison& c, const Tolerance& t)
{
    return msl::format("{}\n"
                       "  max_abs {}  (tolerance {})\n"
                       "  max_rel {}\n"
                       "  rms     {}\n"
                       "  cosine  {}  (minimum {})\n"
                       "  worst at index {}: got {}, reference {}\n"
                       "  compared {} finite values{}",
                       name,
                       formatDouble("%.6e", c.max_abs),
                       formatDouble("%.6e", t.atol),
                       formatDouble("%.6e", c.max_rel),
                       formatDouble("%.6e", c.rms),
                       formatDouble("%.12f", c.cosine),
                       formatDouble("%.12f", t.cosine_min),
                       c.worst_index,
                       formatDouble("%.9g", c.worst_got),
                       formatDouble("%.9g", c.worst_want),
                       c.count,
                       c.nonfinite_mismatch ? "\n  ERROR: infinite/NaN entries do not line up" : "");
}

namespace
{
    std::filesystem::path& parityReportPath()
    {
        static std::filesystem::path path;
        return path;
    }
} // namespace

void setParityReportPath(const std::filesystem::path& path)
{
    parityReportPath() = path;
    std::ofstream(path, std::ios::trunc);
}

void recordParity(const std::string& name, const Comparison& c, const Tolerance& t)
{
    const std::filesystem::path& path = parityReportPath();

    if (path.empty()) {
        return;
    }

    // Ratios rather than raw numbers: a tolerance is per tensor, so 1.9 over
    // budget on step 3 and 0.1 under on layer 12 compare directly.
    const double cosine_budget = 1.0 - t.cosine_min;
    const nlohmann::json line = {
        {"config", currentConfig().label()},
        {"tensor", name},
        {"max_abs", c.max_abs},
        {"atol", t.atol},
        {"max_abs_ratio", t.atol > 0.0 ? c.max_abs / t.atol : 0.0},
        {"cosine_deficit", 1.0 - c.cosine},
        {"cosine_budget", cosine_budget},
        {"cosine_ratio", cosine_budget > 0.0 ? (1.0 - c.cosine) / cosine_budget : 0.0},
        {"nonfinite_mismatch", c.nonfinite_mismatch},
    };

    std::ofstream(path, std::ios::app) << line.dump() << '\n';
}

// ---------------------------------------------------------------------------

Reference::Reference(const std::string& precision)
{
    const RunConfig& config = currentConfig();
    const std::filesystem::path refs = config.refsDir();
    const nlohmann::json manifest = read_manifest(refs / "manifest.json", config);

    size_ = manifest["model"]["size"].get<std::string>();

    if (size_ != config.size) {
        throw std::runtime_error(msl::format("{} was dumped for '{}' but sits in the '{}' directory",
                                             (refs / "manifest.json").string(),
                                             size_,
                                             config.size));
    }
    eos_id_ = manifest["model"]["eos_id"].get<std::int32_t>();
    initial_token_id_ = manifest["model"]["initial_token_id"].get<std::int32_t>();
    n_layer_ = manifest["model"]["num_layers"].get<int>();
    prepend_length_ = manifest["prefix"]["prepend_length"].get<int>();
    mel_frames_ = manifest["prefix"]["self_wav_frames"].get<int>();
    segment_samples_ = manifest["fixture"]["segment_samples"].get<int>();
    num_chunks_ = manifest["generation"]["num_chunks"].get<int>();
    max_decode_steps_ = manifest["generation"]["max_decode_steps_dumped"].get<int>();

    for (const auto& [name, value]: manifest["tolerances"].items()) {
        tolerances_[name] = Tolerance {value["atol"].get<double>(), value["cosine_min"].get<double>()};
    }

    const auto& tokens = manifest["tokens"][precision];
    tokens_.resize(static_cast<std::size_t>(num_chunks_));

    for (int chunk = 0; chunk < num_chunks_; ++chunk) {
        tokens_[static_cast<std::size_t>(chunk)] =
            tokens[msl::format("chunk{}", chunk)].get<std::vector<std::int32_t>>();
    }

    for (const auto& [step_name, value]: manifest["decode_argmax"][precision].items()) {
        const int step = std::stoi(step_name.substr(std::string("dec.step").size()));
        decode_argmax_[step] = value.get<std::int32_t>();
    }

    weights_path_ = config.weightsPath();

    if (!std::filesystem::exists(weights_path_)) {
        throw std::runtime_error(
            msl::format("weights not found at {}.\nGet them with: {}", weights_path_.string(), config.weightsHint()));
    }

    backend_ = ggml_backend_cpu_init();
    file_ = std::make_unique<GgufFile>(refs / manifest["references"][precision].get<std::string>(), backend_);
}

Reference::~Reference()
{
    file_.reset();

    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

const Reference& Reference::fp32()
{
    static PerConfig<Reference> instance;
    return instance.get([] { return std::unique_ptr<Reference>(new Reference("fp32")); });
}

bool Reference::has(const std::string& name) const
{
    return file_->find(name) != nullptr;
}

const std::vector<float>& Reference::tensor(const std::string& name) const
{
    const auto it = cache_.find(name);

    if (it != cache_.end()) {
        return it->second;
    }

    return cache_.emplace(name, msl::tensorToFloat(file_->get(name))).first->second;
}

std::vector<std::int64_t> Reference::shape(const std::string& name) const
{
    const ggml_tensor* tensor = file_->get(name);
    std::vector<std::int64_t> ne(tensor->ne, tensor->ne + GGML_MAX_DIMS);

    while (ne.size() > 1 && ne.back() == 1) {
        ne.pop_back();
    }

    return ne;
}

Tolerance Reference::tolerance(const std::string& name) const
{
    const auto it = tolerances_.find(name);

    if (it == tolerances_.end()) {
        throw std::runtime_error(msl::format("no calibrated tolerance for tensor '{}'", name));
    }

    return it->second;
}

const std::vector<std::int32_t>& Reference::tokens(int chunk) const
{
    return tokens_.at(static_cast<std::size_t>(chunk));
}

std::int32_t Reference::decode_argmax(int step) const
{
    const auto it = decode_argmax_.find(step);

    if (it == decode_argmax_.end()) {
        throw std::runtime_error(msl::format("no dumped argmax for decode step {}", step));
    }

    return it->second;
}

bool gpu_enabled()
{
    return currentConfig().device == Device::gpu;
}

Model& shared_model()
{
    static PerConfig<Model> model;
    return model.get([] {
        const Reference& ref = Reference::fp32();
        Model::Options options;
        options.use_gpu = gpu_enabled();
        return std::make_unique<Model>(Model::load(ref.weights_path(), options));
    });
}

} // namespace msl::test
