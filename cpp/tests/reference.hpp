#pragma once

#include "gguf_file.hpp"
#include "muscriptor/model.hpp"
#include "run_config.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace msl::test
{

/**
 * Per-tensor pass thresholds, calibrated by the dumper from the measured
 * fp32<->fp16 gap of the reference itself (see msl/dump_refs.py).
 */
struct Tolerance {
    double atol = 0.0;
    double cosine_min = 1.0;
};

/** How a computed tensor compares to its reference. */
struct Comparison {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rms = 0.0;
    double cosine = 1.0;
    std::size_t worst_index = 0;
    double worst_got = 0.0;
    double worst_want = 0.0;
    std::size_t count = 0;
    // Set when the two disagree about which entries are +/-inf or NaN, which
    // is a structural mismatch no tolerance should ever excuse.
    bool nonfinite_mismatch = false;
};

/** @return How `got` compares to `want`, one tensor's worth of statistics. */
Comparison compare(const std::vector<float>& got, const std::vector<float>& want);

/** @return A human-readable report of `c` against `t`, for a tensor named `name`. */
std::string describe(const std::string& name, const Comparison& c, const Tolerance& t);

/**
 * Appends one line for `name` to the parity report when `--parity-report` was
 * given: the comparison's distance from each bound as a ratio, 1.0 being the
 * tolerance itself. Does nothing otherwise.
 */
void recordParity(const std::string& name, const Comparison& c, const Tolerance& t);
void setParityReportPath(const std::filesystem::path& path);

/**
 * The dumped PyTorch reference: tensors from the GGUF, everything else from
 * manifest.json.
 */
class Reference
{
public:
    /**
     * The fp32 dump for the current configuration. Shared across test cases
     * so the dump is read once per configuration, not once per test.
     */
    static const Reference& fp32();

    const std::vector<float>& tensor(const std::string& name) const;
    bool has(const std::string& name) const;
    std::vector<std::int64_t> shape(const std::string& name) const;
    Tolerance tolerance(const std::string& name) const;

    const std::vector<std::int32_t>& tokens(int chunk) const;
    int num_chunks() const { return num_chunks_; }

    /** @return Decode steps dumped in full (step 0 is the prefill). */
    int max_decode_steps() const { return max_decode_steps_; }
    std::int32_t decode_argmax(int step) const;

    std::int32_t eos_id() const { return eos_id_; }
    std::int32_t initial_token_id() const { return initial_token_id_; }
    int n_layer() const { return n_layer_; }
    int prepend_length() const { return prepend_length_; }
    int mel_frames() const { return mel_frames_; }
    int segment_samples() const { return segment_samples_; }

    const std::filesystem::path& weights_path() const { return weights_path_; }

    ~Reference();

private:
    explicit Reference(const std::string& precision);

    std::filesystem::path weights_path_;
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<GgufFile> file_;
    mutable std::map<std::string, std::vector<float>> cache_;

    std::map<std::string, Tolerance> tolerances_;
    std::vector<std::vector<std::int32_t>> tokens_;
    std::map<int, std::int32_t> decode_argmax_;

    std::string size_;
    std::int32_t eos_id_ = 0;
    std::int32_t initial_token_id_ = 0;
    int n_layer_ = 0;
    int prepend_length_ = 0;
    int mel_frames_ = 0;
    int segment_samples_ = 0;
    int num_chunks_ = 0;
    int max_decode_steps_ = 0;
};

/**
 * Whether the current configuration runs the ladder on the GPU (`--device gpu`).
 *
 * The two backends are not bit-identical, so the device is part of the
 * configuration rather than something the library picks.
 */
bool gpu_enabled();

/**
 * The converted weights for the current configuration, loaded once and
 * reused. Tests that mutate KV state call `reset()`; the cache is the only
 * mutable state a Model carries.
 */
Model& shared_model();

} // namespace msl::test
