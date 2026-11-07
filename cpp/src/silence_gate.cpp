#include "silence_gate.hpp"

#include "vocabulary.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace msl
{

namespace
{

    constexpr double INF = std::numeric_limits<double>::infinity();

    constexpr std::size_t FRAME_SAMPLES = Transcriber::SAMPLE_RATE / Vocabulary::FRAME_RATE;

    /** @return Whether each 10 ms frame is below `inThresholdDbfs`. */
    std::vector<bool> silentFrames(std::span<const float> inSamples, float inThresholdDbfs)
    {
        const double max_mean_square = std::pow(10.0, static_cast<double>(inThresholdDbfs) / 10.0);
        std::vector<bool> silent;

        for (std::size_t start = 0; start < inSamples.size(); start += FRAME_SAMPLES) {
            const std::span<const float> frame =
                inSamples.subspan(start, std::min(FRAME_SAMPLES, inSamples.size() - start));

            double sum = 0.0;

            for (const float sample: frame) {
                sum += static_cast<double>(sample) * sample;
            }

            silent.push_back(sum / static_cast<double>(frame.size()) < max_mean_square);
        }

        return silent;
    }

} // namespace

SilenceGate::SilenceGate(std::span<const float> inSamples, const SilenceGateOptions& inOptions)
    : mRelease(inOptions.release)
{
    const std::vector<bool> silent = silentFrames(inSamples, inOptions.threshold_dbfs);
    const std::size_t n_frames = silent.size();
    const double frame = 1.0 / Vocabulary::FRAME_RATE;

    if (n_frames == 0) {
        mRegions.push_back({-INF, INF});
        return;
    }

    if (!silent.front()) {
        mRegions.push_back({-INF, 0.0});
    }

    for (std::size_t first = 0; first < n_frames;) {
        if (!silent[first]) {
            ++first;
            continue;
        }

        std::size_t last = first;

        while (last < n_frames && silent[last]) {
            ++last;
        }

        const bool at_start = first == 0;
        const bool at_end = last == n_frames;

        // A run shorter than the release would only cut notes after the sound is back.
        const double min_length = std::max(inOptions.min_duration, inOptions.release);

        if (at_start || at_end || static_cast<double>(last - first) * frame >= min_length) {
            mRegions.push_back({at_start ? -INF : static_cast<double>(first) * frame,
                                at_end ? INF : static_cast<double>(last) * frame});
        }

        first = last;
    }

    if (!silent.back()) {
        mRegions.push_back({static_cast<double>(inSamples.size()) / Transcriber::SAMPLE_RATE, INF});
    }
}

bool SilenceGate::covers(double inStart, double inEnd) const
{
    return std::any_of(mRegions.begin(), mRegions.end(), [&](const Region& inRegion) {
        return inRegion.start + mRelease <= inStart && inEnd <= inRegion.end;
    });
}

std::optional<Note> SilenceGate::apply(Note inNote) const
{
    if (_drops(inNote.onset)) {
        return std::nullopt;
    }

    inNote.offset = std::min(inNote.offset, cutPoint(inNote.onset));
    return inNote;
}

double SilenceGate::cutPoint(double inOnset) const
{
    if (_drops(inOnset)) {
        return INF;
    }

    for (const Region& region: mRegions) {
        if (region.start + mRelease > inOnset) {
            return std::max(region.start + mRelease, inOnset + MINIMUM_NOTE_DURATION_SECONDS);
        }
    }

    return INF;
}

bool SilenceGate::_drops(double inOnset) const
{
    return std::any_of(mRegions.begin(), mRegions.end(), [&](const Region& inRegion) {
        return inOnset >= inRegion.start + mRelease && inOnset < inRegion.end - ONSET_LEAD;
    });
}

} // namespace msl
