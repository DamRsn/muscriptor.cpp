#pragma once

#include "muscriptor/note.hpp"
#include "muscriptor/transcriber.hpp"

#include <optional>
#include <span>
#include <vector>

namespace msl
{

/**
 * Seconds. The model can place an onset this far before the sound it belongs
 * to, so an onset that close to the end of a silence is not taken as decoded
 * from it.
 */
inline constexpr double ONSET_LEAD = 0.05;

/**
 * Where a signal is silent, and what that does to the notes decoded there.
 *
 * Times are seconds in the signal. Everything before it and after it counts as
 * silent, so the first and last regions may reach to minus or plus infinity.
 */
class SilenceGate
{
public:
    /** A gate with no silent regions: changes nothing. */
    SilenceGate() = default;

    SilenceGate(std::span<const float> inSamples, const SilenceGateOptions& inOptions);

    /** @return Whether [inStart, inEnd) lies in one region, after its release. */
    bool covers(double inStart, double inEnd) const;

    /**
     * @return The note with its offset cut where the input goes silent, or
     *         nothing if it starts in silence.
     */
    std::optional<Note> apply(Note inNote) const;

    /**
     * @return The earliest offset `apply` can give a note starting at
     *         `inOnset`, or infinity if it drops or never cuts such a note.
     */
    double cutPoint(double inOnset) const;

private:
    /** @return Whether a note starting at `inOnset` started in silence. */
    bool _drops(double inOnset) const;

    struct Region {
        double start = 0.0;
        double end = 0.0;
    };

    // Sorted, disjoint.
    std::vector<Region> mRegions;
    double mRelease = 0.0;
};

} // namespace msl
