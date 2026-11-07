#include "resume_point.hpp"

#include "muscriptor/error.hpp"

#include "vocabulary.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>

namespace msl
{

namespace
{

    constexpr std::string_view MAGIC = "muscriptor-resume";
    constexpr int FORMAT_VERSION = 2;

    constexpr int MAX_VALUE = std::numeric_limits<int>::max();
    constexpr int MAX_PITCH = Vocabulary::PITCH_COUNT - 1;
    constexpr int MAX_PROGRAM = Vocabulary::PROGRAM_COUNT - 1;

    /** The unsigned integer with the same bits as `T`. */
    template <std::floating_point T>
    using Bits = std::conditional_t<sizeof(T) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;

    /**
     * Settings are stored as their bit pattern: exact, and independent of
     * locale and of floating-point `to_chars`, which macOS 11 lacks.
     */
    template <std::floating_point T>
    Bits<T> toBits(T inValue)
    {
        return std::bit_cast<Bits<T>>(inValue);
    }

    /** Times are stored as whole frames, so no float goes through text. */
    double fromFrames(int inFrames)
    {
        return static_cast<double>(inFrames) / Vocabulary::FRAME_RATE;
    }

    int toFrames(double inSeconds)
    {
        const int frames = static_cast<int>(std::lround(inSeconds * Vocabulary::FRAME_RATE));

        // Exact on purpose: a restored time has to be the very double decoding produced.
        if (fromFrames(frames) != inSeconds) {
            throw Exception(Error::Internal, "resume point time is not on the frame grid");
        }

        return frames;
    }

    /** Whitespace-separated words. Any mismatch marks it failed for good. */
    class Reader
    {
    public:
        explicit Reader(std::string_view inText)
            : mText(inText)
        {
        }

        std::string_view word()
        {
            const std::size_t start = mText.find_first_not_of(WHITESPACE);

            if (start == std::string_view::npos) {
                mFailed = true;
                return {};
            }

            const std::size_t end = std::min(mText.find_first_of(WHITESPACE, start), mText.size());
            const std::string_view word = mText.substr(start, end - start);
            mText.remove_prefix(end);
            return word;
        }

        void expect(std::string_view inWord)
        {
            if (word() != inWord) {
                mFailed = true;
            }
        }

        template <typename T>
        T number(T inMin, T inMax)
        {
            const std::string_view text = word();
            T value = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);

            if (error != std::errc {} || end != text.data() + text.size() || value < inMin || value > inMax) {
                mFailed = true;
                return inMin;
            }

            return value;
        }

        /** Reads a value `toBits` wrote. */
        template <std::floating_point T>
        T bits()
        {
            return std::bit_cast<T>(number<Bits<T>>(0, std::numeric_limits<Bits<T>>::max()));
        }

        bool failed() const { return mFailed; }

        bool atEnd() const { return mText.find_first_not_of(WHITESPACE) == std::string_view::npos; }

    private:
        static constexpr std::string_view WHITESPACE = " \t\r\n";

        std::string_view mText;
        bool mFailed = false;
    };

} // namespace

std::string serializeResumeState(const ResumeState& inState)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());

    out << MAGIC << ' ' << FORMAT_VERSION << '\n';
    out << "samples " << inState.n_samples << '\n';
    out << "prelude " << (inState.prelude_forcing ? 1 : 0) << '\n';
    out << "instruments " << inState.instruments.size();

    for (const InstrumentGroup group: inState.instruments) {
        out << ' ' << static_cast<int>(group);
    }

    const CorrectionSettings& corrections = inState.corrections;
    out << "\nleadin " << corrections.lead_in_frames << '\n';
    out << "gate " << (corrections.silence_gate ? 1 : 0) << ' ' << toBits(corrections.gate_threshold) << ' '
        << toBits(corrections.gate_min_duration) << ' ' << toBits(corrections.gate_release) << '\n';
    out << "decoded " << inState.decoded << '\n';
    out << "prologue " << (inState.in_prologue ? 1 : 0) << '\n';
    out << "open " << inState.open.size() << '\n';

    for (const OpenNoteTracker::OpenNote& note: inState.open) {
        out << note.key.program << ' ' << note.key.pitch << ' ' << toFrames(note.onset) << '\n';
    }

    out << "withheld " << inState.withheld.size() << '\n';

    for (const TrackedNote& tracked: inState.withheld) {
        // A drum hit's offset is always onset + 10 ms, recomputed on resume.
        if (tracked.drum_hit) {
            out << "d " << tracked.key.pitch << ' ' << toFrames(tracked.note.onset) << '\n';
        } else {
            out << "n " << tracked.key.program << ' ' << tracked.key.pitch << ' ' << toFrames(tracked.note.onset) << ' '
                << toFrames(tracked.note.offset) << '\n';
        }
    }

    return out.str();
}

std::optional<ResumeState> parseResumeState(std::string_view inText)
{
    Reader in(inText);
    ResumeState state;

    in.expect(MAGIC);
    in.number(FORMAT_VERSION, FORMAT_VERSION);

    in.expect("samples");
    state.n_samples = in.number<std::size_t>(0, std::numeric_limits<std::size_t>::max());

    in.expect("prelude");
    state.prelude_forcing = in.number(0, 1) == 1;

    in.expect("instruments");
    const int n_instruments = in.number(0, MAX_VALUE);

    for (int i = 0; i < n_instruments && !in.failed(); ++i) {
        state.instruments.push_back(static_cast<InstrumentGroup>(in.number(0, MAX_VALUE)));
    }

    in.expect("leadin");
    state.corrections.lead_in_frames = in.number(0, MAX_VALUE);

    in.expect("gate");
    state.corrections.silence_gate = in.number(0, 1) == 1;
    state.corrections.gate_threshold = in.bits<float>();
    state.corrections.gate_min_duration = in.bits<double>();
    state.corrections.gate_release = in.bits<double>();

    in.expect("decoded");
    state.decoded = in.number(1, MAX_VALUE);

    in.expect("prologue");
    state.in_prologue = in.number(0, 1) == 1;

    in.expect("open");
    const int n_open = in.number(0, MAX_VALUE);

    for (int i = 0; i < n_open && !in.failed(); ++i) {
        OpenNoteTracker::OpenNote note;
        note.key.program = in.number(0, MAX_PROGRAM);
        note.key.pitch = in.number(0, MAX_PITCH);
        note.onset = fromFrames(in.number(-MAX_VALUE, MAX_VALUE));

        // The tracker holds at most one note per key.
        if (std::any_of(state.open.begin(), state.open.end(), [&note](const auto& n) { return n.key == note.key; })) {
            return std::nullopt;
        }

        state.open.push_back(note);
    }

    in.expect("withheld");
    const int n_withheld = in.number(0, MAX_VALUE);

    for (int i = 0; i < n_withheld && !in.failed(); ++i) {
        const std::string_view kind = in.word();

        TrackedNote tracked;

        if (kind == "d") {
            tracked.drum_hit = true;
            tracked.key = {DRUM_PROGRAM, in.number(0, MAX_PITCH)};
            tracked.note.onset = fromFrames(in.number(-MAX_VALUE, MAX_VALUE));
        } else if (kind == "n") {
            tracked.key.program = in.number(0, MAX_PROGRAM);
            tracked.key.pitch = in.number(0, MAX_PITCH);
            tracked.note.onset = fromFrames(in.number(-MAX_VALUE, MAX_VALUE));
            tracked.note.offset = fromFrames(in.number(-MAX_VALUE, MAX_VALUE));
        } else {
            return std::nullopt;
        }

        state.withheld.push_back(tracked);
    }

    if (in.failed() || !in.atEnd()) {
        return std::nullopt;
    }

    return state;
}

} // namespace msl
