#pragma once

#include "muscriptor/log.hpp"

namespace msl
{

/**
 * Point ggml's global log hook at this library's sink, replacing ggml's own
 * handler -- which writes every message to stderr and ignores its level.
 *
 * Idempotent. Called on the model-load path so the default is silence whether
 * or not the host ever calls `setLogCallback`.
 */
void installLogHook();

} // namespace msl
