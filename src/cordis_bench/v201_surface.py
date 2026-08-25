"""V2.0.1 identifier-surface patch.

V2.0 used long randomized numeric identifiers such as ``plugin_849065282``.
V2.0.1 keeps identifiers semantically opaque while shortening the model-facing
surface. The patch deliberately consumes the same RNG draws as V2.0 so a fixed
seed preserves the semantic world: effects, values, interference structure,
schedules and targets are unchanged apart from alpha-renaming.
"""

from __future__ import annotations

from string import ascii_lowercase

from . import v112_formal_challenge as _formal_source
from . import v2_native as _native_source

V201_IDENTIFIER_SURFACE = "short_opaque"
V201_VERSION = "2.0.1"

_SUFFIXES = tuple(ascii_lowercase) + tuple(
    first + second for first in ascii_lowercase for second in ascii_lowercase
)


def _short_native_label(rng, prefix):
    """Consume the V2.0 RNG draw, but render it as a short opaque label."""
    raw = rng.randrange(10**9)
    state = getattr(rng, "_cordis_v201_identifier_state", None)
    if state is None:
        state = {}
        setattr(rng, "_cordis_v201_identifier_state", state)
    used = state.setdefault(prefix, set())
    start = raw % len(_SUFFIXES)
    for offset in range(len(_SUFFIXES)):
        suffix = _SUFFIXES[(start + offset) % len(_SUFFIXES)]
        if suffix not in used:
            used.add(suffix)
            return f"{prefix}_{suffix}"
    raise RuntimeError(f"V2.0.1 exhausted short identifiers for prefix {prefix!r}")


def _short_formal_keys(rng, n):
    """Preserve V2.0 sampling while alpha-renaming capability keys."""
    raw = rng.sample(range(100_000, 999_999), n)
    if n > len(_SUFFIXES):
        raise ValueError("V2.0.1 formal identifier pool exhausted")
    mapping = {
        value: f"cap_{_SUFFIXES[index]}"
        for index, value in enumerate(sorted(raw))
    }
    return [mapping[value] for value in raw]


_ORIGINAL_NATIVE_SAMPLE = _native_source._sample_native_spec


def _sample_native_spec_v201(rng, size):
    # Identifier uniqueness is scoped to one latent native world.
    setattr(rng, "_cordis_v201_identifier_state", {})
    return _ORIGINAL_NATIVE_SAMPLE(rng, size)


def install_v201_identifier_surface():
    """Install the V2.0.1 alpha-renaming layer exactly once."""
    if getattr(_native_source, "_v201_identifier_surface_installed", False):
        return
    _native_source._label = _short_native_label
    _native_source._sample_native_spec = _sample_native_spec_v201
    _native_source._v201_identifier_surface_installed = True
    _formal_source._random_keys = _short_formal_keys
    # V2 formal generation delegates key sampling through the hardened
    # challenge module's private base implementation.
    _formal_source._base._random_keys = _short_formal_keys


install_v201_identifier_surface()
