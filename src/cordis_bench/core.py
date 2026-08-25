from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class Theory:
    undo_order: str = "lifo"

    def __post_init__(self):
        if self.undo_order not in {"lifo", "fifo"}:
            raise ValueError(f"unknown undo order: {self.undo_order}")


DEFAULT_THEORY = Theory()


@dataclass(frozen=True)
class Op:
    kind: str
    i: int
    j: int | None = None
    value: int = 0

    def apply(self, values, modulus):
        x = list(values)
        if self.kind == "add":
            x[self.i] = (x[self.i] + self.value) % modulus
        elif self.kind == "shear":
            x[self.i] = (x[self.i] + self.value * x[self.j]) % modulus
        elif self.kind == "scale":
            x[self.i] = (self.value * x[self.i]) % modulus
        elif self.kind == "swap":
            x[self.i], x[self.j] = x[self.j], x[self.i]
        else:
            raise ValueError(f"unknown operation: {self.kind}")
        return tuple(x)

    def inverse(self, modulus):
        if self.kind in {"add", "shear"}:
            return Op(self.kind, self.i, self.j, -self.value)
        if self.kind == "scale":
            if gcd(self.value, modulus) != 1:
                raise ValueError("scale is not invertible")
            return Op("scale", self.i, value=pow(self.value, -1, modulus))
        if self.kind == "swap":
            return self
        raise ValueError(f"unknown operation: {self.kind}")

    def render(self):
        if self.kind == "add":
            sign = "+=" if self.value >= 0 else "-="
            return f"x{self.i} {sign} {abs(self.value)}"
        if self.kind == "shear":
            sign = "+=" if self.value >= 0 else "-="
            return f"x{self.i} {sign} {abs(self.value)}*x{self.j}"
        if self.kind == "scale":
            return f"x{self.i} *= {self.value}"
        if self.kind == "swap":
            return f"swap(x{self.i}, x{self.j})"
        raise ValueError(f"unknown operation: {self.kind}")


@dataclass(frozen=True)
class Component:
    name: str
    requires: frozenset[str]
    provides: frozenset[str]
    effect: tuple[Op, ...]

    def inverse_effect(self, modulus, undo_order="lifo"):
        inverses = tuple(op.inverse(modulus) for op in self.effect)
        if undo_order == "lifo":
            return tuple(reversed(inverses))
        if undo_order == "fifo":
            return inverses
        raise ValueError(f"unknown undo order: {undo_order}")


@dataclass(frozen=True)
class World:
    modulus: int
    width: int
    components: tuple[Component, ...]

    @property
    def by_name(self):
        return {component.name: component for component in self.components}

    @property
    def provider_by_key(self):
        providers = {}
        for component in self.components:
            for key in component.provides:
                if key in providers:
                    raise ValueError(f"multiple providers for key {key}")
                providers[key] = component.name
        return providers


@dataclass(frozen=True)
class RuntimeState:
    values: tuple[int, ...]
    enabled: frozenset[str]
    active: frozenset[str]


@dataclass(frozen=True)
class Action:
    kind: str
    component: str

    def render(self):
        return f"{self.kind}({self.component})"
