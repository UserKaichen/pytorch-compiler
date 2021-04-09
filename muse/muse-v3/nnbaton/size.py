import numpy


class Size:
    def __init__(self, value):
        self.value = numpy.int64(value)

    def to_b(self) -> int:
        raise NotImplementedError

    def __lt__(self, other):
        return self.to_b().__lt__(other.to_b())

    def __le__(self, other):
        return self.to_b().__le__(other.to_b())

    def __mul__(self, other: int):
        return b(self.to_b() * other)

    def __floordiv__(self, other):
        if isinstance(other, int):
            return b(self.to_b() // other)
        if isinstance(other, Size):
            return self.to_b() // other.to_b()
        raise NotImplementedError

    def __add__(self, other):
        return b(self.to_b() + other.to_b())


class MB(Size):
    def __str__(self):
        return f"{self.value}MB"

    def to_b(self):
        return self.value * 1024 * 1024 * 8


class KB(Size):
    def __str__(self):
        return f"{self.value}KB"

    def to_b(self):
        return self.value * 1024 * 8


class B(Size):
    def __str__(self):
        return f"{self.value}B"

    def to_b(self):
        return self.value * 8


class b(Size):
    def __str__(self):
        return f"{self.value}b"

    def to_b(self):
        return self.value
