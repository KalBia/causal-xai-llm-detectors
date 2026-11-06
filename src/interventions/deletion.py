from .base import Intervention


class Deletion(Intervention):
    name = "deletion"

    def apply(self, tokens, indices):
        drop = set(indices)
        return " ".join(t for i, t in enumerate(tokens) if i not in drop)
