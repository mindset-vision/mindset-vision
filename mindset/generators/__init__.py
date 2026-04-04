"""generator registry for mindset-vision datasets."""

REGISTRY = {}


def get_generator(name):
    """look up a generator by name."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"unknown generator '{name}'. available: {available}")
    return REGISTRY[name]


def list_generators():
    """return all registered generators grouped by category."""
    by_category = {}
    for name, info in sorted(REGISTRY.items()):
        cat = info["category"]
        by_category.setdefault(cat, []).append(name)
    return by_category
