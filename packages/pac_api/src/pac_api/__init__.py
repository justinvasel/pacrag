def __getattr__(name: str) -> object:
    if name == "app":
        from pac_api.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]
