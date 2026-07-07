def __getattr__(name: str):
    if name == "Model":
        from rc_osda.models.reliability_calibrated_osda import Model

        return Model
    raise AttributeError(name)


__all__ = ["Model"]
