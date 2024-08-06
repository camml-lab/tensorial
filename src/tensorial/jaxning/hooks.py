class ModelHooks:
    def on_train_start(self) -> None:
        """Training is about to begin"""

    def on_train_end(self) -> None:
        """Training is ending"""

    def configure_model(self, batch):
        """
        Configure the model before a fit/va/test/predict stage.  This will be called at the
        beginning of each of these stages, so it's important that the implementation is a no-op
        after the first time it is called.
        """
