"""Browser-side motor viewer: serve smoothed 3-D meshes and live checkpoints."""

from organic_motor.web.server import app, create_app

__all__ = ["app", "create_app"]
