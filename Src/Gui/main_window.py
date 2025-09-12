import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk


class VtubePlugin(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.KitsuneSemCalda.VtubePlugin")
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        window = Gtk.ApplicationWindow(application=app)
        window.set_title("VtubePlugin")
        window.set_default_size(400, 300)

        window.show()


def StartApp():
    app = VtubePlugin()
    app.run()
