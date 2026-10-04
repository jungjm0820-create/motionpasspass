"""Motionator - fast FK motion paths (native C++ core) + weight-based bone picking."""
from . import overlay, picker, ui

_modules = (ui, overlay, picker)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
