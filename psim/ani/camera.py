#! /usr/bin/env python

class CustomCamera(object):
    def __init__(self):
        self.node = base.camera

        self.states = {}
        self.last_state = None
        self.has_focus = False


    def create_focus(self, pos=None):
        self.focus = render.attachNewNode("camera_focus")
        self.focus.setHpr(90,0,0)

        if pos is not None:
            self.focus.setPos(*pos)

        self.node.reparentTo(self.focus)
        self.node.setPos(50, 0, 0)
        self.node.lookAt(self.focus)

        self.has_focus = True


    def store_state(self, name, overwrite=False):
        if name in self.states:
            if overwrite:
                self.remove_state(name)
            else:
                raise Exception(f"CustomCamera :: '{name}' is already a camera state")

        self.states[name] = {
            'CamHpr': self.node.getHpr(),
            'CamPos': self.node.getPos(),
            'FocusHpr': self.focus.getHpr() if self.has_focus else None,
            'FocusPos': self.focus.getPos() if self.has_focus else None,
        }

        self.last_state = name


    def load_state(self, name, ok_if_not_exists=False):
        if name not in self.states:
            if ok_if_not_exists:
                return
            else:
                raise Exception(f"CustomCamera :: '{name}' is not a camera state")

        self.node.setPos(self.states[name]['CamPos'])
        self.node.setHpr(self.states[name]['CamHpr'])

        if self.has_focus:
            self.focus.setPos(self.states[name]['FocusPos'])
            self.focus.setHpr(self.states[name]['FocusHpr'])


    def load_last(self, ok_if_not_exists=False):
        """Loads the last state that was stored"""
        self.load_state(self.last_state, ok_if_not_exists=ok_if_not_exists)


    def remove_state(self, name):
        del self.states[name]

