

def keyboard_layout_factory(layout_name):
    if layout_name != 'default':
        raise ValueError('unsupported keyboard layout %r' % (layout_name,))
    return CouldDoBetterKeyboardLayout()


def _key_down_up(name):
    return [KeyDown(name), KeyUp(name)]


class CouldDoBetterKeyboardLayout(object):
    """Converts single characters into keypresses

    This is a very osteer version, and this mechanism my be replaced with a
    better one.
    """

    CHAR_TO_NAME_MAP = {
        '\n': 'return',
        ' ': 'space',
        '\t': 'tab',
        '.': 'period'
    }

    def __call__(self, char):
        if char in CouldDoBetterKeyboardLayout.CHAR_TO_NAME_MAP:
            return _key_down_up(
                CouldDoBetterKeyboardLayout.CHAR_TO_NAME_MAP[char]
            )
        elif char.isalnum():
            if char.isupper():
                return [SHIFT_DOWN] + _key_down_up(char.lower()) + [SHIFT_UP]
            else:
                return _key_down_up(char)
        else:
            raise ValueError('unsupported character %r' % (char,))


class KeyDown(object):
    def __init__(self, keyname):
        self._keyname = keyname

    def __str__(self):
        return self._keyname


class KeyUp(object):
    def __init__(self, keyname):
        self._keyname = keyname

    def __str__(self):
        return self._keyname


class KeyDownUp(object):
    def __init__(self, keyname):
        self._keyname = keyname

    def __str__(self):
        return self._keyname


SHIFT_DOWN = KeyDown('shift')
SHIFT_UP = KeyUp('shift')