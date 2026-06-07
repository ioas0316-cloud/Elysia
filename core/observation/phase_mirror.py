import ctypes
import os

# Define the C structure in Python to match the hardware layer
class TopologyField(ctypes.Structure):
    # Node structure: uint8_t state
    class Node(ctypes.Structure):
        _fields_ = [("state", ctypes.c_uint8)]

    _fields_ = [
        ("nodes", Node * 1024), # FIELD_SIZE is 1024
        ("head", ctypes.c_int)
    ]

class PhaseMirror:
    """
    The Phase Mirror purely reflects the physical state of the underlying
    C hardware layer (the Topology Field) without modifying it.
    It passes the raw ASCII streams down and reads the resulting tensions back up.
    """
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Dynamically resolve path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, "..", "hardware", "libtopology.so")

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Hardware library not found: {lib_path}. Please compile the C library first.")

        self.lib = ctypes.CDLL(os.path.abspath(lib_path))

        # void init_field(TopologyField* field)
        self.lib.init_field.argtypes = [ctypes.POINTER(TopologyField)]
        self.lib.init_field.restype = None

        # void apply_stimulus(TopologyField* field, uint8_t byte_val)
        self.lib.apply_stimulus.argtypes = [ctypes.POINTER(TopologyField), ctypes.c_uint8]
        self.lib.apply_stimulus.restype = None

        # uint8_t get_node_state(const TopologyField* field, int index)
        self.lib.get_node_state.argtypes = [ctypes.POINTER(TopologyField), ctypes.c_int]
        self.lib.get_node_state.restype = ctypes.c_uint8

        # int get_head_position(const TopologyField* field)
        self.lib.get_head_position.argtypes = [ctypes.POINTER(TopologyField)]
        self.lib.get_head_position.restype = ctypes.c_int

        # Instantiate the field
        self.field = TopologyField()
        self.lib.init_field(ctypes.byref(self.field))

    def feed_stream(self, data: bytes):
        """
        Pours the raw ASCII byte stream directly into the hardware field.
        """
        for byte in data:
            self.lib.apply_stimulus(ctypes.byref(self.field), byte)

    def observe_field(self) -> list[int]:
        """
        Reads the entire physical state of the field as an array of bytes.
        This is the 'observation' of the twisted axis.
        """
        field_size = 1024
        state = []
        for i in range(field_size):
            state.append(self.lib.get_node_state(ctypes.byref(self.field), i))
        return state

    def get_head(self) -> int:
        """
        Returns the current position of the flowing stream in the ring buffer.
        """
        return self.lib.get_head_position(ctypes.byref(self.field))
