# Modified from https://github.com/Fraetor/jxl_decode Original license: MIT
# Added partial read support for up to 200x speedup

import os

import io


def get_file_size(file) -> int:
    """
    Get the size of a file.

    Args:
        file: The file object to get the size of.

    Returns:
        int: The size of the file in bytes.
    """
    try:
        return os.fstat(file.fileno()).st_size
    except (AttributeError, io.UnsupportedOperation):
        current_pos = file.tell()
        file.seek(0, 2)
        size = file.tell()
        file.seek(current_pos)
        return size


class JXLBitstream:
    """
    A stream of bits with methods for easy handling.
    """

    def __init__(self, file, offset: int = 0, offsets: list[list[int]] | None = None):
        """
        Initialize the JXLBitstream.

        Args:
            file: The file object to read from.
            offset (int, optional): The offset to start reading from. Defaults to 0.
            offsets (List[List[int]], optional): A list of offsets for partial reading. Defaults to None.
        """
        self.file = file
        self.offsets = offsets
        self.bitstream = bytearray()
        self.shift = 0  # Current bit position relative to start of bitstream

        if self.offsets:
            self.current_box_index = 0
            self.file.seek(self.offsets[0][1])
            self.bytes_read_from_current_box = 0
        else:
            self.file.seek(offset)

    def get_bits(self, length: int = 1) -> int:
        """
        Get the specified number of bits from the bitstream.

        Args:
            length (int, optional): The number of bits to read. Defaults to 1.

        Returns:
            int: The read bits as an integer.
        """
        target_byte_count = (self.shift + length + 7) // 8
        needed_bytes = target_byte_count - len(self.bitstream)

        while needed_bytes > 0:
            if self.offsets:
                # Check availability in current box
                if self.current_box_index >= len(self.offsets):
                    # No more boxes, but needed bytes? Stop reading.
                    break

                box_len = self.offsets[self.current_box_index][2]
                remain_in_box = box_len - self.bytes_read_from_current_box

                to_read = min(needed_bytes, remain_in_box)
                if to_read > 0:
                    data = self.file.read(to_read)
                    self.bitstream.extend(data)
                    self.bytes_read_from_current_box += len(data)
                    needed_bytes -= len(data)

                if self.bytes_read_from_current_box >= box_len:
                    # Move to next box
                    self.current_box_index += 1
                    if self.current_box_index < len(self.offsets):
                        self.file.seek(self.offsets[self.current_box_index][1])
                        self.bytes_read_from_current_box = 0
            else:
                data = self.file.read(needed_bytes)
                if not data:
                    break
                self.bitstream.extend(data)
                needed_bytes -= len(data)

        bitmask = (1 << length) - 1
        bits = (int.from_bytes(self.bitstream, "little") >> self.shift) & bitmask
        self.shift += length
        return bits


def decode_codestream(file, offset: int = 0, offsets: list[list[int]] | None = None) -> tuple[int, int]:
    """
    Decodes the actual codestream.
    JXL codestream specification: http://www-internal/2022/18181-1

    Args:
        file: The file object containing the codestream.
        offset (int, optional): The offset to start reading from. Defaults to 0.
        offsets (List[List[int]], optional): A list of offsets for partial reading. Defaults to None.

    Returns:
        Tuple[int, int]: A tuple containing the width and height of the image.
    """

    # Convert codestream to int within an object to get some handy methods.
    codestream = JXLBitstream(file, offset=offset, offsets=offsets)

    # Skip signature
    codestream.get_bits(16)

    # SizeHeader
    height = 0  # Will be set below
    div8 = codestream.get_bits(1)
    if div8:
        height = 8 * (1 + codestream.get_bits(5))
    else:
        distribution = codestream.get_bits(2)
        match distribution:
            case 0:
                height = 1 + codestream.get_bits(9)
            case 1:
                height = 1 + codestream.get_bits(13)
            case 2:
                height = 1 + codestream.get_bits(18)
            case 3:
                height = 1 + codestream.get_bits(30)
    width = 0  # Will be set below
    ratio = codestream.get_bits(3)
    if div8 and not ratio:
        width = 8 * (1 + codestream.get_bits(5))
    elif not ratio:
        distribution = codestream.get_bits(2)
        match distribution:
            case 0:
                width = 1 + codestream.get_bits(9)
            case 1:
                width = 1 + codestream.get_bits(13)
            case 2:
                width = 1 + codestream.get_bits(18)
            case 3:
                width = 1 + codestream.get_bits(30)
    else:
        match ratio:
            case 1:
                width = height  # TODO: Local variable 'height' might be referenced before assignment
            case 2:
                width = (height * 12) // 10
            case 3:
                width = (height * 4) // 3
            case 4:
                width = (height * 3) // 2
            case 5:
                width = (height * 16) // 9
            case 6:
                width = (height * 5) // 4
            case 7:
                width = (height * 2) // 1
    return width, height  # TODO: Local variable 'width' might be referenced before assignment


def decode_container(file) -> tuple[int, int]:
    """
    Parses the ISOBMFF container, extracts the codestream, and decodes it.
    JXL container specification: http://www-internal/2022/18181-2

    Args:
        file: The file object containing the container.

    Returns:
        Tuple[int, int]: A tuple containing the width and height of the image.

    Raises:
        ValueError: If the container structure is invalid or required boxes are missing.
    """

    def parse_box(file, file_start: int) -> dict:
        file.seek(file_start)
        LBox = int.from_bytes(file.read(4), "big")
        XLBox = None
        if 1 < LBox <= 8:
            raise ValueError(f"Invalid LBox at byte {file_start}.")
        if LBox == 1:
            file.seek(file_start + 8)
            XLBox = int.from_bytes(file.read(8), "big")
            if XLBox <= 16:
                raise ValueError(f"Invalid XLBox at byte {file_start}.")
        if XLBox:
            header_length = 16
            box_length = XLBox
        else:
            header_length = 8
            if LBox == 0:
                box_length = get_file_size(file) - file_start
            else:
                box_length = LBox
        file.seek(file_start + 4)
        box_type = file.read(4)
        file.seek(file_start)
        return {
            "length": box_length,
            "type": box_type,
            "offset": header_length,
        }

    file.seek(0)
    # Reject files missing required boxes. These two boxes are required to be at
    # the start and contain no values, so we can manually check there presence.
    # Signature box. (Redundant as has already been checked.)
    if file.read(12) != bytes.fromhex("0000000C 4A584C20 0D0A870A"):
        raise ValueError("Invalid signature box.")
    # File Type box.
    if file.read(20) != bytes.fromhex("00000014 66747970 6A786C20 00000000 6A786C20"):
        raise ValueError("Invalid file type box.")

    offset = 0
    offsets = []
    data_offset_not_found = True
    container_pointer = 32
    file_size = get_file_size(file)
    while data_offset_not_found:
        box = parse_box(file, container_pointer)
        match box["type"]:
            case b"jxlc":
                offset = container_pointer + box["offset"]
                data_offset_not_found = False
            case b"jxlp":
                file.seek(container_pointer + box["offset"])
                index = int.from_bytes(file.read(4), "big")
                offsets.append([index, container_pointer + box["offset"] + 4, box["length"] - box["offset"] - 4])
        container_pointer += box["length"]
        if container_pointer >= file_size:
            data_offset_not_found = False

    if offsets:
        offsets.sort(key=lambda i: i[0])
    file.seek(0)

    return decode_codestream(file, offset=offset, offsets=offsets)


def get_jxl_size(path: str) -> tuple[int, int]:
    """
    Get the dimensions of a JPEG XL image.

    Args:
        path (str): The path to the JPEG XL file.

    Returns:
        Tuple[int, int]: A tuple containing the width and height of the image.
    """
    with open(path, "rb") as file:
        if file.read(2) == bytes.fromhex("FF0A"):
            return decode_codestream(file)
        return decode_container(file)
