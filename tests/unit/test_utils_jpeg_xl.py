import io
import pytest
from unittest.mock import mock_open, patch
from library.utils.jpeg_xl_util import get_jxl_size, decode_container, decode_codestream

class BitWriter:
    """Helper to construct bitstreams for JXL testing."""
    def __init__(self):
        self.bits = ""
        
    def write(self, value, length):
        """Write integer value with given bit length."""
        binary = format(value, f"0{length}b")
        # JXL reads little-endian for byte loading, but get_bits does:
        # bitmask = 2**length - 1
        # bits = (int.from_bytes(...) >> shift) & bitmask
        # This implies standard LSB-first packing usually?
        # Let's check JXLBitstream.get_bits implementation:
        # self.bitstream.extend(file.read(length))
        # bits = ... >> self.shift
        # This means we simply need to append bytes.
        # Wait, the implementation effectively treats the file as a stream of bits. 
        # But constructing the BYTE array requires care. 
        # Actually, let's look at JXLBitstream again in detail.
        # It reads BYTES, accumulates them into a bytearray/int. 
        # "int.from_bytes(self.bitstream, "little")"
        # So it's Little Endian. The 0th byte is LSB. 0th bit of 0th byte is valid.
        # So we write bits into an integer and then dump to bytes.
        self.bits += binary[::-1] # Reverse because we fill LSB first in the integer logic
        
    def to_bytes(self):
        # Pad to full bytes
        if len(self.bits) % 8 != 0:
            self.bits += "0" * (8 - len(self.bits) % 8)
            
        # Convert bits to bytes (Little Endian)
        # The string is now LSB...MSB order roughly for the construction logic?
        # No, let's keep it simple.
        # We need an integer where the first bit written is the LSB.
        total_val = 0
        for i, bit in enumerate(self.bits):
            if bit == '1':
                total_val |= (1 << i)
                
        num_bytes = len(self.bits) // 8
        return total_val.to_bytes(num_bytes, byteorder='little')

class TestJXLUtils:
    
    # === Smoke Tests (Mocked) ===
    
    def test_get_jxl_size_bare_codestream_mocked(self):
        """Test that bare codestream is detected and decoded using mocks."""
        # Signature FF0A
        mock_data = bytes.fromhex("FF0A") + b"\x00" * 10 
        
        with patch("builtins.open", mock_open(read_data=mock_data)):
            with patch("library.utils.jpeg_xl_util.decode_codestream", return_value=(100, 200)) as mock_decode:
                width, height = get_jxl_size("dummy.jxl")
                
                assert width == 100
                assert height == 200
                mock_decode.assert_called_once()
                
    def test_get_jxl_size_container_format_mocked(self):
        """Test that container format is detected and decoded using mocks."""
        # Signature NOT FF0A, but ISOBMFF like
        mock_data = bytes.fromhex("0000000C") + b"\x00" * 10
        
        with patch("builtins.open", mock_open(read_data=mock_data)):
            with patch("library.utils.jpeg_xl_util.decode_container", return_value=(300, 400)) as mock_decode:
                width, height = get_jxl_size("dummy_container.jxl")
                
                assert width == 300
                assert height == 400
                mock_decode.assert_called_once()
                
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised/propagated."""
        with pytest.raises(FileNotFoundError):
            get_jxl_size("nonexistent_file.jxl")

    # === Logic Tests (Synthetic Binary) ===

    def test_decode_bare_codestream_logic(self):
        """
        Test logic:
        Signature: FF0A (skipped by parser, but we must provide stream starting AFTER offset if we call decode directly? 
        No, decode_codestream takes 'file' and 'offset'.
        Internal JXLBitstream does file.seek(offset). 
        Then skips 16 bits (signature).
        So we need to provide the signature in the bytes if offset points to it.
        
        Construct stream:
        Sig: FF0A
        SizeHeader:
          div8: 1 (1 bit)
          height: 0 (5 bits) -> val 0 -> 8*(1+0) = 8
          ratio: 1 (3 bits) -> width=height = 8
        Total: 16 + 1 + 5 + 3 = 25 bits
        """
        writer = BitWriter()
        
        # We don't write signature into BitWriter because JXLBitstream logic 
        # reads the file using python's read().
        # However, decode_codestream does: codestream.get_bits(16) # Skip signature
        # So we DOES need to write signature bits.
        # 0AFF -> FF 0A in Little Endian?
        # Signature is FF0A. 
        # get_bits(16) -> reads 2 bytes. 
        # int.from_bytes(..., 'little'). 
        # If file has b'\xFF\x0A', int is 0x0AFF.
        # But wait, signature check usually reads bytes.
        # In decode_codestream: "Skip signature ... codestream.get_bits(16)"
        # It just consumes 16 bits. It doesn't validate strictly inside decode_codestream.
        
        # Let's just write dummy 16 bits.
        writer.write(0xFFFF, 16)
        
        # div8 = 1
        writer.write(1, 1)
        # height parameter = 0 (5 bits) -> means 8 pixels if div8 is set
        writer.write(0, 5) 
        # ratio = 1 (3 bits) -> width == height
        writer.write(1, 3) 
        
        data = writer.to_bytes()
        
        f = io.BytesIO(data)
        width, height = decode_codestream(f, offset=0)
        
        assert height == 8 * (1 + 0) # 8
        assert width == 8
        
    def test_decode_codestream_explicit_width(self):
        """
        Test div8=1, ratio=0 (explicit width)
        """
        writer = BitWriter()
        writer.write(0, 16) # Sig
        
        # div8 = 1
        writer.write(1, 1)
        # height param = 1 (5 bits) -> 8*(1+1) = 16
        writer.write(1, 5)
        # ratio = 0 (3 bits) -> explicit width
        writer.write(0, 3)
        # width param = 2 (5 bits) -> 8*(1+2) = 24
        writer.write(2, 5)
        
        f = io.BytesIO(writer.to_bytes())
        width, height = decode_codestream(f)
        
        assert height == 16
        assert width == 24

    def test_decode_codestream_div8_false(self):
        """
        Test div8=0 (variable distribution)
        distribution = 0 (2 bits) -> 1 + get_bits(9)
        val = 99 (9 bits) -> height = 100
        ratio = 1 -> width=height
        """
        writer = BitWriter()
        writer.write(0, 16) # Sig
        
        # div8 = 0
        writer.write(0, 1)
        
        # distribution = 0 (2 bits)
        writer.write(0, 2)
        # height param = 99 (9 bits)
        writer.write(99, 9)
        
        # ratio = 1 (3 bits)
        writer.write(1, 3)
        
        f = io.BytesIO(writer.to_bytes())
        width, height = decode_codestream(f)
        
        assert height == 100
        assert width == 100

    def test_container_structure(self):
        """Test ISOBMFF container structure parsing."""
        # 1. Signature Box (12 bytes)
        # Length: 12 (00 00 00 0C)
        # Type: JXL<space> (4A 58 4C 20)
        # Magic: 0D 0A 87 0A
        sig_box = bytes.fromhex("0000000C 4A584C20 0D0A870A")
        
        # 2. File Type Box (20 bytes)
        # Length: 20 (00 00 00 14)
        # Type: ftyp (66 74 79 70)
        # Content: jxl<space> ...
        ftyp_box = bytes.fromhex("00000014 66747970 6A786C20 00000000 6A786C20")
        
        # 3. jxlc box (Contiguous Codestream)
        # Construct codestream data first
        writer = BitWriter()
        writer.write(0, 16) # Sig inside codestream
        writer.write(1, 1) # div8=1
        writer.write(0, 5) # h=8
        writer.write(1, 3) # w=h
        codestream_data = writer.to_bytes()
        
        # Length: 8 (header) + len(data)
        length = 8 + len(codestream_data)
        # Type: jxlc (6A 78 6C 63)
        
        jxlc_header = length.to_bytes(4, 'big') + b'jxlc'
        
        file_data = sig_box + ftyp_box + jxlc_header + codestream_data
        
        f = io.BytesIO(file_data)
        width, height = decode_container(f)
        
        assert height == 8
        assert width == 8

    def test_container_invalid_signature(self):
        f = io.BytesIO(b"INVALID_SIGNATURE_HERE")
        with pytest.raises(ValueError, match="Invalid signature"):
            decode_container(f)

    def test_container_partial_jxlp(self):
        """
        Test multiple jxlp boxes. 
        Note: The implementation reads partial data from boxes.
        We need to split the codestream across two boxes.
        """
        # Create a codestream that is slightly longer to ensure split matters
        writer = BitWriter()
        writer.write(0, 16) # Sig
        writer.write(1, 1)  # div8=1
        writer.write(0, 5)  # val=0 => h=8
        writer.write(1, 3)  # ratio=1 => w=8
        # Pad slightly to ensure we have bytes to split
        writer.write(0, 8) 
        
        full_data = writer.to_bytes()
        split_point = 2 # Split after 2 bytes (inside signature usually, or right after)
        
        part1 = full_data[:split_point]
        part2 = full_data[split_point:]
        
        # Headers
        sig_box = bytes.fromhex("0000000C 4A584C20 0D0A870A")
        ftyp_box = bytes.fromhex("00000014 66747970 6A786C20 00000000 6A786C20")
        
        # jxlp box 1 (index 0)
        # Length = 8 + 4 (index) + len(part1)
        len1 = 12 + len(part1)
        # Type: jxlp (6A 78 6C 70)
        box1 = len1.to_bytes(4, 'big') + b'jxlp' + (0).to_bytes(4, 'big') + part1
        
        # jxlp box 2 (index 1)
        len2 = 12 + len(part2)
        box2 = len2.to_bytes(4, 'big') + b'jxlp' + (1).to_bytes(4, 'big') + part2
        
        file_data = sig_box + ftyp_box + box1 + box2
        
        f = io.BytesIO(file_data)
        width, height = decode_container(f)
        
        assert height == 8
        assert width == 8
