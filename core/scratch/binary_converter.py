import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def text_to_binary(text: str) -> str:
    """
    Converts a text string to a binary string representation of its UTF-8 bytes.
    Each character's byte is formatted into 8 bits.
    """
    utf8_bytes = text.encode('utf-8')
    binary_representation = ''.join(format(b, '08b') for b in utf8_bytes)
    return binary_representation

def binary_to_text(binary_str: str) -> str:
    """
    Converts a binary string back into a UTF-8 text string.
    Processes the binary string in blocks of 8 bits.
    """
    # Remove any whitespace if present
    binary_str = binary_str.replace(" ", "")
    
    # Check if length is a multiple of 8
    if len(binary_str) % 8 != 0:
        raise ValueError("Binary string length must be a multiple of 8 bits.")
        
    byte_list = []
    for i in range(0, len(binary_str), 8):
        byte_segment = binary_str[i:i+8]
        byte_val = int(byte_segment, 2)
        byte_list.append(byte_val)
        
    try:
        decoded_bytes = bytes(byte_list)
        return decoded_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        return f"[Decode Error: {e}]"

def main():
    print("=" * 60)
    print(" 💻 [텍스트 ↔ 바이너리 인코딩/디코딩 실습 스크립트]")
    print("=" * 60)
    
    # 1. Original Text
    original_text = "Hello World!"
    print(f"\n[1] 원본 텍스트: {original_text}")
    
    # 2. Convert to Binary (0s and 1s)
    binary_data = text_to_binary(original_text)
    print(f"[2] 2진수(바이너리) 변환 결과 (총 {len(binary_data)} 비트):")
    # Print in groups of 8 bits for readability
    readable_binary = ' '.join(binary_data[i:i+8] for i in range(0, len(binary_data), 8))
    print(f"    >> {readable_binary}")
    
    # 3. Decode back to Text
    decoded_text = binary_to_text(binary_data)
    print(f"[3] 2진수를 다시 텍스트로 변환: {decoded_text}")
    
    # 4. Demonstrate Bit Manipulation (how simple changes affect data)
    print("\n[4] 데이터 연결성 테스트 (1비트 변조/오류 주입)")
    # Flip the very first bit of 'H' (which is '01001000' in binary) -> '11001000'
    flipped_binary = '1' + binary_data[1:]
    corrupted_text = binary_to_text(flipped_binary)
    print(f"    >> 원본 첫 글자 'H' 바이너리: {binary_data[0:8]}")
    print(f"    >> 변조된 첫 글자 바이너리  : {flipped_binary[0:8]}")
    print(f"    >> 변조 후 디코딩 결과      : {corrupted_text}")
    print("=" * 60)
    print(" 컴퓨터 파일이나 가중치도 이와 같이 8비트 단위의 데이터 약속으로 구성되어 있으며,")
    print(" 비트가 어긋나면 정상적인 언어 디코딩이 불가능해집니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()
