import reedsolo
import struct
import zlib
import time
import random
import pandas as pd
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import os

# Initialize Reed-Solomon tables (optional but improves performance)
reedsolo.init_tables(0x11d)

# Global encryption key (should be securely managed in production)
encryption_key = get_random_bytes(16)

def encrypt_data(data, key):
    """
    Encrypt data using AES encryption.
    """
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return cipher.nonce + tag + ciphertext

def decrypt_data(encrypted_data, key):
    """
    Decrypt data using AES decryption.
    """
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data

def packetize(data, seq_num, priority_level=1):
    """
    Packetize the data with headers, compression, encryption, and error correction.
    """
    # Compress data
    compressed_data = zlib.compress(data)

    # Calculate checksum
    checksum = zlib.crc32(compressed_data) & 0xffffffff

    # Encrypt data
    encrypted_data = encrypt_data(compressed_data, encryption_key)

    # Append checksum to encrypted data
    encrypted_data_with_checksum = struct.pack('>I', checksum) + encrypted_data

    # Create header: 4 bytes for sequence number, 1 byte for priority, 4 bytes for timestamp
    timestamp = int(time.time())
    header = struct.pack('>IBI', seq_num, priority_level, timestamp)

    # Adjust error correction level based on priority
    ecc_symbols = 10 * priority_level  # Adjust as needed

    # Initialize Reed-Solomon codec
    rs = reedsolo.RSCodec(ecc_symbols)

    # Encode data with error correction
    data_with_ecc = rs.encode(encrypted_data_with_checksum)

    # Combine header and data
    packet = header + data_with_ecc
    return packet

def depacketize(packet):
    """
    Extract and correct data from the packet.
    """
    # Extract header
    header = packet[:9]
    seq_num, priority_level, timestamp = struct.unpack('>IBI', header)

    # Extract data
    data_with_ecc = packet[9:]

    # Adjust error correction level based on priority
    ecc_symbols = 10 * priority_level

    # Initialize Reed-Solomon codec
    rs = reedsolo.RSCodec(ecc_symbols)

    # Correct errors and decode data
    try:
        decoded_result = rs.decode(data_with_ecc)
        # decoded_result can be bytes or a tuple
        if isinstance(decoded_result, tuple):
            encrypted_data_with_checksum = decoded_result[0]
        else:
            encrypted_data_with_checksum = decoded_result
    except reedsolo.ReedSolomonError as e:
        print(f"Error processing packet {seq_num}: {e}")
        return None, seq_num

    # Extract checksum
    checksum_received = struct.unpack('>I', encrypted_data_with_checksum[:4])[0]
    encrypted_data = encrypted_data_with_checksum[4:]

    # Decrypt data
    try:
        compressed_data = decrypt_data(encrypted_data, encryption_key)
    except (ValueError, KeyError) as e:
        print(f"Error decrypting packet {seq_num}: {e}")
        return None, seq_num

    # Verify checksum
    checksum_calculated = zlib.crc32(compressed_data) & 0xffffffff
    if checksum_calculated != checksum_received:
        print(f"Checksum mismatch in packet {seq_num}")
        return None, seq_num

    # Decompress data
    try:
        data = zlib.decompress(compressed_data)
    except zlib.error as e:
        print(f"Error decompressing packet {seq_num}: {e}")
        return None, seq_num

    return data, seq_num

def simulate_transmission(packet, priority_level):
    """
    Simulate transmission errors on the packet.
    """
    packet_with_errors = bytearray(packet)
    # Reed-Solomon can correct up to ecc_symbols // 2 errors
    ecc_symbols = 10 * priority_level
    max_correctable_errors = ecc_symbols // 2

    # Introduce errors up to one more than the maximum correctable errors to test limits
    error_count = random.randint(0, max_correctable_errors + 1)
    for _ in range(error_count):
        idx = random.randint(9, len(packet_with_errors) - 1)
        packet_with_errors[idx] ^= random.getrandbits(8)  # Flip random bits to simulate error
    return bytes(packet_with_errors)

def send_data(data_chunks, starting_seq_num=1, priority_level=1):
    """
    Simulate sending data chunks by packetizing and introducing transmission errors.
    """
    transmitted_packets = []
    seq_num = starting_seq_num
    for chunk in data_chunks:
        packet = packetize(chunk, seq_num, priority_level)
        transmitted_packet = simulate_transmission(packet, priority_level)
        transmitted_packets.append(transmitted_packet)
        seq_num += 1
    return transmitted_packets

def receive_data(packets):
    """
    Simulate receiving data by depacketizing.
    """
    received_data = []
    for packet in packets:
        data, seq_num = depacketize(packet)
        if data:
            print(f"Packet {seq_num} received and corrected successfully.")
            received_data.append((seq_num, data))
        else:
            print(f"Packet {seq_num} could not be corrected.")
    # Sort received data by sequence number
    received_data.sort(key=lambda x: x[0])
    # Extract data
    data_list = [data for seq_num, data in received_data]
    return data_list

# Main processing function
def process_csv_file(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, header=None)
    total_rows = len(df)
    print(f"Total rows in CSV: {total_rows}")

    # Extract the first 10 rows (priority 1)
    first_ten = df.iloc[:10]
    # Extract the next 10 rows (priority 2)
    second_ten = df.iloc[10:20]

    # Convert chunks to bytes
    first_ten_data = first_ten.to_csv(index=False, header=False).encode('utf-8')
    second_ten_data = second_ten.to_csv(index=False, header=False).encode('utf-8')

    # Send the first ten rows with priority level 5 (highest)
    transmitted_packets_1 = send_data([first_ten_data], starting_seq_num=1, priority_level=5)
    # Send the second ten rows with priority level 3
    transmitted_packets_2 = send_data([second_ten_data], starting_seq_num=2, priority_level=3)

    # Simulate receiving the data
    received_data_1 = receive_data(transmitted_packets_1)
    received_data_2 = receive_data(transmitted_packets_2)

    # Combine received data
    all_received_data = received_data_1 + received_data_2

    # Decode and process received data
    for data in all_received_data:
        decoded_data = data.decode('utf-8')
        print("Received Data Chunk:")
        print(decoded_data)
        print("\n")

# Testing the code
if __name__ == "__main__":
    # Seed the random number generator for reproducibility
    random.seed(42)

    # Set the path to your CSV file
    csv_file_path = r"C:\Users\Morales Tommy\Desktop\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1970-01-19HR00_evid00002"
    # Check if file exists
    if os.path.exists(csv_file_path):
        process_csv_file(csv_file_path)
    else:
        print(f"CSV file not found at {csv_file_path}")
