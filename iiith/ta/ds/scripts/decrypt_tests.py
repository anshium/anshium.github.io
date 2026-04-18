import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_key(password: str):
    salt = b'phineas_and_ferb_is_the_best_cartoon' 
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def decrypt_file(enc_path, password):
    key = generate_key(password)
    f = Fernet(key)
    try:
        with open(enc_path, 'rb') as file:
            encrypted_data = file.read()
        decrypted_data = f.decrypt(encrypted_data)
        
        # Save without .enc
        original_name = enc_path[:-4] if enc_path.endswith('.enc') else enc_path + '.dec'
        with open(original_name, 'wb') as file:
            file.write(decrypted_data)
        print(f"Decrypted: {original_name}")
    except Exception as e:
        print(f"Decryption failed for {enc_path}: {e}")

if __name__ == "__main__":
    pw = input("Enter decryption password: ")
    for f in os.listdir("."):
        if f.endswith(".enc"):
            decrypt_file(f, pw)
