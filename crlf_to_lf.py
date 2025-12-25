import os

EXTENSIONS = {'.py', '.js', '.ts', '.json', '.md', '.txt', '.yml', '.yaml', '.html', '.css', '.sh', '.toml'}

def convert_to_lf(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
    
    if b'\r\n' in content:
        content = content.replace(b'\r\n', b'\n')
        with open(filepath, 'wb') as f:
            f.write(content)
        print(f"✓ Fixed: {filepath}")
        return True
    return False

fixed_count = 0
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d != '.git']  # Skip .git folder
    
    for file in files:
        if any(file.endswith(ext) for ext in EXTENSIONS):
            filepath = os.path.join(root, file)
            if convert_to_lf(filepath):
                fixed_count += 1

print(f"\nDone! Fixed {fixed_count} files.")
