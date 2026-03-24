import os

repo_dir = r"c:\Development\webapps\react\image-lemgendizer-old\training\lemgendary-training-suite"
target_cases = [
    ("LemGendary Training Suite", "LemGendary Training Suite"),
    ("lemgendary-training-suite", "lemgendary-training-suite")
]

for root, _, files in os.walk(repo_dir):
    for f in files:
        if f.endswith(('.yaml', '.py', '.md')):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                try:
                    content = file.read()
                except UnicodeDecodeError:
                    continue
                    
            original = content
            for old, new in target_cases:
                content = content.replace(old, new)
                
            if content != original:
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"✅ Recursively mathematically replaced legacy terminology inside explicitly {path}")
