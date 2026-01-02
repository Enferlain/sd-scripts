import os
import re

scripts_dir = 'd:/Projects/sd-scripts/scripts'
results = []

for f in os.listdir(scripts_dir):
    if f.endswith('.py'):
        filepath = os.path.join(scripts_dir, f)
        with open(filepath, encoding='utf-8') as file:
            content = file.read()
        cfg_count = len(re.findall(r'\bcfg\.', content))
        config_count = len(re.findall(r'\bconfig\.', content))
        results.append((f, cfg_count, config_count))

# Sort by config count descending (scripts still using config. a lot)
results.sort(key=lambda x: x[2], reverse=True)

print("Script Config Usage Audit")
print("=" * 50)
print(f"{'Script':<35} {'cfg.':<8} {'config.':<8}")
print("-" * 50)
for name, cfg, config in results:
    status = "✓" if config == 0 else "⚠️" if config < 10 else "❌"
    print(f"{name:<35} {cfg:<8} {config:<8} {status}")
