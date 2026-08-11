import os

output_file = "project_codebase.txt"
lines = []

for root, dirs, files in os.walk("."):
    if ".git" in root or "venv" in root or "__pycache__" in root:
        continue
    for file in files:
        if file.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml")) and file != output_file and not file.endswith(".zip"):
            rel_path = os.path.relpath(os.path.join(root, file), ".")
            try:
                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                lines.append(f"========================================\nFILE: {rel_path}\n========================================\n{content}\n\n")
            except Exception as e:
                pass

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Successfully generated {output_file} with {len(lines)} files.")
