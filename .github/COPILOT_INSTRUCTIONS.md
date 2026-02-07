# Project Documentation & Copilot Guidelines

This file defines the project-specific rules for creating and maintaining documentation, code standards, and repository structure to ensure consistency and high quality.

## 📁 File Naming and Location

### Documentation Files

* **Main Documentation:** `README.md` in the repository root.
  * Contains project overview, installation, and quick start.
  * Maximum of one `README.md` per repository.
* **Changelog:** `CHANGELOG.md` in the repository root.
  * Documents all version changes, migrations, and updates.
  * Format: Markdown with clear versioning.
* **License:** `LICENSE` file in the root directory.

### Code & Assets

* **Core Logic:** `src/` or `app/` (depending on project type).
* **Scripts:** `scripts/` for standalone tools or automation.
* **Tests:** `tests/` directory mirroring the source structure.
* **Archived Code:** `legacy/` folder for deprecated but kept versions.
* **Configuration:** `requirements.txt`, `pyproject.toml`, or `.env.example` in the root.

### Naming Conventions

* **Markdown Files:** ALL_CAPS for `README.md` and `CHANGELOG.md`.
* **Python Modules:** PascalCase or snake_case (maintain project consistency).
* **Directories:** lowercase with underscores or hyphens.
* **Assets/Data:** Descriptive names with underscores.

## 📋 Markdown Structure

### README.md Requirements

* **H1 Title** with Project Name and Version.
* **Introductory Paragraph** (Purpose and Value Proposition).
* **Meta Information** (Author, Date, Version).
* **Table of Contents** (📋).
* **Repository Overview** (🎯 Goals and Scope).
* **Project Structure** (📁) - Represented as an ASCII tree.
* **Quick Start Guide** (🚀).
* **Technology Stack** (📦 Versions and Dependencies).
* **Module/Feature Descriptions** (📚).
* **Installation & Setup** (🔧).
* **License & Contributions** (📝).

### Standard Metadata Block

Use blockquotes for key links and file references at the beginning of sections:

```markdown
> ➡️ **Details:** [Section-Title](#anchor-link)  
> 📖 **Implementation:** [`path/to/file.py`](path/to/file.py)  
> 🔗 **External Resource:** [Title](https://...)
```

### Formatting Rules

* **Emojis:** Use thematically appropriate emojis for headers:
  * 📁 Files/Folders | 🚀 Features/Start | ✅ Success/Done | 🔧 Setup
  * 💾 Code/Modules | 📊 Data/Analysis | ⚠️ Warning | ❓ Questions
* **Links:**
  * Use relative paths for internal files: `[Title](path/to/file)`.
  * Use absolute URLs for external resources.
* **Lists:** Use `-` for unordered and `1.` for ordered steps.

## 💻 Code Standards

### Python Code Blocks

* **Language:** Comments should be in English (or the specified project language), precise, and explanatory.
* **Type Hinting:** Mandatory for function parameters and return types.
* **Docstrings:** Google-style (single-line for simple, multi-line for complex logic).

```python
def process_data(input_path: str) -> bool:
    """
    Brief description of the function.
    
    Args:
        input_path: The filesystem path to the source file.
        
    Returns:
        True if successful, False otherwise.
    """
    # Step 1: Explanation of logic
    result = perform_action()
    
    return result
```

### Shell Blocks (Cross-Platform)

* Use `powershell` for Windows-specific commands.
* Use `bash` for Linux/Mac commands.
* Provide a comment above each command explaining its purpose.

## ✅ Quality Assurance

### Pre-Commit Checklist

* **Markdown Preview:** Ensure headers, tables, and lists render correctly.
* **Broken Links:** Verify all internal (anchors/files) and external links work.
* **Consistency:** Ensure the `README.md` matches the current state of the code.
* **Encoding:** Use UTF-8 for all files.
* **Line Endings:** Use LF (Unix-style).

## 🎯 Summary for AI Assistance

When generating or editing files:

* ✅ Stick to the established folder structure.
* ✅ Follow naming conventions (`README.md`, `snake_case` scripts).
* ✅ Use thematic emojis for scannability.
* ✅ Include detailed comments in code blocks.
* ✅ Use metadata blockquotes for file references.
* ✅ Always prioritize clarity for the end-user.

---

Would you like me to generate a template for your `README.md` based on these rules?
