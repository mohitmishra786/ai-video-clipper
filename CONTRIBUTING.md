# Contributing to AI Video Clipper

First off, thank you for considering contributing to AI Video Clipper!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-video-clipper.git
   cd ai-video-clipper
   ```
3. **Set up the development environment** (see below)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

- **Bug fixes** - Fix issues and improve stability
- **New features** - Add new functionality
- **Documentation** - Improve docs, add examples
- **UI/UX improvements** - Enhance the user interface
- **Performance** - Optimize processing speed
- **Tests** - Add or improve tests

### Good First Issues

Look for issues labeled `good first issue` - these are great for newcomers!

## Development Setup

### Prerequisites

- Python 3.11 or higher
- FFmpeg installed (`brew install ffmpeg` on macOS)
- Git

### Setup Steps

```bash
# 1. Clone and enter directory
git clone https://github.com/mohitmishra786/ai-video-clipper.git
cd ai-video-clipper

# 2. Create virtual environment
python3.11 -m venv venv311
source venv311/bin/activate  # On Windows: venv311\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the development server
python app.py
```

The app will be available at `http://localhost:5000` with auto-reload enabled.

## Code Style

### Python

- Follow **PEP 8** style guidelines
- Use **meaningful variable names**
- Add **docstrings** to functions:
  ```python
  def analyze_transcript(segments, keyword=None, num_clips=5, logger=None):
      """
      Analyze transcript to find the most engaging clips.
      
      Args:
          segments: List of transcription segments
          keyword: Optional keyword to boost matching segments
          num_clips: Number of clips to extract
          logger: Logger instance
      
      Returns:
          List of clip dictionaries with start, end, title, description
      """
  ```
- Use **type hints** where helpful
- Keep functions focused and under 50 lines when possible

### HTML/CSS

- Use **semantic HTML5** elements
- Follow **BEM naming convention** for CSS classes
- Keep CSS organized with clear comments

### JavaScript

- Use **ES6+ syntax**
- Use **const/let** instead of var
- Add comments for complex logic

## Pull Request Process

### Before Submitting

1. **Test your changes** locally
2. **Update documentation** if needed
3. **Add comments** for complex code
4. **Run the linter** (if available)

### PR Guidelines

1. **Create a descriptive PR title**:
   - Good: `feat: add subtitle overlay option`
   - Good: `fix: resolve clip ending mid-sentence`
   - Bad: `Update app.py`

2. **Fill out the PR template** (if provided)

3. **Link related issues**:
   ```
   Fixes #123
   Related to #456
   ```

4. **Keep PRs focused** - One feature/fix per PR

5. **Be responsive** to review feedback

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add keyword-based clip filtering
fix: prevent clips from cutting mid-word
docs: update installation instructions
refactor: simplify transcription pipeline
perf: optimize FFmpeg encoding settings
```

## Reporting Bugs

### Before Reporting

1. **Search existing issues** - It might already be reported
2. **Try the latest version** - The bug might be fixed
3. **Check the logs** - Look in the `logs/` directory

### Bug Report Template

```markdown
**Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS 14.0]
- Python version: [e.g., 3.11]
- Browser: [e.g., Chrome 120]

**Logs**
```
Paste relevant log output here
```

**Screenshots**
If applicable, add screenshots.
```

## Suggesting Features

### Feature Request Template

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How would you like it to work?

**Alternatives Considered**
Any alternative solutions you've thought about.

**Additional Context**
Any other context or screenshots.
```

## Questions?

- Open a [GitHub Discussion](https://github.com/mohitmishra786/ai-video-clipper/discussions)
- Check existing issues and discussions first

---

Thank you for contributing!
