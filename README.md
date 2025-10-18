Day 1: Bridged from Pandas to NumPy for signal processing. Practiced array creation, vectorization, & slicing. Simulated a noisy sine wave, then filtered it using SciPy & visualized the results with Matplotlib. Project managed with Git.


Day 2:We set up a professional Python project with src and tests folders. After debugging complex configuration issues, we wrote our first passing unit test using pytest, created a virtual environment, and successfully pushed the entire structure to GitHub.


Day 3: Automating with GitLab CI/CD.We'll build a pipeline to automatically lint the code and run all unit tests on every push. This guarantees code quality and stability, professionalizing our development workflow and catching errors early.


Day 4/5:Built a 2-stage GitLab CI/CD pipeline to automate code quality checks. Stage 1 uses flake8 for linting, and Stage 2 uses pytest for unit testing. Proved its value by catching an intentional bug, demonstrating a working automated safety net.

Day 6: Full Automation & Project Completion. Professionalized the local workflow by implementing pre-commit hooks to automatically format (black), lint (flake8), and test (pytest) code before every commit. Completed the analysis pipeline by adding event detection logic, with corresponding unit tests, and wrapped all functionality into a user-friendly Command-Line Interface (CLI).

Day 7: Refactored the project to a modular, driver-based architecture. Integrated real sensor data via a CSV reader and built/tested feature extraction functions (mean, RMS) for windowed signal analysis. Resolved all CI/CD and linter configuration issues.

