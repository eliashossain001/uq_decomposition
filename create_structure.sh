#!/bin/bash

# UAT-Lite ACL Project Structure Generator
echo "Creating UAT-Lite folder structure..."

# Main directories
mkdir -p configs/{models,datasets,experiments}
mkdir -p src/{models,data,training,evaluation,baselines,adversarial,visualization,utils}
mkdir -p scripts
mkdir -p experiments/{general_nlp/{squad,mnli,sst2},clinical/{medqa,pubmedqa,mimic3},ablation,baselines}
mkdir -p results/{tables,figures,analysis}
mkdir -p notebooks
mkdir -p tests
mkdir -p docs
mkdir -p paper/{latex,supplementary}

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/baselines/__init__.py
touch src/adversarial/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "✓ Directory structure created!"
echo "✓ Total directories: $(find . -type d | wc -l)"
echo "✓ Total files: $(find . -type f | wc -l)"
