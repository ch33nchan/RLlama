#!/bin/bash

echo "Creating clean repository structure..."


mkdir -p clean_rllama/rllama/{core,rewards/{components,models},utils,integration,dashboard}


cp rllama/__init__.py clean_rllama/rllama/
cp rllama/engine.py clean_rllama/rllama/core/
cp rllama/agent.py clean_rllama/rllama/
cp rllama/memory.py clean_rllama/rllama/
cp rllama/dashboard.py clean_rllama/rllama/
cp rllama/run_dashboard.py clean_rllama/rllama/


cp rllama/dashboard/*.py clean_rllama/rllama/dashboard/
touch clean_rllama/rllama/dashboard/__init__.py


cp rllama/rewards/*.py clean_rllama/rllama/rewards/
cp -r rllama/rewards/components clean_rllama/rllama/rewards/

cp rllama/integration/*.py clean_rllama/rllama/integration/
touch clean_rllama/rllama/integration/__init__.py


touch clean_rllama/rllama/core/__init__.py
touch clean_rllama/rllama/rewards/models/__init__.py
touch clean_rllama/rllama/utils/__init__.py


if [ -f "rllama/rewards/domain_models.py" ]; then
    cp rllama/rewards/domain_models.py clean_rllama/rllama/rewards/models/
fi


if [ -f "rllama/rewards/vision_rewards.py" ]; then
    cp rllama/rewards/vision_rewards.py clean_rllama/rllama/rewards/models/
fi


cp rllama/utils/config_loader.py clean_rllama/rllama/utils/

touch clean_rllama/rllama/rewards/__init__.py
touch clean_rllama/rllama/rewards/components/__init__.py


cp LICENSE clean_rllama/
cp README.md clean_rllama/
cp pyproject.toml clean_rllama/

echo "Clean repository structure created in clean_rllama/"
echo "Verify the contents before replacing the original."
