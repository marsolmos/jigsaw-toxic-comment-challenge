Toxic Comment Classifier
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/api
   model
   training
   project_report


How to Work with This Project
=============================

Update the Docs (locally)
--------------------------

1. Edit or add docstrings to your Python source files inside ``src/``.
2. Rebuild the documentation with:

   .. code-block:: bash

      sphinx-build -b html docs docs/_build/html

3. Open ``docs/_build/html/index.html`` in your browser to preview.

Run the FastAPI API Locally
---------------------------

1. Ensure a trained model exists under ``models/latest/``.
2. Install required dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Start the API server:

   .. code-block:: bash

      uvicorn src.api:app --reload

4. Access the API docs at: http://localhost:8000/docs

Train a New Model
-----------------

Run the training script with optional parameters:

.. code-block:: bash

   python src/train.py --model-id "v1" --max-features 10000 --C 2.0

This will train the model, save it under ``models/v1/``, and log metrics if MLflow is enabled.

Debug the Training in VS Code
-----------------------------

1. Open the Run and Debug sidebar in VS Code.
2. Select the configuration named ``Train model (train.py)``.
3. Press ``F5`` to start debugging.

Ensure your ``.vscode/launch.json`` includes:

.. code-block:: json

   {
     "name": "Train model (train.py)",
     "type": "python",
     "request": "launch",
     "program": "${workspaceFolder}/src/train.py",
     "console": "internalConsole",
     "justMyCode": false,
     "args": ["--model-id", "debug"]
   }

Run MLflow UI to Inspect Models
-------------------------------

1. Start MLflow’s web UI locally with:

   .. code-block:: bash

      mlflow ui --port 5000

2. Open your browser and visit: http://localhost:5000

3. Use the UI to:
   - View experiment runs and metrics
   - Compare different model versions
   - Track parameters and artifacts

4. To stop MLflow UI, press ``Ctrl+C`` in the terminal.



API Reference
=============

.. automodule:: model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: train
   :members:
   :undoc-members:
   :show-inheritance:
