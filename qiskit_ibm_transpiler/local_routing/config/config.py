# -*- coding: utf-8 -*-

# (C) Copyright 2022 IBM. All Rights Reserved.


"""Configuration and other utilities."""

import json
import logging
import os
import re
import time

import jsonschema


class MLConfig:
    def __init__(self, service_name, json_content):
        self.MODELS_PATH = os.getenv(
            f"{service_name}_MODELS_PATH",
            json_content["ML"][service_name]["MODELS_PATH"],
        )
        self.AVAILABLE_MODELS = os.getenv(
            f"{service_name}_AVAILABLE_MODELS",
            json_content["ML"][service_name]["AVAILABLE_MODELS"],
        )


class AITranspilerConfig:
    """
    AITranspilerConfig configuration, read from a .json file.
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        json_filename=os.path.join(os.path.dirname(__file__), "config.json"),
    ):
        if not os.path.isfile(json_filename):
            # In this case we use Python logging instead of our CustomLogger to avoid circular imports
            logging.warning(
                f"WARNING: {json_filename} does not exist. Using default config."
            )
            time.sleep(3)
            json_filename = os.path.join(os.path.dirname(__file__), "config.json")

        self.json_filename = json_filename
        with open(json_filename, "r") as json_file:
            contents = json_file.read()

        # Strip the comments from the file.
        self.json_content = json.loads(
            re.sub("///.*", "", contents, flags=re.MULTILINE)
        )
        jsonschema.validate(self.json_content, schema=CONFIG_SCHEMA)

        # Define ML - Model related parameters
        # Routing
        self.ML_ROUTING_CONFIG = MLConfig(
            service_name="ROUTING_AI", json_content=self.json_content
        )



ML_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "MODELS_PATH": {"type": "string"},
        "AVAILABLE_MODELS": {
            "type": "array",
            "AVAILABLE_MODELS": {
                "properties": {
                    "model_name": {
                        "type": "string",
                    },
                    "qubits": {
                        "type": "integer",
                    },
                    "coupling_map_hash": {
                        "type": "string",
                    },
                    "topology": {
                        "type": "string",
                    },
                    "backends": {
                        "type": "array",
                    },
                    "n_envs": {
                        "type": "integer",
                    },
                    "default_n_steps": {
                        "type": "integer",
                    },
                },
            },
        },
    },
    "required": [
        "MODELS_PATH",
        "AVAILABLE_MODELS",
    ],
}

CONFIG_SCHEMA = {
    "definitions": {
        "ml": {
            "type": "object",
            "properties": {
                "ROUTING_AI": ML_CONFIG_SCHEMA,
                "PERMUTATION_AI": ML_CONFIG_SCHEMA,
                "LINEAR_FUNCTIONS_AI": ML_CONFIG_SCHEMA,
                "CLIFFORD_AI": ML_CONFIG_SCHEMA,
            },
        },
    }
}
