"""
metadata.py

Model metadata service.
"""

from dataclasses import dataclass


@dataclass
class ModelMetadataService:

    metadata: dict

    @property
    def model_name(self):

        return self.metadata.get(
            "model_name",
            "Logistic Regression",
        )

    @property
    def training_date(self):

        return self.metadata.get(
            "training_date",
            "Unknown",
        )

    @property
    def auc(self):

        return self.metadata.get(
            "roc_auc",
        )

    @property
    def ks(self):

        return self.metadata.get(
            "ks",
        )

    @property
    def gini(self):

        return self.metadata.get(
            "gini",
        )

    @property
    def version(self):

        return self.metadata.get(
            "version",
            "1.0.0",
        )