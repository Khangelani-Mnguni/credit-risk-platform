"""
validators.py

Application input validation.
"""

from app.config import VALIDATION_LIMITS


def validate_inputs(applicant: dict):
    """
    Validate applicant values.

    Raises
    ------
    ValueError
        If any value is outside acceptable limits.
    """

    for feature, limits in VALIDATION_LIMITS.items():

        if feature not in applicant:
            continue

        minimum, maximum = limits

        value = applicant[feature]

        if value < minimum or value > maximum:

            raise ValueError(
                f"{feature} must be between {minimum} and {maximum}."
            )

    return True