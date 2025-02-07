import re
import unicodedata

import ray
from loguru import logger


@ray.remote
class TextNormalizer:
    """Normalizes the text by lower casing it and removing accents and non-english characters"""

    def normalize(self, row: dict[str, str | bool]) -> dict[str, str | bool]:
        """Normalizes the text by lower casing it and removing accents and non-english characters

        Args:
            row (dict[str, str]): The row containing the text to be normalized

        Returns:
            dict[str, str | bool]: The row with the normalized text
        """
        text = row["text"].lower()  # type: ignore
        row["is_filtered_out"] = False
        try:
            text = unicodedata.normalize("NFKD", text)
            text = text.encode("ascii", "ignore").decode("utf-8")
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            row["text"] = text
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            row["is_filtered_out"] = True
        return row
