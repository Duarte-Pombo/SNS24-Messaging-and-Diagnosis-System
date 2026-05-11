import unittest

from src.triage_logic import TRIAGE_BY_DIAGNOSIS, triage_diagnosis


class TriageLogicTests(unittest.TestCase):
    def test_every_supported_diagnosis_maps_to_color_and_recommendation(self):
        expected_colors = {
            "Enfarte do Miocárdio": "Red",
            "AVC": "Red",
            "Pneumonia": "Orange",
            "Apendicite": "Orange",
            "Asma (crise)": "Yellow",
            "Fratura Óssea": "Yellow",
            "Gripe": "Green",
            "Gastroenterite": "Green",
            "Enxaqueca": "Green",
            "Infeção Urinária": "Blue",
            "Constipação": "Blue",
        }

        self.assertEqual(set(TRIAGE_BY_DIAGNOSIS), set(expected_colors))

        for diagnosis, expected_color in expected_colors.items():
            color, recommendation = triage_diagnosis(diagnosis)
            self.assertEqual(color, expected_color)
            self.assertIsInstance(recommendation, str)
            self.assertTrue(recommendation)

    def test_english_alias_is_supported(self):
        color, recommendation = triage_diagnosis("Heart Attack")
        self.assertEqual(color, "Red")
        self.assertTrue(recommendation)

    def test_unsupported_diagnosis_raises(self):
        with self.assertRaises(ValueError):
            triage_diagnosis("Unknown condition")


if __name__ == "__main__":
    unittest.main()
