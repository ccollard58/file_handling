import unittest

from tokenharbor_models import parse_models_page, parse_vision_models

PAGE = (
    '<script>self.__next_f.push([1,"{\\"rows\\":[{\\"surface\\":\\"paid-vl\\",\\"priceIn\\":2,'
    '\\"priceOut\\":6,\\"isFree\\":false,\\"inputModalities\\":[\\"text\\",\\"image\\"],'
    '\\"outputModalities\\":\\"$1d:ref\\"},{\\"surface\\":\\"paid-text\\",\\"priceIn\\":1,'
    '\\"priceOut\\":2,\\"isFree\\":false,\\"inputModalities\\":[\\"text\\"],'
    '\\"outputModalities\\":[\\"text\\"]}],\\"freeRows\\":[{\\"surface\\":\\"free-vl:free\\",'
    '\\"priceIn\\":0,\\"priceOut\\":0,\\"isFree\\":true,\\"inputModalities\\":[\\"text\\",'
    '\\"image\\"],\\"outputModalities\\":[\\"text\\"]}],\\"otherRows\\":[]}"])</script>'
)


class TokenHarborModelTests(unittest.TestCase):
    def test_free_vision_models_only_by_default(self):
        rows = parse_models_page(PAGE)
        self.assertEqual([m.model_id for m in parse_vision_models(rows)], ["free-vl:free"])

    def test_paid_vision_models_included_on_request_and_labeled(self):
        models = parse_vision_models(parse_models_page(PAGE), include_paid=True)
        self.assertEqual([m.model_id for m in models], ["free-vl:free", "paid-vl"])
        self.assertIn("[PAID: $2/M input, $6/M output]", models[1].display_name)

    def test_unrecognized_page_raises(self):
        with self.assertRaises(ValueError):
            parse_models_page("<html></html>")


if __name__ == "__main__":
    unittest.main()
