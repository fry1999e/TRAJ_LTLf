from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook, load_workbook

from dataset_workbook import workbook_bytes, read_input_workbook
from requirement_loader import load_requirement_records


class XLSXRequirementTests(unittest.TestCase):
    def write_fixture(self, folder, rows, *, headers=("Natural Language", "Requirement ID")):
        path = Path(folder) / "requirements.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
        workbook.save(path)
        workbook.close()
        return path

    def test_preserves_source_unicode_spaces_ids_and_undeclared_ap_profile(self):
        text = "  请求 occurs,\nthen grant eventually occurs.  "
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_fixture(folder, [(text, "A72:REQ-000002")])
            record, = load_requirement_records(path)
        self.assertEqual(record.requirement_id, "A72-REQ-000002")
        self.assertEqual(record.text, text)
        self.assertEqual(record.effective_semantic_source.source_text, text)
        self.assertTrue(record.ap_profile_declared)
        self.assertIsNone(record.declared_ap_catalog)

    def test_rejects_reference_columns_in_online_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_fixture(folder, [("a holds", "REQ-001", "F a")], headers=("Natural Language", "Requirement ID", "Normalized Formula"))
            with self.assertRaisesRegex(ValueError, "参考答案"):
                load_requirement_records(path)

    def test_rejects_additional_hidden_worksheet(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_fixture(folder, [("a holds", "REQ-001")])
            workbook = load_workbook(path)
            workbook.create_sheet("answers").sheet_state = "hidden"
            workbook.save(path)
            workbook.close()
            with self.assertRaisesRegex(ValueError, "一个需求工作表"):
                load_requirement_records(path)

    def test_rejects_duplicate_ids_after_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_fixture(folder, [("a holds", "A72:REQ-000001"), ("b holds", "A72-REQ-000001")])
            with self.assertRaisesRegex(ValueError, "重复"):
                load_requirement_records(path)

    def test_rejects_formula_cells_and_missing_or_numeric_source_values(self):
        for row in [("=1+1", "REQ-001"), (123, "REQ-001"), (None, "REQ-001"), ("a holds", None), ("a holds", "bad id")]:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as folder:
                path = self.write_fixture(folder, [row])
                with self.assertRaises(ValueError):
                    load_requirement_records(path)

    def test_ignores_fully_blank_rows_but_not_empty_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_fixture(folder, [(None, None), ("a holds", "REQ-002")])
            self.assertEqual([r.requirement_id for r in load_requirement_records(path)], ["REQ-002"])

    def test_export_is_repeatable_and_preserves_literal_formula_like_text(self):
        sheets = {"Requirements": [("=this is literal NL", "REQ-001")]}
        first = workbook_bytes(sheets)
        self.assertEqual(first, workbook_bytes(sheets))
        workbook = load_workbook(io.BytesIO(first), data_only=False)
        self.assertEqual(workbook.active["A2"].data_type, "s")
        workbook.close()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "literal.xlsx"
            path.write_bytes(first)
            self.assertEqual(read_input_workbook(path), [("REQ-001", "=this is literal NL")])

    def test_markdown_custom_requirements_remain_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom.md"
            path.write_text("- a holds\n- b holds\n", encoding="utf-8")
            self.assertEqual([r.text for r in load_requirement_records(path)], ["a holds", "b holds"])

    def test_cli_loads_dataset_xlsx_with_stable_ids_before_pipeline(self):
        from tests.test_manual_release import ManualEntryTests, MANUAL_ARGS

        call = ManualEntryTests()._capture_batch([*MANUAL_ARGS, "--requirements-file", "data/input/A72.xlsx"])
        self.assertEqual(len(call.args[0]), 72)
        self.assertEqual(call.kwargs["requirement_ids"][1], "A72-REQ-000002")
        self.assertEqual(call.args[0][1], "Globally it is the case that if a holds then eventually a and b hold.")
