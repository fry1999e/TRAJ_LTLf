from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import cli
from llm.errors import LLMClientError
from llm.langchain_client import LangChainJSONClient
from terminal_progress import ProgressConfig, create_progress, progress_session
from tests.test_manual_release import MANUAL_ARGS


class TTYBuffer(io.StringIO):
    def __init__(self):
        super().__init__()
        self.updated_frame = threading.Event()

    def isatty(self):
        return True

    def write(self, text):
        result = super().write(text)
        if text == "\r/":
            self.updated_frame.set()
        return result


@contextmanager
def progress_fixture(mode="auto", *, tty=True):
    stream = TTYBuffer() if tty else io.StringIO()
    with redirect_stdout(stream), progress_session():
        progress = create_progress(mode)
        # Keep synchronization event driven, with a short animation interval.
        progress.config = ProgressConfig(mode=mode, spinner_interval_seconds=0.01)
        yield progress, stream


def fixture_client(**kwargs):
    return LangChainJSONClient(api_key="private-fixture-key", model="fixture", **kwargs)


class WaitingDisplayTests(unittest.TestCase):
    def test_spinner_animates_during_blocking_call_and_stops_before_human_input(self):
        with progress_fixture() as (progress, stream):
            threads = []

            def delayed_reply(*_args, **_kwargs):
                threads.append(progress._spinner_thread)
                self.assertTrue(threads[-1].is_alive())
                self.assertTrue(stream.updated_frame.wait(timeout=2))
                self.assertIn("\r/", stream.getvalue())
                return '{"ok": true}'

            with patch.object(LangChainJSONClient, "_invoke_chat_model", side_effect=delayed_reply):
                self.assertEqual(fixture_client().complete_json("fixture", {
                    "task": "lossless_surface_segmentation",
                    "source_text": "private-fixture-source",
                }), {"ok": True})
            self.assertIsNone(progress._spinner_thread)
            self.assertFalse(threads[0].is_alive())
            at_prompt = stream.getvalue()
            self.assertIn("\r|", at_prompt)
            self.assertNotIn("等待模型响应", at_prompt)
            self.assertNotIn("已等待", at_prompt)
            self.assertNotIn("生成需求切分", at_prompt)
            self.assertRegex(at_prompt, r"^(?:\r[|/\\-]|\r +\r)+$")
            self.assertNotIn("private-fixture", at_prompt)
            last_frame, cleared_line, _ = at_prompt.rsplit("\r", 2)
            last_frame = last_frame.rsplit("\r", 1)[-1]
            self.assertEqual(cleared_line, " " * len(last_frame))

            def user_input(_prompt):
                self.assertIsNone(progress._spinner_thread)
                self.assertEqual(stream.getvalue(), at_prompt)
                return "a"

            with patch("builtins.input", side_effect=user_input):
                self.assertEqual(input("人工审核: "), "a")

    def test_chinese_stage_spinner_clears_full_terminal_width(self):
        with progress_fixture() as (progress, stream):
            with progress.task("读取文件"):
                self.assertTrue(progress._spinner_thread.is_alive())
            frames, cleared_line, _ = stream.getvalue().rsplit("\r", 2)
            last_frame = frames.rsplit("\r", 1)[-1]
            self.assertGreater(len(cleared_line), len(last_frame))

    def test_non_tty_and_plain_use_static_wait_lines(self):
        for mode, tty in (("auto", False), ("plain", True)):
            with self.subTest(mode=mode, tty=tty), progress_fixture(mode, tty=tty) as (progress, stream):
                with patch.object(LangChainJSONClient, "_invoke_chat_model", return_value='{}'):
                    fixture_client().complete_json("fixture", {})
                self.assertEqual(stream.getvalue(), "...\n")
                self.assertNotIn("\r", stream.getvalue())
                self.assertIsNone(progress._spinner_thread)

    def test_off_is_silent(self):
        with progress_fixture("off") as (progress, stream):
            with patch.object(LangChainJSONClient, "_invoke_chat_model", return_value='{}'):
                fixture_client().complete_json("fixture", {})
            self.assertEqual(stream.getvalue(), "")
            self.assertIsNone(progress._spinner_thread)

    def test_error_and_ctrl_c_stop_spinner(self):
        for error in (LLMClientError("fixture failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), progress_fixture() as (progress, _stream):
                threads = []

                def fail(*_args, **_kwargs):
                    threads.append(progress._spinner_thread)
                    raise error

                with patch.object(LangChainJSONClient, "_invoke_chat_model", side_effect=fail):
                    with self.assertRaises(type(error)):
                        fixture_client(single_physical_request=True).complete_json("fixture", {})
                self.assertIsNone(progress._spinner_thread)
                self.assertFalse(threads[0].is_alive())

    def test_spinner_remains_active_during_transport_retry_and_json_repair(self):
        with progress_fixture() as (progress, stream):
            # One transport failure followed by malformed JSON and JSON repair.
            chat = Mock()
            chat.invoke.side_effect = [TimeoutError("timed out"), "not JSON", '{"ok": true}']
            workers = []

            def active_during_backoff(*_args):
                worker = progress._spinner_thread
                self.assertTrue(worker.is_alive())
                workers.append(worker)

            def active_chat(**_kwargs):
                active_during_backoff()
                return chat

            with (
                patch.object(LangChainJSONClient, "_build_chat_model", side_effect=active_chat),
                patch("llm.langchain_client._sleep_before_retry", side_effect=active_during_backoff) as backoff,
            ):
                result = fixture_client(response_format_mode="none").complete_json("fixture", {})
            self.assertEqual(result, {"ok": True})
            self.assertEqual(chat.invoke.call_count, 3)
            backoff.assert_called_once()
            self.assertTrue(all(worker is workers[0] for worker in workers))
            self.assertFalse(workers[0].is_alive())
            self.assertEqual(stream.getvalue().count("\r|"), 1)

    def test_session_cleans_unfinished_stage_and_does_not_leak_to_library_calls(self):
        stream = TTYBuffer()
        with redirect_stdout(stream):
            with self.assertRaises(KeyboardInterrupt):
                with progress_session():
                    progress = create_progress()
                    progress.task_start("fixture stage")
                    worker = progress._spinner_thread
                    raise KeyboardInterrupt()
            self.assertFalse(worker.is_alive())
            self.assertIsNone(progress._spinner_thread)
            previous = stream.getvalue()
            with patch.object(LangChainJSONClient, "_invoke_chat_model", return_value='{}'):
                fixture_client().complete_json("fixture", {})
            self.assertEqual(stream.getvalue(), previous)


class ManualCommandWaitingTests(unittest.TestCase):
    def test_default_file_single_and_explicit_batch_commands_enable_waiting(self):
        for extra in ([], ["Whenever request occurs, grant should eventually occur."],
                      ["--requirements-file", "data/input/A72.xlsx"]):
            with self.subTest(extra=extra):
                stream = TTYBuffer()
                observed = []

                def pipeline_fixture(*_args, **kwargs):
                    progress = kwargs["progress"]
                    observed.append(progress)

                    def reply(*_args, **_kwargs):
                        self.assertTrue(progress._spinner_thread.is_alive())
                        self.assertIn("\r|", stream.getvalue())
                        self.assertNotIn("等待模型响应", stream.getvalue())
                        return '{}'

                    with patch.object(LangChainJSONClient, "_invoke_chat_model", side_effect=reply):
                        fixture_client().complete_json("fixture", {"task": "extract_source_ap_candidates"})
                    self.assertIsNone(progress._spinner_thread)
                    # Stop at the real entry boundary without output artifacts.
                    raise KeyboardInterrupt()

                with (
                    patch.object(sys, "argv", ["main.py", *MANUAL_ARGS, *extra]),
                    patch.object(sys.stdin, "isatty", return_value=True),
                    patch.object(cli, "prepare_checkpoint_run", return_value=None),
                    patch.object(cli, "_prepare_artifacts"),
                    patch.object(cli, "run_traj_ltlf_pipeline", side_effect=pipeline_fixture),
                    patch.object(cli, "run_batch_traj_ltlf_pipeline", side_effect=pipeline_fixture),
                    redirect_stdout(stream), redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        cli.main()
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0].config.mode, "auto")
                self.assertIsNone(observed[0]._spinner_thread)


if __name__ == "__main__":
    unittest.main()
