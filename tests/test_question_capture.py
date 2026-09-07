"""Question capture uses synthetic Git repositories, never the real orga tree."""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/q"


class QuestionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="q-capture-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "orga"
        self.repo.mkdir()
        self.file = self.repo / "membrane/questions-for-marc.md"
        self.file.parent.mkdir()
        self.file.write_text("# Questions\n\n## Open\n\n## Answered\n\nExisting answer.\n")
        self.git("init", "-q")
        self.git("config", "user.name", "Capture test")
        self.git("config", "user.email", "capture@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.hooksPath", "/dev/null")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.script = self.root / "q"
        # Allows the same regression tests to reproduce the historical script's
        # flaws without altering HOME or touching the real question file.
        self.script.write_text(SCRIPT.read_text().replace('orga="$HOME/src/orga"', 'orga="${Q_ORGA_DIR:?}"'))
        self.env = {**os.environ, "Q_ORGA_DIR": str(self.repo)}

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def capture(self, question, **kwargs):
        return subprocess.run(
            ["bash", str(self.script), question],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=10,
            **kwargs,
        )

    def test_literal_backslashes_and_multiline_wording_survive(self):
        question = r"Why do \\n, \t and C:\new\tree differ?" + "\nActual next line."
        result = self.capture(question)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.file.read_text()
        self.assertIn(question + "\n", text)
        self.assertLess(text.index(question), text.index("## Answered"))
        self.assertEqual(self.git("rev-list", "--count", "HEAD").strip(), "2")

    def test_concurrent_capture_waits_through_edit_and_commit(self):
        wrappers = self.root / "bin"
        wrappers.mkdir()
        barrier = self.root / "barrier"
        barrier.mkdir()
        real_cat = shutil.which("cat")
        (wrappers / "cat").write_text(
            "#!/usr/bin/env bash\n"
            'if [ -n "${Q_PAUSE_BEFORE_WRITE:-}" ]; then\n'
            '  touch "$Q_PAUSE_BEFORE_WRITE/ready"\n'
            "  for ((i=0; i<500; i++)); do\n"
            '    [ -e "$Q_PAUSE_BEFORE_WRITE/release" ] && break\n'
            "    sleep 0.01\n"
            "  done\n"
            "fi\n"
            f'exec "{real_cat}" "$@"\n'
        )
        (wrappers / "cat").chmod(0o755)
        base_env = {**self.env, "PATH": f"{wrappers}:{self.env['PATH']}"}
        first = subprocess.Popen(
            ["bash", str(self.script), "first question"],
            cwd=self.root,
            env={**base_env, "Q_PAUSE_BEFORE_WRITE": str(barrier)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        second = None
        try:
            deadline = time.monotonic() + 5
            while not (barrier / "ready").exists():
                if time.monotonic() > deadline:
                    self.fail("first capture did not reach the edit boundary")
                time.sleep(0.01)
            second = subprocess.Popen(
                ["bash", str(self.script), "second question"],
                cwd=self.root,
                env=base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.2)
            blocked = second.poll() is None
            (barrier / "release").touch()
            first_output = first.communicate(timeout=10)
            second_output = second.communicate(timeout=10)
            text = self.file.read_text()
            self.assertIn("first question", text)
            self.assertIn("second question", text)
            self.assertTrue(blocked, "second writer bypassed the capture lock")
            self.assertEqual(first.returncode, 0, first_output)
            self.assertEqual(second.returncode, 0, second_output)
            self.assertEqual(self.git("rev-list", "--count", "HEAD").strip(), "3")
            subjects = self.git("log", "-2", "--format=%s")
            self.assertIn("+ first question", subjects)
            self.assertIn("+ second question", subjects)
        finally:
            (barrier / "release").touch()
            for process in (first, second):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_missing_open_section_does_not_change_file_or_commit(self):
        self.file.write_text("# Questions without a section\n")
        before = self.file.read_bytes()
        head = self.git("rev-parse", "HEAD")
        result = self.capture("where should this go?")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.file.read_bytes(), before)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)


if __name__ == "__main__":
    unittest.main()
