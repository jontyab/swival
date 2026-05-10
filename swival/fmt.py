"""ANSI-formatted stderr output using Rich."""

import contextlib
import difflib
import os
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.spinner import Spinner
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

_console = Console(stderr=True)
_stdout_console = Console(stderr=False)

_think_count = 0


def reset_state() -> None:
    """Reset all module-level rendering state (think tree counter, etc.)."""
    global _think_count
    _think_count = 0


def init(*, color: bool = False, no_color: bool = False) -> None:
    """Reconfigure the module-level console from CLI flags.

    Call once at startup, before any output.
    """
    global _console, _stdout_console
    kwargs: dict = {"stderr": True}
    stdout_kwargs: dict = {"stderr": False}
    if color:
        kwargs["force_terminal"] = True
        kwargs["no_color"] = False
    if no_color:
        kwargs["no_color"] = True
        stdout_kwargs["no_color"] = True
    _console = Console(**kwargs)
    _stdout_console = Console(**stdout_kwargs)


# -- Turn structure ----------------------------------------------------------


_TURN_GRADIENT = [
    (0, 180, 220),  # cyan
    (60, 120, 220),  # blue
    (160, 80, 200),  # magenta
]


class _GradientRule:
    """A horizontal rule with a gradient color ramp and centered title."""

    def __init__(self, title: str):
        self.title = title

    def __rich_console__(self, console, options):
        width = options.max_width
        title = f" {self.title} "
        side = max((width - len(title)) // 2, 0)
        text = Text()
        for i in range(side):
            t = i / max(width - 1, 1)
            r, g, b = _lerp_color(_TURN_GRADIENT, t)
            text.append("\u2500", style=Style(color=f"rgb({r},{g},{b})"))
        text.append(title, style=Style(bold=True, color="white"))
        for i in range(side + len(title), width):
            t = i / max(width - 1, 1)
            r, g, b = _lerp_color(_TURN_GRADIENT, t)
            text.append("\u2500", style=Style(color=f"rgb({r},{g},{b})"))
        yield from text.__rich_console__(console, options)


def turn_header(n: int, max_n: int, token_est: int) -> None:
    reset_state()
    _console.print()
    title = f"Turn {n}/{max_n} (~{token_est} tokens)"
    if _console.is_terminal:
        _console.print(_GradientRule(title))
    else:
        _console.print(Rule(title, style="cyan"))


def llm_timing(elapsed: float, finish_reason: str) -> None:
    style = "green" if finish_reason == "stop" else "yellow"
    text = Text()
    text.append(f"  LLM responded in {elapsed:.1f}s", style=style)
    text.append(f"  finish_reason={escape(str(finish_reason))}", style=style)
    _console.print(text)


_SPINNER_PHASES: list[tuple[float, str, str, str]] = [
    # (min_seconds, spinner_name, style, verb)
    (0, "dots", "bright_cyan", "Thinking"),
    (2, "arc", "cyan", "Reasoning"),
    (5, "bouncingBall", "bright_blue", "Analyzing"),
    (10, "moon", "blue", "Composing"),
    (18, "dots3", "bright_magenta", "Elaborating"),
    (28, "earth", "magenta", "Synthesizing"),
    (40, "clock", "bright_cyan", "Polishing"),
]


class _PhaseSpinner:
    """Phase-cycling spinner on stderr.  Call start()/stop() to control."""

    def __init__(self, label: str = "Thinking"):
        suffix = ""
        if "(" in label:
            idx = label.index("(")
            suffix = " " + label[idx:].strip()
            initial_desc = f"{_SPINNER_PHASES[0][3]}{suffix}"
        else:
            initial_desc = label
        self.suffix = suffix

        self._spinner_col = SpinnerColumn("dots", style="cyan", speed=2.0)
        self._progress = Progress(
            self._spinner_col,
            TextColumn("  {task.description}"),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
            refresh_per_second=16,
        )
        self._initial_desc = initial_desc
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._progress.start()
        tid = self._progress.add_task(self._initial_desc, total=None)
        suffix = self.suffix
        spinner_col = self._spinner_col
        progress = self._progress
        stop = self._stop_event

        def _cycle(task_id):
            t0 = time.monotonic()
            phase_idx = 0
            while not stop.wait(0.3):
                elapsed = time.monotonic() - t0
                new_idx = phase_idx
                for i, (threshold, _, _, _) in enumerate(_SPINNER_PHASES):
                    if elapsed >= threshold:
                        new_idx = i
                if new_idx != phase_idx:
                    phase_idx = new_idx
                    _, name, style, verb = _SPINNER_PHASES[phase_idx]
                    spinner_col.spinner = Spinner(name, style=style, speed=2.0)
                    progress.update(task_id, description=f"{verb}{suffix}")

        self._thread = threading.Thread(target=_cycle, args=(tid,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        self._progress.stop()


@contextlib.contextmanager
def llm_spinner(label: str = "Thinking"):
    """Context manager showing a phase-cycling spinner on stderr."""
    if not _console.is_terminal:
        yield
        return

    spinner = _PhaseSpinner(label)
    spinner.start()
    try:
        yield
    finally:
        spinner.stop()


def completion(turns: int, exit_code: str, elapsed: float | None = None) -> None:
    timing = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    if exit_code == "ok":
        _console.print(
            Text(f"  \u2713 Agent finished: {turns} turns{timing}", style="bold green")
        )
    else:
        _console.print(
            Text(
                f"  Agent finished: {turns} turns{timing}, exit={exit_code}",
                style="bold red",
            )
        )


# -- Tool calls --------------------------------------------------------------


class _ToolLine:
    """Live spinner for an in-progress tool call on TTY."""

    def __init__(self, name: str):
        self._name = name
        self._live = Live(
            self._render_spinning(name),
            console=_console,
            transient=True,
            refresh_per_second=12,
        )
        self._live.start()

    @staticmethod
    def _render_spinning(name: str) -> Text:
        text = Text()
        text.append("  \u25b6 ", style="bold magenta")
        text.append(name, style="bold magenta")
        text.append("  \u2026", style="dim")
        return text

    def finish(self, success: bool, detail: str, elapsed: float) -> None:
        self._live.stop()
        header = Text()
        if success:
            header.append(f"  \u2713 {self._name}", style="green")
            header.append(f"  {elapsed:.1f}s", style="green")
        else:
            header.append(f"  \u2717 {self._name}", style="bold red")
            header.append(f"  {detail}", style="red")
        _console.print(header)


def tool_call(name: str, args_json: str) -> "_ToolLine | None":
    """Print tool invocation; returns a live handle on TTY."""
    if _console.is_terminal:
        return _ToolLine(name)
    header = Text()
    header.append("  \u25b6 ", style="bold magenta")
    header.append(name, style="bold magenta")
    _console.print(header)
    if args_json:
        for line in args_json.splitlines():
            _console.print(Text(f"    {line}", style="dim"))
    return None


def tool_result(
    name: str,
    elapsed: float,
    preview: str,
    handle: "_ToolLine | None" = None,
) -> None:
    if handle:
        handle.finish(True, f"{elapsed:.1f}s", elapsed)
        if preview:
            _console.print(Text(f"    {preview}", style="dim"))
        return
    header = Text()
    header.append(f"  \u2713 {name}", style="green")
    header.append(f"  {elapsed:.1f}s", style="green")
    _console.print(header)
    if preview:
        _console.print(Text(f"    {preview}", style="dim"))


_DIFF_MAX_LINES = 50
_DIFF_MAX_BYTES = 4096


def tool_diff(file_path: str, old: str, new: str) -> None:
    """Print a colored unified diff of an edit to stderr."""
    diff_lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )
    if not diff_lines:
        return

    additions = sum(
        1 for dl in diff_lines if dl.startswith("+") and not dl.startswith("+++")
    )
    deletions = sum(
        1 for dl in diff_lines if dl.startswith("-") and not dl.startswith("---")
    )

    output = Text()
    total_bytes = 0
    shown = 0
    for line in diff_lines:
        if shown >= _DIFF_MAX_LINES or total_bytes >= _DIFF_MAX_BYTES:
            remaining = len(diff_lines) - shown
            output.append(f"... {remaining} more lines\n", style="dim")
            break
        if line.startswith("---") or line.startswith("+++"):
            style = "bold"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("+"):
            style = "green"
        else:
            style = "dim"
        encoded = line.encode("utf-8")
        budget = _DIFF_MAX_BYTES - total_bytes
        if len(encoded) > budget:
            encoded = encoded[:budget]
            line = encoded.decode("utf-8", errors="ignore")
        display = line if _console.is_terminal else f"    {line}"
        output.append(display, style=style)
        if not display.endswith("\n"):
            output.append("\n")
        total_bytes += len(encoded)
        shown += 1

    if _console.is_terminal:
        subtitle = Text()
        subtitle.append(f"+{additions}", style="green")
        subtitle.append(" / ", style="dim")
        subtitle.append(f"-{deletions}", style="red")
        panel = Panel(
            output,
            title=file_path,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style="dim",
            padding=(0, 1),
        )
        _console.print(panel)
    else:
        _console.print(output, end="")


def tool_error(
    name: str,
    msg: str,
    handle: "_ToolLine | None" = None,
) -> None:
    if handle:
        handle.finish(False, msg, 0)
        return
    header = Text()
    header.append(f"  \u2717 {name}", style="bold red")
    header.append(f"  {msg}", style="red")
    _console.print(header)


def tool_repair(name: str, repairs: list[dict]) -> None:
    for r in repairs:
        line = Text()
        line.append(f"  ~ {name}", style="bold yellow")
        line.append(f"  repaired: {r['type']} on {r.get('field', '?')}", style="yellow")
        _console.print(line)


def guardrail(tool_name: str, count: int, error: str) -> None:
    line = Text()
    line.append("  \u26a0 Guardrail: ", style="bold yellow")
    line.append(
        f"{tool_name} repeated the same error {count} times. Last error: {error}",
        style="yellow",
    )
    _console.print(line)


# -- Think steps -------------------------------------------------------------


_BRANCH_COLORS = ["cyan", "blue", "magenta"]


def think_step(
    number: int,
    total: int,
    text: str,
    *,
    is_revision: bool = False,
    revises_thought: int | None = None,
    branch_id: str | None = None,
    branch_from_thought: int | None = None,
) -> None:
    global _think_count

    if _think_count == 0:
        _console.print(Text("  [think]", style="yellow"))
    _think_count += 1

    line = Text()
    if is_revision and revises_thought is not None:
        line.append("  \u2502  \u2514\u2500 ", style="yellow")
        line.append(f"rev: {text}", style="dim italic")
    elif branch_id is not None and branch_from_thought is not None:
        color = _BRANCH_COLORS[hash(branch_id) % len(_BRANCH_COLORS)]
        line.append(f"  \u251c\u2500 {_think_count}. ", style=color)
        line.append(f"[branch:{branch_id}] ", style=color)
        line.append(text, style="dim italic")
    else:
        line.append(f"  \u251c\u2500 {_think_count}. ", style="yellow")
        line.append(text, style="dim italic")
    _console.print(line)


# -- Todo updates ------------------------------------------------------------


def todo_update(action: str, detail: str) -> None:
    prefix_map = {"add": "+1", "done": "\u2713", "remove": "-1", "cleared": "cleared"}
    tag = prefix_map.get(action, action)
    line = Text()
    line.append(f"  [todo {tag}]", style="yellow")
    line.append(f" {detail}", style="dim italic")
    _console.print(line)


_PROGRESS_STOPS = [(220, 50, 50), (220, 180, 50), (50, 220, 50)]


def _progress_bar_color(ratio: float) -> str:
    """Return an RGB color interpolated from red (0%) to yellow (50%) to green (100%)."""
    r, g, b = _lerp_color(_PROGRESS_STOPS, ratio)
    return f"rgb({r},{g},{b})"


def todo_list(
    items: list,
    action: str | None = None,
    changed_task: str | None = None,
    note: str | None = None,
) -> None:
    """Render the full todo checklist with an optional action annotation."""
    total = len(items)
    done = sum(1 for i in items if i.done)
    header = Text()
    header.append("  [todo]", style="yellow")
    header.append(f" {done}/{total}", style="dim")

    if total > 0:
        ratio = done / total
        bar_width = 10
        filled = int(ratio * bar_width)
        empty = bar_width - filled
        color = _progress_bar_color(ratio)
        header.append(" ", style="")
        header.append("\u2588" * filled, style=color)
        header.append("\u2591" * empty, style="dim")
        pct = int(ratio * 100)
        header.append(f" {pct}%", style="dim")
        if done == total:
            header.append(" done!", style="bold green")

    if note:
        header.append(f"  ({note})", style="dim italic")
    _console.print(header)
    for item in items:
        line = Text()
        is_changed = changed_task is not None and item.text == changed_task
        if item.done:
            line.append("  \u2611 ", style="dim")
            line.append(item.text, style="bold dim" if is_changed else "dim")
        else:
            line.append("  \u2610 ", style="")
            line.append(item.text, style="bold" if is_changed else "")
        _console.print(line)


# -- Assistant text ----------------------------------------------------------

_ASSISTANT_MAX_LINES = 100


class _LeftBar:
    """Renders a child renderable with a blue left-border bar (│)."""

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        inner_width = max(options.max_width - 4, 20)
        inner_options = options.update_width(inner_width)
        lines = console.render_lines(self.renderable, inner_options, pad=False)
        bar = Segment("  │ ", Style(color="blue"))
        newline = Segment("\n")
        for line in lines:
            yield bar
            yield from line
            yield newline


def assistant_text(text: str) -> None:
    src_lines = text.split("\n")
    if len(src_lines) > _ASSISTANT_MAX_LINES:
        remaining = len(src_lines) - _ASSISTANT_MAX_LINES
        text = "\n".join(src_lines[:_ASSISTANT_MAX_LINES])
        md = Markdown(text)
        _console.print(_LeftBar(md), end="")
        _console.print(
            Text(f"  │ ... {remaining} more lines (truncated)", style="blue dim")
        )
    else:
        md = Markdown(text)
        _console.print(_LeftBar(md), end="")


def repl_answer(text: str) -> None:
    """Print a REPL answer to stdout, with syntax highlighting when on a TTY."""
    if _stdout_console.is_terminal and not _stdout_console.no_color:
        from rich.syntax import Syntax

        highlighted = Syntax(
            text,
            "markdown",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
        _stdout_console.print(highlighted)
    else:
        print(text)


# -- Reviewer feedback -------------------------------------------------------


def review_feedback(review_round: int, text: str) -> None:
    header = Text()
    header.append(f"  [review round {review_round}] ", style="bold magenta")
    header.append("Reviewer requested changes:", style="magenta")
    _console.print(header)
    for line in text.splitlines():
        _console.print(Text(f"    {line}", style="magenta"))


def review_sending(review_round: int) -> None:
    _console.print(
        Text(
            f"  ▶ Review round {review_round}: sending answer to reviewer",
            style="bold cyan",
        )
    )


def review_accepted(review_round: int) -> None:
    _console.print(
        Text(
            f"  \u2713 Reviewer accepted the answer (round {review_round})",
            style="bold green",
        )
    )


# -- Diagnostics -------------------------------------------------------------


def info(msg: str) -> None:
    _console.print(Text(f"  {msg}", style="dim"))


model_info = info


def context_stats(label: str, tokens: int) -> None:
    _console.print(Text(f"  {label}: ~{tokens} tokens", style="dim"))


def think_summary(line: str) -> None:
    _console.print(Text(f"  {line}", style="dim"))


todo_summary = think_summary


def warning(msg: str) -> None:
    line = Text()
    line.append("  \u26a0 Warning: ", style="yellow")
    line.append(msg, style="yellow")
    _console.print(line)


def error(msg: str) -> None:
    if _console.is_terminal:
        _console.print(
            Panel(
                Text(msg, style="red"),
                title="Error",
                title_align="left",
                border_style="bold red",
                padding=(0, 1),
            )
        )
    else:
        line = Text()
        line.append("Error: ", style="bold red")
        line.append(msg, style="red")
        _console.print(line)


sandbox_hint = info


def quick_shell(cmd: str, returncode: int, output: str) -> None:
    header = Text()
    header.append(f"  $ {cmd}", style="bold dim")
    _console.print(header)
    if output:
        _console.print(output)
    if returncode != 0:
        _console.print(Text(f"  exit {returncode}", style="red dim"))


def repl_banner() -> None:
    _console.print(Text("Interactive mode. Type /exit or Ctrl-D to quit.", style="dim"))


_LOGO = r"""
 ███ █   █ █ █   █ ███ █
 █   █   █ █ █   █ █ █ █
 ███ █ █ █ █  █ █  ███ █
   █ ██ ██ █  █ █  █ █ █
 ███ █   █ █   █   █ █ ███
""".strip("\n")

_GRADIENT_STOPS = [
    (0, 80, 220),  # cyan
    (120, 60, 220),  # blue-purple
    (200, 50, 200),  # magenta
    (220, 160, 40),  # yellow
]


def _lerp_color(stops: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    """Interpolate between color stops at position t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    n = len(stops) - 1
    idx = min(int(t * n), n - 1)
    local_t = (t * n) - idx
    r0, g0, b0 = stops[idx]
    r1, g1, b1 = stops[idx + 1]
    return (
        int(r0 + (r1 - r0) * local_t),
        int(g0 + (g1 - g0) * local_t),
        int(b0 + (b1 - b0) * local_t),
    )


def repl_splash(
    model: str = "",
    provider: str = "",
    workspace: str = "",
) -> None:
    """Print a colorful startup splash banner to stderr."""
    if not _console.is_terminal:
        return

    logo_lines = _LOGO.split("\n")
    max_len = max(len(ln) for ln in logo_lines)
    text = Text()
    for row_idx, row in enumerate(logo_lines):
        padded = row.ljust(max_len)
        for col_idx, ch in enumerate(padded):
            t = col_idx / max(max_len - 1, 1)
            r, g, b = _lerp_color(_GRADIENT_STOPS, t)
            text.append(ch, style=Style(color=f"rgb({r},{g},{b})", bold=True))
        text.append("\n")

    _console.print()
    _console.print(text, end="")
    _console.print(Text("  https://swival.dev", style="dim"))

    if model or provider or workspace:
        info_line = Text()
        if model:
            info_line.append(f"  model: {model}", style="dim")
        if provider:
            if model:
                info_line.append(" · ", style="dim")
            info_line.append(f"provider: {provider}", style="dim")
        if workspace:
            if model or provider:
                info_line.append(" · ", style="dim")
            info_line.append(f"workspace: {workspace}", style="dim")
        _console.print(info_line)

    grad_rule = Text()
    width = _console.width or 80
    for i in range(width):
        t = i / max(width - 1, 1)
        r, g, b = _lerp_color(_GRADIENT_STOPS, t)
        grad_rule.append("─", style=Style(color=f"rgb({r},{g},{b})"))
    _console.print(grad_rule)


def stderr_is_terminal() -> bool:
    """True when stderr is a TTY."""
    return _console.is_terminal


class _BlinkingCursor:
    """Renderable that appends a blinking block cursor to streamed text.

    The blink is time-driven: __rich_console__ reads time.monotonic() and
    toggles the cursor at ~2 Hz.  The Live refresh loop (12 fps) drives
    updates — no threads or sleeps needed.
    """

    def __init__(self, text_lines: str, style: str = "dim"):
        self._text = text_lines
        self._style = style

    def __rich_console__(self, console, options):
        show_cursor = int(time.monotonic() * 2) % 2 == 0
        display = self._text + ("\u258c" if show_cursor else " ")
        yield from Text(display, style=self._style).__rich_console__(console, options)


def _detect_code_fence(text: str) -> tuple[str | None, str | None]:
    """If the text has an open code fence, return (lang, code_content)."""
    lines = text.split("\n")
    fence_start = None
    lang = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") and fence_start is None:
            lang = stripped[3:].strip() or None
            fence_start = i
        elif stripped == "```" and fence_start is not None:
            fence_start = None
            lang = None
    if fence_start is not None:
        code_content = "\n".join(lines[fence_start + 1 :])
        return lang, code_content
    return None, None


class streaming_preview:
    """Spinner until first chunk, then live dim streamed-text preview.

    Used as a context manager.  While active, call ``update(delta)`` to
    append text and ``reset()`` to clear the buffer (e.g. before a retry).
    On exit whichever display is active (spinner or Live) is torn down.
    """

    def __init__(self, label: str = "Thinking"):
        self._label = label
        self._buf: list[str] = []
        self._spinner = _PhaseSpinner(label)
        self._live: Live | None = None
        self._cached_renderable = None
        try:
            self._max_lines = os.get_terminal_size(2).lines - 2
        except OSError:
            self._max_lines = 24

    def __enter__(self):
        self._spinner.start()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.stop()
            self._live = None
        else:
            self._spinner.stop()
        return False

    def update(self, delta: str) -> None:
        if not delta:
            return
        if self._live is None:
            self._spinner.stop()
            self._live = Live(
                Text(""),
                console=_console,
                transient=True,
                refresh_per_second=12,
            )
            self._live.start()
        self._buf.append(delta)
        text = "".join(self._buf)
        lines = text.split("\n")
        if len(lines) > self._max_lines:
            lines = lines[-self._max_lines :]
        visible = "\n".join(lines)

        if _console.is_terminal:
            if "\n" in delta:
                lang, code = _detect_code_fence(visible)
                if lang and code:
                    from rich.syntax import Syntax

                    self._cached_renderable = Syntax(
                        code,
                        lang,
                        theme="ansi_dark",
                        background_color="default",
                        word_wrap=True,
                    )
                else:
                    self._cached_renderable = None
            if self._cached_renderable is None:
                renderable = _BlinkingCursor(visible)
            else:
                renderable = self._cached_renderable
            self._live.update(renderable)
        else:
            self._live.update(Text(visible, style="dim"))

    def reset(self) -> None:
        self._buf.clear()
        self._cached_renderable = None
        if self._live is not None:
            self._live.stop()
            self._live = None
        else:
            self._spinner.stop()
        self._spinner = _PhaseSpinner(self._label)
        self._spinner.start()


# -- External servers (MCP / A2A) --------------------------------------------


def _server_start(kind: str, name: str, tool_count: int) -> None:
    line = Text()
    line.append(f"  {kind} {name}", style="cyan")
    line.append(f"  {tool_count} tool(s)", style="dim")
    _console.print(line)


def _server_error(kind: str, name: str, error: str) -> None:
    line = Text()
    line.append(f"  {kind} {name}", style="bold red")
    line.append(f"  {error}", style="red")
    _console.print(line)


def mcp_server_start(name: str, tool_count: int) -> None:
    _server_start("MCP", name, tool_count)


def mcp_server_error(name: str, error: str) -> None:
    _server_error("MCP", name, error)


def a2a_server_start(name: str, tool_count: int) -> None:
    _server_start("A2A", name, tool_count)


def a2a_server_error(name: str, error: str) -> None:
    _server_error("A2A", name, error)
