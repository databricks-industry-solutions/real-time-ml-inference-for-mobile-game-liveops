"""Smoke test: ask the deployed Genie Agent its own curated sample questions.

Confirms the natural-language layer works end to end: each sample question goes
through the Genie Conversation API, Genie writes SQL against the
offer_performance_metrics metric view, and the answer, the generated SQL and the
first result rows are written as a markdown transcript.
Run locally once `bundle deploy` has created the Genie Agent:

    python jobs/smoke_genie.py <space-id> [genie_transcript.md]

Uses the CLI profile's auth (WorkspaceClient()). Writes to stdout when no output
path is given. Not wired into the DAB pipeline; a manual post-deploy verification.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage, MessageStatus

from offer_recommender.logging_config import get_logger

logger = get_logger(__name__)

# Rows shown per answer: enough to check the result shape while keeping the
# transcript short.
MAX_ROWS = 5


def _sample_questions(ws: WorkspaceClient, space_id: str) -> list[str]:
    """Return the sample questions configured on the Genie Agent, in order."""
    space = ws.genie.get_space(space_id, include_serialized_space=True)
    config = json.loads(space.serialized_space or "{}").get("config", {})
    return [q["question"][0] for q in config.get("sample_questions", [])]


def markdown_table(
    columns: Sequence[str], rows: Sequence[Sequence[object]]
) -> list[str]:
    """Render result rows as a markdown table."""
    header = [f"| {' | '.join(columns)} |", f"|{'---|' * len(columns)}"]
    return header + [
        f"| {' | '.join('' if v is None else str(v) for v in row)} |" for row in rows
    ]


def _render(ws: WorkspaceClient, message: GenieMessage) -> list[str]:
    """Render one Genie answer: text reply, generated SQL and leading result rows."""
    lines = [f"### {message.content}", ""]
    if message.status != MessageStatus.COMPLETED:
        logger.error("Genie failed on %r: %s", message.content, message.error)
        return [*lines, f"FAILED: {message.status} {message.error}", ""]
    for attachment in message.attachments or []:
        if attachment.text and attachment.text.content:
            lines += [attachment.text.content, ""]
        if not (attachment.query and attachment.attachment_id):
            continue
        if attachment.query.description:
            lines += [attachment.query.description, ""]
        lines += ["```sql", attachment.query.query or "", "```", ""]
        response = ws.genie.get_message_attachment_query_result(
            message.space_id,
            message.conversation_id,
            message.message_id,
            attachment.attachment_id,
        ).statement_response
        if (
            response
            and response.manifest
            and response.manifest.schema
            and response.result
        ):
            schema = response.manifest.schema
            columns = [c.name or "" for c in schema.columns or []]
            rows = response.result.data_array or []
            lines += markdown_table(columns, rows[:MAX_ROWS])
            lines += [f"\n{len(rows)} rows returned.", ""]
    return lines


def main(space_id: str, out_path: str | None) -> None:
    """Ask every sample question and write the transcript to `out_path` or stdout."""
    ws = WorkspaceClient()
    questions = _sample_questions(ws, space_id)
    if not questions:
        raise RuntimeError(f"Genie Agent {space_id} has no sample questions")
    lines = [f"# Genie Agent transcript (space {space_id})", ""]
    for question in questions:
        logger.info("Asking: %s", question)
        lines += _render(ws, ws.genie.start_conversation_and_wait(space_id, question))
    report = "\n".join(lines)
    if out_path is None:
        sys.stdout.write(report)
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report)
    logger.info("Wrote %d answers to %s", len(questions), out_path)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python jobs/smoke_genie.py <space-id> [output.md]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
