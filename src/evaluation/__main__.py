"""Evaluation CLI."""

import asyncio
import click
from pathlib import Path
from dotenv import load_dotenv

from .evaluator import RetrievalEval

load_dotenv()


@click.command()
@click.option(
    "--queries",
    default="data/queries/queries.jsonl",
    help="Path to queries JSONL file",
)
@click.option(
    "--output",
    default="data/eval_results.jsonl",
    help="Path to save results",
)
def eval_command(queries: str, output: str):
    """Run retrieval evaluation."""
    async def run():
        results = await RetrievalEval.run(queries, output)
        click.echo(RetrievalEval.format_results(results))
        click.echo(f"\nResults saved to {output}")

    asyncio.run(run())


if __name__ == "__main__":
    eval_command()
